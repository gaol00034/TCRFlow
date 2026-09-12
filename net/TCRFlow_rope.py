# TCRFlow.py - 添加RoPE和RMSNorm技术模块（保持原有框架不变）

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange, repeat


# ========== 1. RMSNorm (移植自ELF) ==========
class RMSNorm(nn.Module):
    """RMS Normalization - 完全移植自ELF的RMSNorm"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


# ========== 2. RoPE辅助函数 (移植自ELF) ==========
def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.chunk(2, dim=-1)
    x = torch.cat((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


class RotaryEmbedding(nn.Module):
    """
    1D Rotary Position Embedding - 完全移植自ELF的TextRotaryEmbeddingFast
    支持empty tokens（用于mode tokens和time tokens）
    """

    def __init__(
            self,
            dim: int,
            max_seq_len: int = 512,
            theta: float = 10000.0,
            num_empty_tokens: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.num_empty_tokens = num_empty_tokens

        self._build_cache()

    def _build_cache(self):
        """预计算cos和sin表"""
        # 计算频率
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))

        # 位置序列
        positions = torch.arange(self.max_seq_len, dtype=torch.float32)

        # 计算频率矩阵
        freqs_matrix = torch.einsum('..., f -> ... f', positions, freqs)
        freqs_matrix = repeat(freqs_matrix, '... n -> ... (n r)', r=2)

        D = freqs_matrix.shape[-1]

        # 构建cos和sin表
        if self.num_empty_tokens > 0:
            empty_cos = torch.ones(self.num_empty_tokens, D)
            empty_sin = torch.zeros(self.num_empty_tokens, D)
            self.register_buffer('freqs_cos', torch.cat([empty_cos, torch.cos(freqs_matrix)], dim=0))
            self.register_buffer('freqs_sin', torch.cat([empty_sin, torch.sin(freqs_matrix)], dim=0))
        else:
            self.register_buffer('freqs_cos', torch.cos(freqs_matrix))
            self.register_buffer('freqs_sin', torch.sin(freqs_matrix))

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        应用RoPE旋转
        x: (B, H, L, D) or (B, L, D)
        """
        if x.dim() == 4:
            _, _, seq_len, _ = x.shape
        else:
            _, seq_len, _ = x.shape

        cos = self.freqs_cos[start_pos: (start_pos + seq_len)]
        sin = self.freqs_sin[start_pos: (start_pos + seq_len)]

        # 扩展到x的维度
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        return x * cos + rotate_half(x) * sin


# ========== 3. 带RoPE的CrossAttentionDecoderBlock ==========
class CrossAttentionDecoderBlock(nn.Module):
    """
    Cross-Attention Decoder Block with RoPE support
    包含：Self-Attention (with RoPE) → Cross-Attention (with RoPE) → FFN
    使用RMSNorm替代LayerNorm
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            dropout: float = 0.1,
            use_rope: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_rope = use_rope

        # Self-Attention（使用RMSNorm）
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.self_q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.self_k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.self_v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.self_out_proj = nn.Linear(hidden_size, hidden_size)

        # Cross-Attention（使用RMSNorm）
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.cross_q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.cross_k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.cross_v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.cross_out_proj = nn.Linear(hidden_size, hidden_size)

        # FFN（使用RMSNorm）
        self.norm3 = RMSNorm(hidden_size, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),  # ELF使用gelu
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

        # QK Norm（ELF风格）
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.cross_q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.cross_k_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, H) -> (B, num_heads, L, head_dim)"""
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _reshape_back(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        """(B, num_heads, L, head_dim) -> (B, L, H)"""
        return x.transpose(1, 2).contiguous().view(B, L, self.hidden_size)

    def forward(
            self,
            x: torch.Tensor,
            cond_h: torch.Tensor,
            self_attn_mask: Optional[torch.Tensor] = None,
            cross_attn_mask: Optional[torch.Tensor] = None,
            target_mask: Optional[torch.Tensor] = None,
            qk_norm: bool = True,
            rope_self: Optional[RotaryEmbedding] = None,
            self_start_pos: int = 0,
            cross_start_pos: int = 0,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        _, L_cond, _ = cond_h.shape

        # ========== Self-Attention with RoPE ==========
        residual = x
        x = self.norm1(x)

        q = self._reshape_for_attention(self.self_q_proj(x))
        k = self._reshape_for_attention(self.self_k_proj(x))
        v = self._reshape_for_attention(self.self_v_proj(x))

        # QK Norm
        if qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE to self-attention
        if self.use_rope and rope_self is not None:
            q = rope_self(q, start_pos=self_start_pos)
            k = rope_self(k, start_pos=self_start_pos)

        # Self-Attention计算
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self_attn_mask is not None:
            attn_weights = attn_weights + self_attn_mask

        # 应用 target_mask (padding mask)
        if target_mask is not None:
            # target_mask: (B, L) where 1=valid, 0=pad
            # 扩展为 (B, 1, L, L) 用于 attention
            target_mask_expanded = target_mask[:, None, None, :]  # (B, 1, 1, L)
            # 对于每个 query，只允许 attend 到 valid keys
            # 对于 causal mask，已经确保不会看到未来，这里再加 padding mask
            attn_weights = attn_weights.masked_fill(
                ~target_mask_expanded.bool(),
                float('-inf')
            )


        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)
        attn_out = self._reshape_back(attn_out, B, L)
        attn_out = self.self_out_proj(attn_out)

        x = residual + self.dropout(attn_out)

        # ========== Cross-Attention with RoPE ==========
        residual = x
        x = self.norm2(x)

        q = self._reshape_for_attention(self.cross_q_proj(x))
        k = self._reshape_for_attention(self.cross_k_proj(cond_h))
        v = self._reshape_for_attention(self.cross_v_proj(cond_h))

        # QK Norm
        q = self.cross_q_norm(q)
        k = self.cross_k_norm(k)


        # Cross-Attention计算
        cross_attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if cross_attn_mask is not None:
            cross_attn_weights = cross_attn_weights + cross_attn_mask

        cross_attn_weights = F.softmax(cross_attn_weights, dim=-1)
        cross_attn_weights = self.dropout(cross_attn_weights)

        cross_attn_out = torch.matmul(cross_attn_weights, v)
        cross_attn_out = self._reshape_back(cross_attn_out, B, L)
        cross_attn_out = self.cross_out_proj(cross_attn_out)

        x = residual + self.dropout(cross_attn_out)

        # ========== FFN ==========
        residual = x
        x = self.norm3(x)
        x = residual + self.dropout(self.ffn(x))

        return x


# ========== 4.1 TimestepEmbedder（完全移植自ELF） ==========
class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size

        self.mlp_0 = nn.Linear(frequency_embedding_size, hidden_size)
        self.mlp_2 = nn.Linear(hidden_size, hidden_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mlp_0.weight, std=0.02)
        nn.init.zeros_(self.mlp_0.bias)
        nn.init.normal_(self.mlp_2.weight, std=0.02)
        nn.init.zeros_(self.mlp_2.bias)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp_0(t_emb)
        t_emb = F.silu(t_emb)
        t_emb = self.mlp_2(t_emb)
        return t_emb

# ========== 4.2 TargetLengthEmbedder ==========
class LengthEmbedder(nn.Module):
    """将标量长度（如 15）嵌入为向量，与时间嵌入类似"""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, length: torch.Tensor) -> torch.Tensor:
        # length: (B,)  e.g., [15, 14, 16]
        # 使用与 Timestep 同样的正弦编码，但针对长度
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(length.device)
        args = length[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)


# ========== 5. BottleneckTextProj（ELF风格） ==========
class BottleneckTextProj(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_size: int,
            bottleneck_dim: int,
            use_bias_proj1: bool = False,
            use_bias_proj2: bool = True,
            init_scale: float = 0.02,
    ):
        super().__init__()
        self.proj1 = nn.Linear(input_dim, bottleneck_dim, bias=use_bias_proj1)
        self.proj2 = nn.Linear(bottleneck_dim, hidden_size, bias=use_bias_proj2)
        self._init_weights(init_scale)

    def _init_weights(self, scale: float):
        nn.init.normal_(self.proj1.weight, std=scale)
        nn.init.normal_(self.proj2.weight, std=scale)
        if self.proj1.bias is not None:
            nn.init.zeros_(self.proj1.bias)
        if self.proj2.bias is not None:
            nn.init.zeros_(self.proj2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj1(x)
        x = self.proj2(x)
        return x


# ========== 6. 主模型 TCRFlow（添加RoPE和RMSNorm） ==========
class TCRFlow(nn.Module):
    """
    融合ELF机制的Cross-Attention蛋白质序列生成模型
    添加了RoPE位置编码和RMSNorm归一化
    """

    def __init__(self, config):
        super().__init__()

        # 模型参数
        self.hidden_size = config.hidden_size
        self.target_encoder_dim = config.target_encoder_dim
        self.vocab_size = config.vocab_size
        self.num_model_mode_tokens = config.num_model_mode_tokens
        self.num_time_tokens = config.num_time_tokens
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.bottleneck_dim = config.bottleneck_dim
        self.use_rope = config.use_rope

        # ========== 条件路径 ==========
        if config.use_bottleneck_for_cond:
            self.cond_projection = BottleneckTextProj(
                input_dim=config.cond_encoder_dim,
                hidden_size=self.hidden_size,
                bottleneck_dim=self.bottleneck_dim,
            )
        else:
            self.cond_projection = nn.Linear(config.cond_encoder_dim, self.hidden_size)

        # ========== 目标路径 ==========
        if config.use_bottleneck_for_target:
            self.target_projection = BottleneckTextProj(
                input_dim=config.target_encoder_dim,
                hidden_size=self.hidden_size,
                bottleneck_dim=self.bottleneck_dim,
            )
        else:
            self.target_projection = nn.Linear(config.target_encoder_dim, self.hidden_size)

        # ========== Mode Tokens ==========
        self.mode_tokens = nn.Parameter(
            torch.randn(1, self.num_model_mode_tokens, self.hidden_size) * 0.02
        )

        # ========== Time Tokens（ELF风格） ==========
        self.t_emb_tokens = nn.Parameter(
            torch.randn(1, self.num_time_tokens, self.hidden_size) * 0.02
        )
        self.t_embedder = TimestepEmbedder(self.hidden_size)

        # ========== Length Tokens ==========
        self.length_embedder = LengthEmbedder(self.hidden_size)

        # ========== RoPE Embeddings（新增） ==========
        if self.use_rope:
            # Self-attention的RoPE（包含time和mode tokens作为empty tokens）
            total_prefix_tokens = self.num_time_tokens + self.num_model_mode_tokens
            self.rope_self = RotaryEmbedding(
                dim=self.hidden_size // self.num_heads,
                max_seq_len=config.max_length,
                num_empty_tokens=total_prefix_tokens,
            )
        else:
            # 如果不使用RoPE，使用绝对位置编码
            self.max_seq_len = config.max_seq_len + self.num_time_tokens + self.num_model_mode_tokens
            self.pos_embedding = nn.Embedding(self.max_seq_len, self.hidden_size, padding_idx=0)
            self.register_buffer(
                "position_ids",
                torch.arange(self.max_seq_len).unsqueeze(0)
            )

        # ========== Cross-Attention Decoder Blocks（使用新的带RoPE的Block） ==========
        self.decoder_blocks = nn.ModuleList([
            CrossAttentionDecoderBlock(
                self.hidden_size,
                self.num_heads,
                dropout=config.dropout,
                use_rope=self.use_rope,
            )
            for _ in range(self.num_layers)
        ])

        # ========== Final Layer ==========
        self.final_norm = RMSNorm(self.hidden_size, eps=1e-6)  # 改用RMSNorm
        self.final_proj = nn.Linear(self.hidden_size, self.target_encoder_dim)

        # ========== Decoder Head ==========
        self.proj_kernel = nn.Parameter(
            torch.randn(self.hidden_size, self.target_encoder_dim) * 0.02
        )
        self.proj_bias = nn.Parameter(torch.zeros(self.target_encoder_dim))
        self.unembed_kernel = nn.Parameter(
            torch.randn(self.target_encoder_dim, self.vocab_size) * 0.02
        )
        self.unembed_bias = nn.Parameter(torch.zeros(self.vocab_size))
        '''nn.init.normal_(self.proj_kernel, std=1.0 / math.sqrt(self.proj_kernel.shape[0]))
        nn.init.zeros_(self.proj_bias)

        # 输出层：零初始化（初始 logits=0，CE 从 log(V) 开始，最温和）
        nn.init.zeros_(self.unembed_kernel)
        nn.init.zeros_(self.unembed_bias)'''

    def forward(
            self,
            x_t: torch.Tensor,
            t: torch.Tensor,
            target_length: torch.Tensor,
            cond_emb: torch.Tensor,
            cond_mask: Optional[torch.Tensor] = None,
            target_mask: Optional[torch.Tensor] = None,
            decoder_step_active: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L_tgt, _ = x_t.shape
        L_cond = cond_emb.shape[1]
        device = x_t.device

        # ========== 1. 条件路径 ==========
        cond_h = self.cond_projection(cond_emb)

        # ========== 2. 目标路径 ==========
        target_h = self.target_projection(x_t)

        # ========== 3. Mode Tokens ==========
        model_mode_offset = 0
        if self.num_model_mode_tokens > 0:
            mode_tokens = self.mode_tokens.expand(B, -1, -1)
            if decoder_step_active is not True:
                mode_tokens = torch.zeros_like(mode_tokens)
            target_h = torch.cat([mode_tokens, target_h], dim=1)
            model_mode_offset = self.num_model_mode_tokens

        # ========== 4. Time Tokens（ELF风格前缀注入）+ Length Tokens ==========
        time_emb = self.t_embedder(t)
        len_emb = self.length_embedder(target_length)
        combined_emb = time_emb + len_emb
        time_tokens = self.t_emb_tokens.expand(B, -1, -1)
        time_tokens = time_tokens + combined_emb[:, None, :]
        target_h = torch.cat([time_tokens, target_h], dim=1)

        # ========== 5. 位置编码 ==========
        total_len = self.num_time_tokens + model_mode_offset + L_tgt

        if self.use_rope:
            # 使用RoPE时，不需要额外的位置编码
            # RoPE会在attention中处理，需要传递start_pos
            self_start_pos = 0  # time和mode tokens作为empty tokens不旋转
            cross_start_pos = 0
        else:
            # 绝对位置编码
            pos_ids = self.position_ids[:, :total_len].expand(B, -1)
            target_h = target_h + self.pos_embedding(pos_ids)
            self_start_pos = 0
            cross_start_pos = 0

        # 更新attention mask
        if target_mask is not None:
            time_mask = torch.ones(B, self.num_time_tokens, device=device, dtype=target_mask.dtype)
            mode_mask = torch.ones(B, model_mode_offset, device=device, dtype=target_mask.dtype)
            target_mask = torch.cat([time_mask, mode_mask, target_mask], dim=1)

        # ========== 6. 构建Attention Masks ==========
        self_attn_mask = self._build_causal_mask_with_mode(
            model_mode_offset, self.num_time_tokens, L_tgt, device
        )
        cross_attn_mask = self._build_cross_attn_mask(cond_mask, device, B, L_cond)

        # ========== 6. 构建Attention Masks ==========
        # 使用双向/full self-attention，不再添加 causal triangular mask。
        # target padding 仍由 target_mask 在 attention block 内处理。
        self_attn_mask = None

        cross_attn_mask = self._build_cross_attn_mask(
            cond_mask, device, B, L_cond
        )

        # ========== 7. Cross-Attention Decoder处理 ==========
        for block in self.decoder_blocks:
            if self.use_rope:
                target_h = block(
                    x=target_h,
                    cond_h=cond_h,
                    self_attn_mask=self_attn_mask,
                    cross_attn_mask=cross_attn_mask,
                    target_mask=target_mask,
                    rope_self=self.rope_self,
                    self_start_pos=self_start_pos,
                    cross_start_pos=cross_start_pos,
                )
            else:
                target_h = block(
                    x=target_h,
                    cond_h=cond_h,
                    self_attn_mask=self_attn_mask,
                    cross_attn_mask=cross_attn_mask,
                    target_mask=target_mask,
                )

        # ========== 8. 移除前缀 ==========
        target_h = target_h[:, self.num_time_tokens + model_mode_offset:, :]

        # ========== 9. Final Layer ==========
        target_h = self.final_norm(target_h)
        output = self.final_proj(target_h)

        # ========== 10. Decoder Head ==========
        decoder_logits = None
        if decoder_step_active is not None:
            if decoder_step_active:
                proj_out = F.gelu(target_h @ self.proj_kernel + self.proj_bias)
                decoder_logits = proj_out @ self.unembed_kernel + self.unembed_bias
            else:
                decoder_logits = torch.zeros(B, L_tgt, self.vocab_size, device=device)
        '''print(f"target_h: nan={torch.isnan(target_h).any().item()}, "
              f"max={target_h.abs().max().item():.2f}")
        print(f"logits:   nan={torch.isnan(decoder_logits).any().item()}, "
              f"max={decoder_logits.abs().max().item():.2f}")'''
        return output, decoder_logits

    def _build_causal_mask_with_mode(
            self,
            num_mode_tokens: int,
            num_time_tokens: int,
            seq_len: int,
            device: torch.device
    ) -> torch.Tensor:
        """构建causal mask（time和mode tokens可看到所有，target tokens因果）"""
        total_len = num_time_tokens + num_mode_tokens + seq_len
        mask = torch.zeros(total_len, total_len, device=device)

        # Target tokens部分：causal
        for i in range(num_time_tokens + num_mode_tokens, total_len):
            for j in range(i + 1, total_len):
                mask[i, j] = float('-inf')

        return mask

    def _build_cross_attn_mask(
            self,
            cond_mask: Optional[torch.Tensor],
            device: torch.device,
            B: int,
            L_cond: int
    ) -> Optional[torch.Tensor]:
        if cond_mask is None:
            return None
        mask = (1.0 - cond_mask.float()) * -1e9
        mask = mask[:, None, None, :]
        return mask


# ========== 7. Config类（添加use_rope选项） ==========
'''class Config:
    def __init__(self):
        # 模型参数
        self.hidden_size = 384
        self.target_encoder_dim = 256
        self.input_dim = 256
        self.bottleneck_dim = 128
        self.vocab_size = 21
        self.num_model_mode_tokens = 4
        self.num_time_tokens = 4
        self.num_layers = 6
        self.num_heads = 4
        self.dropout = 0.1
        self.max_seq_len = 30

        # 预训练embedding维度
        self.cond_encoder_dim = 256

        # 可选配置
        self.use_bottleneck_for_cond = True
        self.use_bottleneck_for_target = False

        # 新增：是否使用RoPE（默认启用）
        self.use_rope = True'''

'''from config.configs import Model_config
# ========== 使用示例 ==========
if __name__ == "__main__":
    config = Model_config()
    model = TCRFlow(config)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Using RoPE: {config.use_rope}")

    # 模拟数据
    batch_size = 4
    L_cond = config.max_input_length
    L_tgt = config.max_length

    x_t = torch.randn(batch_size, L_tgt, config.target_encoder_dim)
    t = torch.rand(batch_size)
    targetlen = torch.randn(batch_size)
    cond_emb = torch.randn(batch_size, L_cond, config.cond_encoder_dim)
    cond_mask = torch.ones((batch_size, L_cond))
    target_mask = torch.ones((batch_size, L_tgt))

    # Denoiser模式
    output, decoder_logits = model(x_t, t, targetlen, cond_emb, decoder_step_active=False, cond_mask=cond_mask, target_mask=target_mask)
    print(f"\nDenoiser模式:")
    print(f"  output (x_pred) shape: {output.shape}")

    # Decoder模式
    output, decoder_logits = model(x_t, t, targetlen, cond_emb, decoder_step_active=True, cond_mask=cond_mask, target_mask=target_mask)
    print(f"\nDecoder模式:")
    print(f"  output shape: {output.shape}, decoder_logits shape: {decoder_logits.shape}")'''