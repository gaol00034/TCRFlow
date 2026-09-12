# seed_noise_screening.py
"""
方案二：基于Reference集合的种子噪声筛选（修正版）
通过反向ODE将Reference序列映射到噪声空间，然后进行可控扰动生成局部变体
"""
import ast

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from collections import defaultdict
import pandas as pd
import random

from config.configs5 import Sampling_config, Model_config, Train_config
from Dataprocessing.batch_loader import BatchLoader


# =====================================================================
# Inference engine
# =====================================================================
class TCRFlowInference:
    def __init__(
            self,
            model: nn.Module,
            config,
            device='cuda:1',
    ):
        self.model = model.eval().to(device)
        self.config = config
        self.device = device
        self.t_eps = float(config.t_eps)

    # -----------------------------------------------------------------
    # Denoiser 调用
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _denoiser_call(self, z, t, tar_len, cond, cond_mask):
        """调用底层 denoiser，返回 (v, x_pred)"""
        x_pred,_ = self.model(z, t, tar_len, cond, cond_mask, decoder_step_active=False)
        B = z.shape[0]
        one_minus_t = (1.0 - t).clamp(min=self.t_eps).view(B, 1, 1)
        v = (x_pred - z) / one_minus_t
        return v, x_pred

    @torch.no_grad()
    def _vector_field(self, z, t, tar_len=None, cond=None, cond_mask=None) -> torch.Tensor:
        v, _ = self._denoiser_call(z, t, tar_len, cond, cond_mask)  # , attn_mask)
        return v

    # -----------------------------------------------------------------
    # 扩散系数
    # -----------------------------------------------------------------
    def _diffusion_g2(self, t: torch.Tensor, churn: float) -> torch.Tensor:
        """
        g^2(t): 应始终 >= 0
        若出现负值，说明 churn 参数或时间调度设置有误
        """
        g2 = 2.0 * churn * (1.0 - t)
        if (g2 < 0).any():
            # 记录警告而不是静默 clamp
            print(f"[WARN] g^2 出现负值，检查 churn={churn} 与 t 范围")
        return g2.clamp(min=0.0)

    # -----------------------------------------------------------------
    # Heun 预测-校正 SDE 单步
    # -----------------------------------------------------------------
    '''@torch.no_grad()
    def heun_sde_step(
            self,
            z: torch.Tensor,
            t_cur: torch.Tensor,
            t_next: torch.Tensor,
            tar_len: torch.Tensor,
            cond: torch.Tensor,
            cond_mask: torch.Tensor,
            churn: float = 0.0,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Heun 预测-校正 SDE 步进
        z_{t+dt} = z_t + 0.5*dt*(drift(t) + drift(t+dt)) + sqrt(g^2 * dt) * eps
        噪声项只加一次
        """
        dt = (t_next - t_cur).view(-1, 1, 1)

        # ---- Predictor ----
        v1, _ = self._denoiser_call(z, t_cur, tar_len, cond, cond_mask)
        g2 = self._diffusion_g2(t_cur, churn).view(-1, 1, 1)
        noise = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)
        diffusion = torch.sqrt(g2 * dt.abs()) * noise
        drift1 = v1  # 若含 score 校正项在此加

        # 预测点（仅用于 corrector 的 drift 估计，不返回）
        z_pred = z + dt * drift1 + diffusion

        # ---- Corrector ----
        v2, _ = self._denoiser_call(z_pred, t_next, tar_len, cond, cond_mask)
        drift2 = v2

        # 最终：漂移用梯形，噪声项只加一次
        z_next = z + 0.5 * dt * (drift1 + drift2) + diffusion
        return z_next'''

    # ------------------------------------------------------------------
    # ODE 求解器
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _step_euler(self, z, t_cur, t_next, tar_len, cond, cond_mask):
        dt = (t_next - t_cur)
        v = self._vector_field(z, t_cur, tar_len, cond, cond_mask)
        z_next = z + dt[0]*v
        return z_next


    # -----------------------------------------------------------------
    # ODE 反向积分：数据 -> 噪声
    # -----------------------------------------------------------------
    @torch.no_grad()
    def invert_to_noise(
            self,
            x_data: torch.Tensor,          # 参考样本在 latent 空间的表示 (t≈1)
            tar_len: torch.Tensor,
            cond: torch.Tensor,
            cond_mask: torch.Tensor,
            num_steps: int = 50,
    ) -> torch.Tensor:
        """
        Probability-Flow ODE 反向积分：从 t=1-eps 走到 t=t_eps
        采用 Heun (deterministic, churn=0) 保证可逆性
        """
        ts = torch.linspace(1.0 - self.t_eps, self.t_eps, num_steps + 1, device=self.device)
        '''z = torch.randn(self.config.num_sampling_steps - 1, device=self.device)
        z = z * self.config.denoiser_p_std + self.config.denoiser_p_mean
        mids, _ = torch.sort(torch.sigmoid(z))
        t_span = torch.cat([
            torch.tensor([1.0 - self.t_eps], device=self.device),
            mids,
            torch.tensor([self.t_eps], device=self.device),
        ])
        ts = torch.unique_consecutive(t_span)'''

        z = x_data.clone()
        B = z.shape[0]

        for i in range(num_steps):
            t_cur = ts[i].expand(B)
            t_next = ts[i + 1].expand(B)
            '''z = self.heun_sde_step(
                z, t_cur, t_next, tar_len, cond, cond_mask,
                churn=0.0, generator=None  # 反演必须 deterministic
            )'''
            # 使用 Euler 方法进行反向积分
            z = self._step_euler(
                z, t_cur, t_next, tar_len, cond, cond_mask
            )
        return z

    # -----------------------------------------------------------------
    # ODE 正向积分：噪声 -> 数据
    # -----------------------------------------------------------------
    @torch.no_grad()
    def sample_from_noise(
            self,
            z_noise: torch.Tensor,
            tar_len: torch.Tensor,
            cond: torch.Tensor,
            cond_mask: torch.Tensor,
            num_steps: int = 50,
            churn: float = 0.0,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """从噪声开始的正向 ODE/SDE 采样"""
        ts = torch.linspace(self.t_eps, 1.0 - self.t_eps,
                            num_steps + 1, device=self.device)
        '''z = torch.randn(self.config.num_sampling_steps - 1, device=self.device)
        z = z * self.config.denoiser_p_std + self.config.denoiser_p_mean
        mids, _ = torch.sort(torch.sigmoid(z))
        t_span = torch.cat([
            torch.tensor([1.0 - self.t_eps], device=self.device),
            mids,
            torch.tensor([self.t_eps], device=self.device),
        ])
        ts = torch.unique_consecutive(t_span)'''

        z = z_noise.clone()
        B = z.shape[0]

        for i in range(num_steps):
            t_cur = ts[i].expand(B)
            t_next = ts[i + 1].expand(B)
            '''z = self.heun_sde_step(
                z, t_cur, t_next, tar_len, cond, cond_mask,
                churn=churn, generator=generator
            )'''
            z = self._step_euler(
                z, t_cur, t_next, tar_len, cond, cond_mask
            )
        return z

    # -----------------------------------------------------------------
    # Token 采样（top-k / top-p / temperature）
    # -----------------------------------------------------------------
    @torch.no_grad()
    def sample_tokens(
            self,
            logits: torch.Tensor,             # (B, L, V)
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        logits = logits / max(temperature, 1e-6)

        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            topk_vals, _ = torch.topk(logits, k, dim=-1)
            thresh = topk_vals[..., -1:].expand_as(logits)
            logits = torch.where(logits < thresh,
                                 torch.full_like(logits, float('-inf')),
                                 logits)

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
            probs_sorted = torch.softmax(sorted_logits, dim=-1)
            cum = probs_sorted.cumsum(dim=-1)
            mask = cum > top_p
            mask[..., 0] = False  # 保底至少留一个 token
            sorted_logits = sorted_logits.masked_fill(mask, float('-inf'))
            logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)

        probs = torch.softmax(logits, dim=-1)

        # ---- 修正：判定「某个位置所有 vocab 都是 -inf」而非「整个 tensor 全是 inf」----
        all_inf_per_pos = torch.isinf(logits).all(dim=-1)          # (B, L)
        zero_prob_per_pos = (probs.sum(dim=-1) == 0)                # (B, L)
        if all_inf_per_pos.any() or zero_prob_per_pos.any():
            # 回退到均匀分布，避免 multinomial 崩溃
            probs = torch.where(
                zero_prob_per_pos.unsqueeze(-1).expand_as(probs),
                torch.full_like(probs, 1.0 / probs.size(-1)),
                probs,
            )

        B, L, V = probs.shape
        flat = probs.reshape(-1, V)
        tokens = torch.multinomial(flat, 1, generator=generator).reshape(B, L)
        return tokens


# =====================================================================
# Reference-Guided Sampler
# =====================================================================
class ReferenceGuidedSampler:
    """
    基于 Reference 引导的采样器
    1. 将 Reference 序列反向映射到噪声空间（ODE inversion）
    2. 在噪声空间中进行 SDEdit 风格的方差保持扰动
    3. 正向生成局部变体
    """

    def __init__(
            self,
            inference_engine: TCRFlowInference,
            reference_sequences: List[str],
            reference_latent: torch.Tensor,
            epitope_sequences: List[str],
            device= "cuda:0",
            token_to_id: Optional[Dict] = None,
            id_to_token: Optional[Dict] = None,
    ):
        assert len(reference_sequences) == len(epitope_sequences), \
            "reference 与 epitope 数量必须一致"

        self.inference = inference_engine
        self.device = device
        self.reference_seqs = reference_sequences
        self.reference_latent = reference_latent
        self.epitope_seqs = epitope_sequences

        # AA -> ID
        default_map = {
            'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
            'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14,
            'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
        }
        self.token_to_id = token_to_id or default_map
        self.id_to_token = id_to_token or {v: k for k, v in self.token_to_id.items()}

    # -----------------------------------------------------------------
    # 编码/解码
    # -----------------------------------------------------------------
    def _encode_sequence(self, seq: str) -> List[int]:
        return [self.token_to_id.get(aa, 0) for aa in seq]

    def _decode_sequences(self, tokens: torch.Tensor) -> List[str]:
        out = []
        for row in tokens.tolist():
            ind = row.index(22)
            r = row[1:ind]
            token = "".join(self.id_to_token.get(int(t), "X") for t in r if int(t) != 0)
            #index = token.index(22)
            #token = token[:index+1] + [0] * (len(token) - index - 1)
            out.append(token)
        return out

    # -----------------------------------------------------------------
    # 通过 model 的 token embedding 得到 latent（若模型提供）
    # -----------------------------------------------------------------
    def _tokens_to_latent(self, tokens: torch.Tensor) -> torch.Tensor:
        '''model = self.inference.model
        if hasattr(model, "token_embedding"):
            return model.token_embedding(tokens.to(self.device))
        elif hasattr(model, "embed_tokens"):
            return model.embed_tokens(tokens.to(self.device))
        else:
            raise AttributeError(
                "模型未提供 token_embedding / embed_tokens，无法进行 latent 反演。"
                "请使用 _alternative_noise_estimation 替代。"
            )'''

    # -----------------------------------------------------------------
    # 反向映射：Reference tokens -> noise seed
    # -----------------------------------------------------------------
    @torch.no_grad()
    def reverse_map_to_noise(
            self,
            x_latent: torch.Tensor,
            cond: torch.Tensor,
            cond_mask: torch.Tensor,
            tar_len: torch.Tensor,
            num_reverse_steps: int = 50,
    ) -> torch.Tensor:
        #x_latent = self._tokens_to_latent(ref_tokens)
        z_noise = self.inference.invert_to_noise(
            x_latent, tar_len, cond, cond_mask, num_steps=num_reverse_steps
        )
        return z_noise

    # -----------------------------------------------------------------
    # SDEdit 风格方差保持扰动
    # -----------------------------------------------------------------
    @staticmethod
    def _variance_preserving_perturb(
            z: torch.Tensor,
            scale: float,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        z' = sqrt(1 - s^2) * z + s * eps
        scale in [0, 1]：0 完全保留 reference，1 完全随机
        """
        s = float(np.clip(scale, 0.0, 1.0))
        eps = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)
        return np.sqrt(1.0 - s ** 2) * z + s * eps

    # -----------------------------------------------------------------
    # 替代方案：无 token embedding 时的噪声估计
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _alternative_noise_estimation(
            self,
            ref_tokens: torch.Tensor,
            cond: torch.Tensor,
            tar_len: torch.Tensor,
            latent_dim: int = 256,
    ) -> torch.Tensor:
        B = cond.shape[0]
        seq_len = ref_tokens.shape[1]
        z = torch.randn(B, seq_len, latent_dim, device=self.device)

        # 极简迭代：用中间时刻的 x_pred 反推
        for _ in range(10):
            t = torch.full((B,), 0.5, device=self.device)
            _, x_pred = self.inference._denoiser_call(z, t, tar_len, cond, None)
            z = z + 0.01 * (x_pred - z)
        return z

    # -----------------------------------------------------------------
    # 正向采样：z -> tokens
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _forward_sample(
            self,
            z: torch.Tensor,
            cond: torch.Tensor,
            cond_mask: torch.Tensor,
            tar_len: torch.Tensor,
            generator: Optional[torch.Generator] = None,
            num_steps: int = 50,
            temperature: float = 1.0,
            top_k: Optional[int] = 5,
            top_p: Optional[float] = 0.9,
    ) -> torch.Tensor:
        z_final = self.inference.sample_from_noise(
            z, tar_len, cond, cond_mask,
            num_steps=num_steps, churn=0.0, generator=generator,
        )
        # 通过模型 head 得到 logits
        model = self.inference.model
        B = z_final.shape[0]
        t_one = torch.ones(B, device=self.device, dtype=z_final.dtype)
        _, logits = model(z_final, t_one, tar_len, cond, cond_mask, decoder_step_active=True)
        '''if hasattr(model, "to_logits"):
            logits = model.to_logits(z_final)
        elif hasattr(model, "output_head"):
            logits = model.output_head(z_final)
        else:
            raise AttributeError("模型未提供 to_logits / output_head")'''

        tokens = self.inference.sample_tokens(
            logits, temperature=temperature, top_k=top_k, top_p=top_p,
            generator=generator,
        )
        return tokens

    # -----------------------------------------------------------------
    # Round-trip 一致性验证
    # -----------------------------------------------------------------
    @torch.no_grad()
    def verify_roundtrip(
            self,
            ref_tokens: torch.Tensor,
            ref_lantent: torch.Tensor,
            cond: torch.Tensor,
            cond_mask: torch.Tensor,
            tar_len: torch.Tensor,
            num_steps: int = 50,
    ) -> Dict[str, float]:
        """
        验证 decode(invert(encode(x))) ≈ x
        返回 token 重合率
        """
        z_noise = self.reverse_map_to_noise(
            ref_lantent, cond, cond_mask, tar_len, num_reverse_steps=num_steps
        )
        recon_tokens = self._forward_sample(
            z_noise, cond, cond_mask, tar_len,
            generator=None, num_steps=num_steps,
            temperature=1e-4,  # 近似 argmax 保证确定性
            top_k=1, top_p=None,
        )
        result = recon_tokens.clone()
        for i in range(recon_tokens.shape[0]):
            row = recon_tokens[i]
            # 查找target在行中的位置
            mask = (row == 22)
            if mask.any():
                # 获取第一个target的位置
                pos = torch.nonzero(mask)[0, 0].item()
                # 将该位置之后的所有元素置0
                result[i, pos + 1:] = 0

        match = (result == ref_tokens.to(self.device)).float()
        return {
            "token_accuracy": match.mean().item(),
            "seq_exact_match": (match.mean(dim=-1) == 1.0).float().mean().item(),
        }

    # -----------------------------------------------------------------
    # 主流程：生成 reference-guided 变体
    # -----------------------------------------------------------------
    @torch.no_grad()
    def generate_variants(
            self,
            batch_data: Dict,
            batch_size: int = 8,
            perturbation_scales: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.3),
            num_variants_per_ref: int = 4,
            num_reverse_steps: int = 50,
            num_forward_steps: int = 50,
            use_alternative: bool = False,
    ) -> Tuple[List[str], List[Dict]]:

        cond = batch_data['cond'].to(self.device)
        cond_mask = batch_data.get('cond_mask')
        if cond_mask is not None:
            cond_mask = cond_mask.to(self.device)
        tar_len = batch_data['tar_len'].to(self.device)

        all_sequences: List[str] = []
        all_metadata: List[Dict] = []

        for ref_idx, ref_seq in enumerate(self.reference_seqs):
            ref_tokens = self._encode_sequence(ref_seq)
            ref_latent = self.reference_latent
            print(f"Processing reference {ref_idx + 1}/{len(self.reference_seqs)}: {ref_seq}")

            ref_tensor = torch.tensor([ref_tokens], device=self.device).repeat(batch_size, 1)

            # ---- 反向映射 ----
            if use_alternative:
                noise_seed = self._alternative_noise_estimation(
                    ref_tensor, cond[:batch_size], tar_len[:batch_size]
                )
            else:
                try:
                    noise_seed = self.reverse_map_to_noise(
                        ref_latent,#ref_tensor,
                        cond[:batch_size],
                        cond_mask[:batch_size] if cond_mask is not None else None,
                        tar_len[:batch_size],
                        num_reverse_steps,
                    )
                except AttributeError as e:
                    print(f"[WARN] {e}\n回退到 alternative 方案")
                    noise_seed = self._alternative_noise_estimation(
                        ref_tensor, cond[:batch_size], tar_len[:batch_size]
                    )

            # ---- 不同扰动强度 ----
            for scale in perturbation_scales:
                for variant_idx in range(num_variants_per_ref):
                    seed = ref_idx * 10000 + int(scale * 1000) * 10 + variant_idx
                    generator = torch.Generator(device=self.device).manual_seed(seed)

                    z_perturbed = self._variance_preserving_perturb(
                        noise_seed, scale=scale, generator=generator
                    )

                    try:
                        generated_tokens = self._forward_sample(
                            z_perturbed,
                            cond[:batch_size],
                            cond_mask[:batch_size] if cond_mask is not None else None,
                            tar_len[:batch_size],
                            generator=generator,
                            num_steps=num_forward_steps,
                        )
                        gen_seqs = self._decode_sequences(generated_tokens)
                    except Exception as e:
                        print(f"[ERROR] forward sampling failed: {e}")
                        continue

                    for b, seq in enumerate(gen_seqs):
                        all_sequences.append(seq)
                        all_metadata.append({
                            "ref_idx": ref_idx,
                            "ref_seq": ref_seq,
                            "epitope": self.epitope_seqs[ref_idx],
                            "scale": scale,
                            "variant_idx": variant_idx,
                            "batch_idx": b,
                            "seed": seed,
                        })

        return all_sequences, all_metadata


# =====================================================================
# 评估数据加载：按 epitope 组织
# =====================================================================
class EpitopeEvalDataset:
    """
    评估数据集：每个样本 = 一个 epitope + 其 reference CDR3β 集合
    """
    def __init__(
        self,
        data_df: pd.DataFrame,       # 需含列: epitope, cdr3b
        token_to_id: Dict,
        cond_encoder: Callable,       # 将 epitope 字符串 -> (cond, cond_mask)
        tcr_encoder: Callable,
        max_cdr3_len: int = 25,
        min_refs_per_epitope: int = 2,
    ):
        self.token_to_id = token_to_id
        self.cond_encoder = cond_encoder
        self.tcr_encoder = tcr_encoder
        self.max_cdr3_len = max_cdr3_len
        self.units = self._build_units(data_df, min_refs_per_epitope)

    def _encode_cdr3(self, seq: str) -> Tuple[List[int], int]:
        #seq = seq[: self.max_cdr3_len]
        latents, _ = self.tcr_encoder(seq)
        tokens = [self.token_to_id.get(a, 0) for a in seq[5:-5]]
        length = len(seq)-10+2
        tokens = [21]+tokens+[22] + [0] * (self.max_cdr3_len - length)

        return tokens, latents, length

    def _build_units(self, df: pd.DataFrame, min_refs: int) -> List[Dict]:
        units = []
        for ind, r in df.iterrows():#groupby("epitope"):
            epitope = '[PMHC]'+r['Epitope']+'[SEP]'+r['pseudo']+'[EOS]'
            refs = ast.literal_eval(r["references"])
            if len(refs) < min_refs:
                continue

            ref_tokens, ref_latent, ref_lens = [], [], []
            for r in refs:
                if len(r) <= 20:
                    tokens, latent, l = self._encode_cdr3('[TCR]' + r + '[EOS]')
                    ref_tokens.append(tokens)
                    ref_latent.append(latent)
                    ref_lens.append(l)


            cond, cond_mask = self.cond_encoder(epitope)  # 用户提供的编码器

            units.append({
                "epitope": epitope,
                "epitope_cond": cond,                            # (L_ep, D) or (L_ep,)
                "cond_mask": cond_mask,                          # (L_ep,)
                "references": refs,
                "ref_tokens": torch.tensor(ref_tokens), # (N_ref, L, )
                "ref_latent": torch.concat(ref_latent),  # (N_ref, L, D)
                "ref_lens": torch.tensor(ref_lens, dtype=torch.long),      # (N_ref,)
            })
        return units

    def __len__(self):
        return len(self.units)

    def __getitem__(self, idx):
        return self.units[idx]


# =====================================================================
# 单个 epitope 的变体生成
# =====================================================================
@torch.no_grad()
def generate_variants_for_epitope(
    sampler: "ReferenceGuidedSampler",
    eval_unit: Dict,
    device = 'cuda:0',
    perturbation_scales: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.3),
    variants_per_scale: int = 32,
    ref_chunk_size: int = 32,
    num_reverse_steps: int = 50,
    num_forward_steps: int = 50,
    use_alternative: bool = False,
) -> Tuple[Dict[float, List[str]], List[Dict]]:
    """
    对单个 epitope 生成变体
    返回:
        variants_by_scale: {scale: [变体序列, ...]}
        records: 每条变体的元数据列表
    """
    epitope = eval_unit["epitope"]
    ep_cond = eval_unit["epitope_cond"].to(device)
    cond_mask_1d = eval_unit["cond_mask"].to(device)
    ref_latent = eval_unit["ref_latent"].to(device)
    ref_lens = eval_unit["ref_lens"].to(device)
    references = eval_unit["references"]
    N_ref = ref_latent.size(0)

    print(f"[Epitope={epitope}] N_ref={N_ref}, scales={perturbation_scales}, "
          f"variants_per_scale={variants_per_scale}")

    # ---- 1. 反演所有 reference 到噪声空间 ----
    all_noise_seeds = []
    for st in range(0, N_ref, ref_chunk_size):
        ed = min(st + ref_chunk_size, N_ref)
        B = ed - st

        # cond 广播 (expand 不占额外显存)
        #if ep_cond.dim() == 2:  # (L_ep, D)
        cond_b = ep_cond.expand(B, -1, -1).contiguous()
        '''else:                   # (L_ep,) 需要模型内部再 embed
            cond_b = ep_cond.unsqueeze(0).expand(B, -1).contiguous()'''
        mask_b = cond_mask_1d.expand(B, -1).contiguous()

        chunk = ref_latent[st:ed]
        chunk_lens = ref_lens[st:ed]

        if use_alternative:
            seeds = sampler._alternative_noise_estimation(
                chunk, cond_b, chunk_lens
            )
        else:
            try:
                latent = chunk#sampler._tokens_to_latent(chunk)
                seeds = sampler.inference.invert_to_noise(
                    latent, chunk_lens, cond_b, mask_b,
                    num_steps=num_reverse_steps,
                )
                print()
            except AttributeError as e:
                print(f"  [WARN] {e} → fallback to alternative")
                seeds = sampler._alternative_noise_estimation(
                    chunk, cond_b, chunk_lens
                )
        all_noise_seeds.append(seeds)

    all_noise_seeds = torch.cat(all_noise_seeds, dim=0)  # (N_ref, L, D)

    # ---- 2. 每个 scale 下多次扰动生成 ----
    variants_by_scale: Dict[float, List[str]] = {s: [] for s in perturbation_scales}
    records: List[Dict] = []

    for scale in perturbation_scales:
        for v_idx in range(variants_per_scale):
            seed_val = hash((epitope, float(scale), v_idx)) & 0xFFFFFFFF
            gen = torch.Generator(device=device).manual_seed(seed_val)

            # 分块 forward 采样
            for st in range(0, N_ref, ref_chunk_size):
                ed = min(st + ref_chunk_size, N_ref)
                B = ed - st

                #if ep_cond.dim() == 2:
                cond_b = ep_cond.expand(B, -1, -1).contiguous()
                '''else:
                    cond_b = ep_cond.unsqueeze(0).expand(B, -1).contiguous()'''
                mask_b = cond_mask_1d.expand(B, -1).contiguous()

                z_pert = sampler._variance_preserving_perturb(
                    all_noise_seeds[st:ed], scale=scale, generator=gen
                )

                try:
                    tokens = sampler._forward_sample(
                        z_pert, cond_b, mask_b, ref_lens[st:ed],
                        generator=gen, num_steps=num_forward_steps,
                    )
                    seqs = sampler._decode_sequences(tokens)
                except Exception as e:
                    print(f"  [ERROR] forward failed @ scale={scale}, v={v_idx}: {e}")
                    continue

                variants_by_scale[scale].extend(seqs)
                for i, seq in enumerate(seqs):
                    records.append({
                        "epitope": epitope,
                        "scale": scale,
                        "variant_idx": v_idx,
                        "ref_idx": st + i,
                        "ref_seq": references[st + i],
                        "generated": seq,
                        "seed": seed_val,
                    })

    return variants_by_scale, records


# =====================================================================
# 简单评估指标（可按需扩展）
# =====================================================================
def _edit_distance(a: str, b: str) -> int:
    """经典 Levenshtein 距离"""
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr[j] = min(ins, dele, sub)
        prev = curr
    return prev[-1]


def compute_basic_metrics(
    variants_by_scale: Dict[float, List[str]],
    references: List[str],
) -> Dict[float, Dict[str, float]]:
    """计算每个 scale 下的基础统计指标"""
    metrics = {}
    ref_set = set(references)

    for scale, variants in variants_by_scale.items():
        if len(variants) == 0:
            metrics[scale] = {"n": 0}
            continue

        # 邻域一致性: 每个 variant 到最近 reference 的编辑距离
        min_dists = [
            min(_edit_distance(v, r) for r in references) for v in variants
        ]
        # 唯一性 / 多样性
        unique_variants = set(variants)
        # 与 reference 完全一致的比例（scale→0 应接近 1）
        exact_match = sum(1 for v in variants if v in ref_set) / len(variants)

        metrics[scale] = {
            "n": len(variants),
            "n_unique": len(unique_variants),
            "diversity": len(unique_variants) / len(variants),
            "mean_min_dist": float(np.mean(min_dists)),
            "median_min_dist": float(np.median(min_dists)),
            "exact_match_rate": exact_match,
        }
    return metrics




# =====================================================================
# 入口
# =====================================================================
def main_with_reference_guidance():
    print("=" * 60)
    print("Reference-Guided 评估")
    print("Pipeline: reference → 反演 → 扰动 → 正向生成 → 指标评估")
    print("=" * 60)

    # ---------- 配置 ----------
    sampling_config = Sampling_config()
    model_config = Model_config()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    perturbation_scales = (0.05, 0.1, 0.2, 0.3)
    variants_per_scale = 32
    ref_chunk_size = 32
    output_dir = "/home/gaoletao/TCRFlow/benchmark_inference/reference_guided"
    import os
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 加载模型 ----------
    from Model.TCRFlow_rope import TCRFlow
    model = TCRFlow(model_config).to(device)
    ckpt_path = '/home/gaoletao/TCRFlow/ckpt/models_0717_config5_benchmark/checkpoint_epoch_150.pt'  # 按你的 config 调整
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()

    inference = TCRFlowInference(model, sampling_config, device=device)




    # ---------- 准备 epitope 编码器 ----------
    # 你需要根据模型接口实现这个函数
    from TCRT5_encoder.tcrt5_encoder import tcrEncoder
    import pickle
    with open('/home/gaoletao/TCRFlow/Data/pmhc_mean.pkl', "rb") as f:
        pmhc_mean = pickle.load(f)
    with open('/home/gaoletao/TCRFlow/Data/pmhc_std.pkl', "rb") as af:
        pmhc_std = pickle.load(af)
    with open('/home/gaoletao/TCRFlow/Data/tcr_mean.pkl', "rb") as cf:
        tcr_mean = pickle.load(cf)
    with open('/home/gaoletao/TCRFlow/Data/tcr_std.pkl', "rb") as ef:
        tcr_std = pickle.load(ef)
    model_name = '/home/gaoletao/dkarthikeyan1/tcrt5_pre_tcrdb'
    pmhcencoder = tcrEncoder(enc_mean=pmhc_mean, enc_std=pmhc_std, model_name=model_name, device=device, max_seq_len=model_config.max_input_length)
    tcrencoder = tcrEncoder(enc_mean=tcr_mean, enc_std=tcr_std, model_name=model_name, device=device, max_seq_len=model_config.max_length)

    def cond_encoder(epitope: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将 epitope 字符串编码为 (cond, cond_mask)
        返回:
            cond: (L_ep,) 或 (L_ep, D)
            cond_mask: (L_ep,)
        """
        # === 用你的实际编码逻辑替换 ===
        '''max_ep_len = getattr(model_config, "max_input_length", 15)
        aa_to_id = {  # 与模型训练时一致
            'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,
            'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,
        }
        ids = [aa_to_id.get(a, 0) for a in epitope[:max_ep_len]]
        length = len(ids)
        ids = ids + [0] * (max_ep_len - length)
        cond = torch.tensor(ids, dtype=torch.long)
        cond_mask = torch.zeros(max_ep_len, dtype=torch.long)
        cond_mask[:length] = 1'''
        cond,_, cond_mask = pmhcencoder.encode(epitope)
        return cond, cond_mask

    def tcr_encoder(cdr3b: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将 epitope 字符串编码为 (cond, cond_mask)
        返回:
            cond: (L_ep,) 或 (L_ep, D)
            cond_mask: (L_ep,)
        """
        # === 用你的实际编码逻辑替换 ===
        '''max_ep_len = getattr(model_config, "max_input_length", 15)
        aa_to_id = {  # 与模型训练时一致
            'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,
            'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,
        }
        ids = [aa_to_id.get(a, 0) for a in epitope[:max_ep_len]]
        length = len(ids)
        ids = ids + [0] * (max_ep_len - length)
        cond = torch.tensor(ids, dtype=torch.long)
        cond_mask = torch.zeros(max_ep_len, dtype=torch.long)
        cond_mask[:length] = 1'''
        latent_tcr,_, attention_mask = tcrencoder.encode(cdr3b)
        return latent_tcr, attention_mask

    # ---------- 加载评估数据 ----------
    # 假设你有一个 csv，含 epitope 和 cdr3b 两列
    eval_csv = '/home/gaoletao/TCRFlow/Data/test_benchmark.csv'
    df = pd.read_csv(eval_csv)
    print(f"Loaded {len(df)} (epitope, cdr3b) pairs, "
          f"{df['Epitope'].nunique()} unique epitopes")

    dataset = EpitopeEvalDataset(
        data_df=df,
        token_to_id={  # 与 cond_encoder 保持一致
            'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,
            'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,
        },
        cond_encoder=cond_encoder,
        tcr_encoder=tcr_encoder,
        max_cdr3_len=model_config.max_length,
        min_refs_per_epitope=2,
    )
    print(f"Built {len(dataset)} evaluation units (epitopes with ≥2 refs)")

    # ---------- 遍历所有 epitope ----------
    all_records: List[Dict] = []
    all_metrics: List[Dict] = []

    for idx in range(len(dataset)):
        eval_unit = dataset[idx]

        # ----- (可选) round-trip 一致性检查（仅第一个 epitope 上做） -----
        if idx == 0:
            print("\n--- Round-trip sanity check on first epitope ---")
            ref_t = eval_unit["ref_tokens"][:4].to(device)  # 前4条
            ref_l = eval_unit["ref_lens"][:4].to(device)
            ref_latent = eval_unit['ref_latent'][:4].to(device)
            ep_cond = eval_unit["epitope_cond"].to(device)
            cond_mask = eval_unit["cond_mask"].to(device)
            B = ref_t.size(0)
            #if ep_cond.dim() == 2:

            cond_b = ep_cond.expand(B, -1, -1).contiguous()
            '''else:
                cond_b = ep_cond.unsqueeze(0).expand(B, -1).contiguous()'''
            mask_b = cond_mask.expand(B, -1).contiguous()

            # 构造一个临时 sampler 用于 verify
            temp_sampler = ReferenceGuidedSampler(
                inference_engine=inference,
                reference_sequences=eval_unit["references"][:4],
                reference_latent=ref_latent,
                epitope_sequences=[eval_unit["epitope"]] * 4,
                device=device,
            )
            try:
                metrics_rt = temp_sampler.verify_roundtrip(
                    ref_t, ref_latent, cond_b, mask_b, ref_l, num_steps=sampling_config.num_sampling_steps
                )
                print(f"  token_acc={metrics_rt['token_accuracy']:.3f}, "
                      f"exact_match={metrics_rt['seq_exact_match']:.3f}")
                if metrics_rt["token_accuracy"] < 0.7:
                    print("  [WARN] Round-trip 一致性偏低，反演质量可能不足")
            except Exception as e:
                print(f"  [WARN] round-trip check skipped: {e}")

        # ----- 构造该 epitope 的 sampler -----
        sampler = ReferenceGuidedSampler(
            inference_engine=inference,
            reference_sequences=eval_unit["references"],
            reference_latent=eval_unit["ref_latent"],
            epitope_sequences=[eval_unit["epitope"]] * len(eval_unit["references"]),
            device=device,
        )

        # ----- 生成变体 -----
        try:
            variants_by_scale, records = generate_variants_for_epitope(
                sampler=sampler,
                eval_unit=eval_unit,
                device=device,
                perturbation_scales=perturbation_scales,
                variants_per_scale=variants_per_scale,
                ref_chunk_size=ref_chunk_size,
                num_reverse_steps=50,
                num_forward_steps=50,
            )
        except Exception as e:
            print(f"[ERROR] epitope={eval_unit['epitope']} failed: {e}")
            continue

        # ----- 指标 -----
        metrics = compute_basic_metrics(variants_by_scale, eval_unit["references"])
        print(f"  [Metrics] {eval_unit['epitope']}:")
        for s, m in metrics.items():
            print(f"    scale={s}: n={m.get('n',0)}, "
                  f"div={m.get('diversity',0):.3f}, "
                  f"mean_dist={m.get('mean_min_dist',0):.2f}, "
                  f"exact={m.get('exact_match_rate',0):.3f}")

        for s, m in metrics.items():
            all_metrics.append({"epitope": eval_unit["epitope"],
                                "scale": s, **m})
        all_records.extend(records)

        # ----- 每个 epitope 单独存变体明细 -----
        pd.DataFrame(records).to_csv(
            os.path.join(output_dir, f"variants_{eval_unit['epitope']}.csv"),
            index=False,
        )

    # ---------- 汇总输出 ----------
    records_df = pd.DataFrame(all_records)
    metrics_df = pd.DataFrame(all_metrics)

    records_df.to_csv(os.path.join(output_dir, "all_variants.csv"), index=False)
    metrics_df.to_csv(os.path.join(output_dir, "all_metrics.csv"), index=False)

    print("\n" + "=" * 60)
    print(f"Done. Total variants: {len(records_df)}")
    print("Aggregate metrics by scale:")
    print(metrics_df.groupby("scale")[
        ["diversity", "mean_min_dist", "exact_match_rate"]
    ].mean())
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main_with_reference_guidance()
