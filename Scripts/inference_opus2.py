# inference.py
"""
TCRFlow 推理（约定: t=0=noise, t=1=clean）

支持的求解器（通过 sampler 参数选择）:
  ODE: 'euler' | 'heun' | 'midpoint' | 'rk4'
  SDE: 'elf_sde'

"""

from typing import Optional, Callable

import pandas as pd
import torch
import torch.nn as nn

from config.configs5_greedy import Sampling_config, Model_config, Train_config
from Dataprocessing.batch_loader import BatchLoader
#from sampling import *
import random


# =====================================================================
# Inference engine
# =====================================================================
class TCRFlowInference:
    def __init__(
        self,
        model: nn.Module,
        config,
        device: torch.device = torch.device("cuda:0"),
    ):
        self.model = model.eval().to(device)
        self.config = config
        self.device = device
        self.t_eps = float(config.t_eps)

    # ------------------------------------------------------------------
    # 时间网格
    # ------------------------------------------------------------------
    def get_sampling_steps(
        self,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.config.time_schedule == "uniform":
            return torch.linspace(self.t_eps, 1.0 - self.t_eps, self.config.num_sampling_steps + 1)

        if self.config.time_schedule == "logit_normal":
            z = torch.randn(self.config.num_sampling_steps - 1, generator=generator, device=self.device)
            z = z * self.config.denoiser_p_std + self.config.denoiser_p_mean
            mids, _ = torch.sort(torch.sigmoid(z))
            t_span = torch.cat([
                torch.tensor([self.t_eps], device=self.device),
                mids,
                torch.tensor([1.0 - self.t_eps], device=self.device),
            ])
            return torch.unique_consecutive(t_span)

        if self.config.time_schedule == "cosine":
            # 末端更密集，对 1-t 奇点更友好
            u = torch.linspace(0.0, 1.0, self.config.num_sampling_steps + 1)
            t = 1.0 - torch.cos(0.5 * torch.pi * u)
            return t.clamp(self.t_eps, 1.0 - self.t_eps)

        raise ValueError(f"Unknown time_schedule: {self.config.time_schedule}")

    # ------------------------------------------------------------------
    # 模型前向: 返回 (v, x_pred) 同时提供，便于 score 计算
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _denoiser_call(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        tar_len=None,
        cond=None,
        cond_mask=None,
        #attn_mask=None,
    ):
        x_pred, _ = self.model(z, t, tar_len, cond, cond_mask, decoder_step_active=False)
        B = z.shape[0]
        one_minus_t = (1.0 - t).clamp(min=self.t_eps).view(B, 1, 1)
        v = (x_pred - z) / one_minus_t
        return v, x_pred

    @torch.no_grad()
    def _vector_field(self, z, t, tar_len=None, cond=None, cond_mask=None) -> torch.Tensor:
        v, _ = self._denoiser_call(z, t, tar_len, cond, cond_mask)#, attn_mask)
        return v

    '''@torch.no_grad()
    def _score(self, z, t, cond=None, cond_mask=None, attn_mask=None) -> torch.Tensor:
        """
        s(z,t) = -(z - t * x_pred) / (1 - t)^2
              = - noise_pred / (1 - t)
        """
        _, x_pred = self._denoiser_call(z, t, cond, cond_mask, attn_mask)
        B = z.shape[0]
        one_minus_t = (1.0 - t).clamp(min=self.t_eps).view(B, 1, 1)
        noise_pred = (z - t.view(B, 1, 1) * x_pred) / one_minus_t
        score = -noise_pred / one_minus_t
        return score, x_pred'''

    # ------------------------------------------------------------------
    # ODE 求解器
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _step_euler(self, z, t_cur, t_next, tar_len, cond, cond_mask):
        dt = (t_next - t_cur)
        v = self._vector_field(z, t_cur.expand(z.shape[0]), tar_len, cond, cond_mask)
        return z + dt * v

    '''@torch.no_grad()
    def _step_heun(self, z, t_cur, t_next, cond, attn_mask):
        B = z.shape[0]
        dt = (t_next - t_cur)
        v1 = self._vector_field(z, t_cur.expand(B), cond, attn_mask)
        z_pred = z + dt * v1
        # 终点 t_next 可能就是 1-ε，仍可调用
        v2 = self._vector_field(z_pred, t_next.expand(B), cond, attn_mask)
        return z + 0.5 * dt * (v1 + v2)

    @torch.no_grad()
    def _step_midpoint(self, z, t_cur, t_next, cond, attn_mask):
        B = z.shape[0]
        dt = (t_next - t_cur)
        t_mid = t_cur + 0.5 * dt
        v1 = self._vector_field(z, t_cur.expand(B), cond, attn_mask)
        z_mid = z + 0.5 * dt * v1
        v_mid = self._vector_field(z_mid, t_mid.expand(B), cond, attn_mask)
        return z + dt * v_mid

    @torch.no_grad()
    def _step_rk4(self, z, t_cur, t_next, cond, attn_mask):
        B = z.shape[0]
        dt = (t_next - t_cur)
        t_mid = t_cur + 0.5 * dt
        k1 = self._vector_field(z,                t_cur.expand(B),  cond, attn_mask)
        k2 = self._vector_field(z + 0.5*dt*k1,    t_mid.expand(B),  cond, attn_mask)
        k3 = self._vector_field(z + 0.5*dt*k2,    t_mid.expand(B),  cond, attn_mask)
        k4 = self._vector_field(z + dt*k3,        t_next.expand(B), cond, attn_mask)
        return z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)'''

    # ------------------------------------------------------------------
    # SDE 求解器
    #   ELF
    # ------------------------------------------------------------------
    def _restore_cond(self, z, cond_seq, cond_seq_mask):
        """
        Restore condition parts in latent space.
        cond_seq_mask: 1 for condition fixed, 0 for generated.
        """
        if cond_seq is None:
            return z
        return cond_seq * cond_seq_mask + z * (1 - cond_seq_mask)

    def _elf_sde_step(self, z, t, t_next, tar_len, cond_seq, cond_seq_mask):
        h = t_next - t
        alpha = torch.clip(1.0 - self.config.gamma * h, 0.0, 1.0)
        t_back = alpha * t

        # Noise injection
        eps = torch.randn_like(z) * self.config.denoiser_noise_scale
        z_back = alpha * z + (1.0 - alpha) * eps

        # Restore condition parts (e.g., known pixels/latents)
        if cond_seq is not None:
            z_back = self._restore_cond(z_back, cond_seq, cond_seq_mask)

        # Prepare time batch
        t_batch = t_back.expand(z.shape[0]) if torch.is_tensor(t_back) else torch.full((z.shape[0],), t_back,
                                                                                       device=z.device)
        '''# --- CFG with self-conditioning ---
        # Unconditional input (null condition)
        null_cond = torch.zeros_like(cond_seq) if cond_seq is not None else None

        # Model forward with and without condition
        if cfg_scale != 1.0:
            # Conditional forward
            v_cond, x_cond = model(z_back, t_batch, cond_seq, x_pred_prev)
            # Unconditional forward
            v_uncond, x_uncond = model(z_back, t_batch, null_cond, x_pred_prev)
            # CFG combination
            v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            x_pred = x_uncond + cfg_scale * (x_cond - x_uncond)
        else:
            v_pred, x_pred = model(z_back, t_batch, cond_seq, x_pred_prev)
            
        # Self-conditioning blending (if used)
        if self_cond_cfg_scale != 1.0:
            # Re-run with x_pred as self condition
            if cfg_scale != 1.0:
                v_self, x_self = model(z_back, t_batch, cond_seq, x_pred)
                v_unself, x_unself = model(z_back, t_batch, null_cond, x_pred)
                v_pred = v_unself + self_cond_cfg_scale * (v_self - v_unself)
                x_pred = x_unself + self_cond_cfg_scale * (x_self - x_unself)
            else:
                v_pred, x_pred = model(z_back, t_batch, cond_seq, x_pred)'''

        v_pred = self._vector_field(z, t_batch, tar_len, cond_seq, cond_seq_mask)

        # Euler update
        z_next = z_back + (t_next - t_back) * v_pred

        return z_next, v_pred

    '''# ------------------------------------------------------------------
    # SDE 求解器
    #   dz = [v + (g^2/2) * s] dt + g sqrt(dt) * eps
    #   g(t)^2 = churn * 2(1-t) / (t + eps)   (churn=η, 可调)
    # ------------------------------------------------------------------
    def _diffusion_g2(self, t: torch.Tensor, churn: float) -> torch.Tensor:
        return churn * 2.0 * (1.0 - t) / (t + self.t_eps)

    @torch.no_grad()
    def _step_sde_euler(self, z, t_cur, t_next, cond, attn_mask, churn, generator):
        B = z.shape[0]
        dt = (t_next - t_cur)
        score, _ = self._score(z, t_cur.expand(B), cond, attn_mask)
        v = self._vector_field(z, t_cur.expand(B), cond, attn_mask)
        g2 = self._diffusion_g2(t_cur, churn)
        drift = v + 0.5 * g2 * score
        noise = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)
        diffusion = torch.sqrt(g2.clamp(min=0.0) * dt.clamp(min=0.0)) * noise
        return z + dt * drift + diffusion

    @torch.no_grad()
    def _step_sde_heun(self, z, t_cur, t_next, cond, attn_mask, churn, generator):
        """
        Predictor (Euler-Maruyama) + Corrector (Heun on drift, 噪声不二次注入).
        """
        B = z.shape[0]
        dt = (t_next - t_cur)

        score1, _ = self._score(z, t_cur.expand(B), cond, attn_mask)
        v1 = self._vector_field(z, t_cur.expand(B), cond, attn_mask)
        g2_1 = self._diffusion_g2(t_cur, churn)
        drift1 = v1 + 0.5 * g2_1 * score1

        noise = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)
        diffusion = torch.sqrt(g2_1.clamp(min=0.0) * dt.clamp(min=0.0)) * noise
        z_pred = z + dt * drift1 + diffusion

        # corrector: 只对漂移项做梯形修正
        score2, _ = self._score(z_pred, t_next.expand(B), cond, attn_mask)
        v2 = self._vector_field(z_pred, t_next.expand(B), cond, attn_mask)
        g2_2 = self._diffusion_g2(t_next, churn)
        drift2 = v2 + 0.5 * g2_2 * score2

        return z + 0.5 * dt * (drift1 + drift2) + diffusion'''

    # ------------------------------------------------------------------
    # 调度：根据 sampler 名称选 step 函数
    # ------------------------------------------------------------------
    def _get_step_fn(self) -> Callable:
        table = {
            "ode_euler":     self._step_euler,
            "elf_sde":   self._elf_sde_step,
            #"heun":      self._step_heun,
            #"midpoint":  self._step_midpoint,
            #"rk4":       self._step_rk4,
            #"sde_euler": self._step_sde_euler,
            #"sde_heun":  self._step_sde_heun,
        }
        if self.config.sampling_method not in table:
            raise ValueError(
                f"Unknown sampler: {self.config.sampling_method}. "
                f"Choices: {list(table.keys())}"
            )
        return table[self.config.sampling_method]

    @staticmethod
    def _is_sde(sampler: str) -> bool:
        return sampler.startswith("elf")

    # ------------------------------------------------------------------
    # 积分主循环
    # ------------------------------------------------------------------
    @torch.no_grad()
    def integrate(
        self,
        z: torch.Tensor,
        t_span: torch.Tensor,
        tar_len: torch.Tensor,
        cond=None,
        cond_mask=None,
        #attn_mask=None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        step_fn = self._get_step_fn()
        is_sde = self._is_sde(self.config.sampling_method)
        t_span = t_span.to(self.device).to(z.dtype)

        for k in range(len(t_span) - 1):
            t_cur, t_next = t_span[k], t_span[k + 1]
            z = step_fn(z, t_cur, t_next, tar_len, cond, cond_mask)
        return z

    # ------------------------------------------------------------------
    # Decoder 头 (t=1)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode_tokens(self, z, tar_len=None, cond=None, cond_mask=None) -> torch.Tensor:
        B = z.shape[0]
        t_one = torch.ones(B, device=self.device, dtype=z.dtype)
        _, logits = self.model(z, t_one, tar_len, cond, cond_mask, decoder_step_active=True)
        return logits

    # ------------------------------------------------------------------
    # 端到端采样
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_length: int,
        latent_dim: int,
        tar_len=None,
        cond=None,
        cond_mask=None,
        #attn_mask=None,
        generator: Optional[torch.Generator] = None,
        return_latent: bool = False,
    ):
        # 1) t=0 初始噪声
        noise_scale = float(getattr(self.config, "denoiser_noise_scale", 1.0))
        z = torch.randn(
            batch_size, seq_length, latent_dim,
            generator=generator, device=self.device,
        ).to(self.device) * noise_scale

        # 2) 时间网格
        t_span = self.get_sampling_steps(generator)

        # 3) 积分
        z_clean = self.integrate(
            z, t_span, tar_len,
            cond=cond, cond_mask=cond_mask,# attn_mask=attn_mask,
            generator=generator,
        )

        # 4) 解码
        logits = self.decode_tokens(z_clean, tar_len=tar_len, cond=cond, cond_mask=cond_mask)#, attn_mask=attn_mask)

        # 5) token
        token_ids = self._logits_to_tokens(
            logits, greedy=self.config.greedy, temperature=self.config.temperature,
            top_k=self.config.top_k, top_p=self.config.top_p, num_samples=self.config.top_sampling_num,
            generator = generator,
        )
        return (token_ids, z_clean) if return_latent else token_ids

    @torch.no_grad()
    def sample_with_seed_range(
            self,
            tar_len,
            cond,
            cond_mask,
            seed_start: int = 0,
            seed_end: int = 4,
            batch_size:int =1024,
            seq_length:int =22,
            latent_dim:int =256,
    ):
        """在种子范围内采样"""
        all_tokens = []

        for seed in range(seed_start, seed_end):
            print(seed)
            generator = torch.Generator(device=self.device).manual_seed(seed)

            # 修改 sample 方法接受 generator
            token_ids = self.sample(
                batch_size=batch_size,
                tar_len=tar_len,
                cond=cond,
                cond_mask=cond_mask,
                generator=generator,
                seq_length=seq_length,
                latent_dim=latent_dim,
            )
            all_tokens.append(token_ids)

        return torch.cat(all_tokens, dim=0)

    # ------------------------------------------------------------------
    # logits → tokens
    # ------------------------------------------------------------------

    @staticmethod
    def _logits_to_tokens(
            logits,
            greedy=True,
            temperature=1.0,
            top_k=None,
            top_p=None,
            generator=None,
            min_tokens_to_keep=1,
            num_samples=1,  # 新增：采样次数
            return_probs=False,  # 新增：是否返回概率
    ):
        """
        将 logits 转换为 token IDs，支持多次采样

        Args:
            logits: (B, L, V) 模型输出的 logits
            greedy: 是否使用贪婪解码
            temperature: 温度参数 (0.0-2.0)
            top_k: Top-K 采样参数，None/0/False 表示不使用
            top_p: Top-P (Nucleus) 采样参数 (0.0-1.0)，None/0 表示不使用
            generator: torch.Generator 用于可重复采样
            min_tokens_to_keep: Top-P 采样时最少保留的 token 数量
            num_samples: 采样次数，>1 时返回多个样本
            return_probs: 是否返回每个 token 的概率

        Returns:
            如果 num_samples=1: (B, L) token IDs
            如果 num_samples>1: (B, num_samples, L) token IDs
            如果 return_probs=True: (tokens, probs)
        """
        # 1. Greedy 模式（忽略 num_samples）
        if greedy:
            print('greedy')
            result = logits.argmax(dim=-1)
            if return_probs:
                probs = torch.softmax(logits, dim=-1)
                max_probs = probs.max(dim=-1)[0]
                return result, max_probs
            return result

        # 2. 温度处理
        if temperature <= 0:
            result = logits.argmax(dim=-1)
            if return_probs:
                probs = torch.softmax(logits, dim=-1)
                max_probs = probs.max(dim=-1)[0]
                return result, max_probs
            return result
        logits = logits / temperature

        # 3. 应用过滤策略
        logits = TCRFlowInference._apply_sampling_filters(
            logits,
            top_k=top_k,
            top_p=top_p,
            min_tokens_to_keep=min_tokens_to_keep
        )

        # 4. 计算概率分布
        probs = torch.softmax(logits, dim=-1)
        B, L, V = probs.shape

        # 检查是否所有概率为 0
        if torch.isinf(logits).all() or (probs.sum(dim=-1) == 0).any():
            result = logits.argmax(dim=-1)
            if return_probs:
                return result, torch.ones_like(result, dtype=torch.float32) / V
            return result

        #print(num_samples)

        # 5. 多次采样
        if num_samples == 1:
            # 单次采样
            idx = torch.multinomial(
                probs.reshape(B * L, V),
                1,
                generator=generator
            )
            result = idx.view(B, L)
        else:
            # 多次采样
            # 方法1: 使用 torch.multinomial 的 num_samples 参数
            #for n in range(num_samples):
            idx = torch.multinomial(
                probs.reshape(B * L, V),
                num_samples,
                generator=generator,
                replacement=True  # 允许重复采样
            )

            # 重塑为 (B, num_samples, L)
            result = idx.view(B, L, num_samples).transpose(1, 2)
            #print(result.shape)

        # 6. 返回结果
        if return_probs:
            # 计算采样 token 的概率
            if num_samples == 1:
                sampled_probs = torch.gather(
                    probs.reshape(B * L, V),
                    -1,
                    result.reshape(B * L, 1)
                ).view(B, L)
            else:
                # (B, num_samples, L)
                sampled_probs = torch.gather(
                    probs.reshape(B * L, V).unsqueeze(1).expand(-1, num_samples, -1),
                    -1,
                    result.reshape(B * num_samples * L, 1)
                ).view(B, num_samples, L)
            return result, sampled_probs

        return result

    @staticmethod
    def _apply_sampling_filters(logits, top_k=None, top_p=None, min_tokens_to_keep=1):
        """应用 Top-K 和/或 Top-P 过滤"""

        # 模式1: 仅 Top-K
        if top_k is not None and top_k > 0 and (top_p is None or top_p <= 0):
            return TCRFlowInference._apply_top_k(logits, top_k)

        # 模式2: 仅 Top-P
        if top_p is not None and 0 < top_p < 1 and (top_k is None or top_k <= 0):
            return TCRFlowInference._apply_top_p(logits, top_p, min_tokens_to_keep)

        # 模式3: Top-K + Top-P 组合
        if top_k is not None and top_k > 0 and top_p is not None and 0 < top_p < 1:
            logits = TCRFlowInference._apply_top_k(logits, top_k)
            logits = TCRFlowInference._apply_top_p(logits, top_p, min_tokens_to_keep)
            return logits

        return logits

    @staticmethod
    def _apply_top_k(logits, k):
        """应用 Top-K 过滤"""
        k = min(k, logits.size(-1))
        v, _ = torch.topk(logits, k, dim=-1)
        thresh = v[..., -1:].expand_as(logits)
        return torch.where(
            logits < thresh,
            torch.full_like(logits, float("-inf")),
            logits
        )

    @staticmethod
    def _apply_top_p(logits, p, min_tokens_to_keep=1):
        """应用 Top-P (Nucleus) 过滤"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > p
        if min_tokens_to_keep > 0:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = False

        sorted_logits = sorted_logits.masked_fill(
            sorted_indices_to_remove,
            float("-inf")
        )

        return torch.zeros_like(logits).scatter(
            -1, sorted_indices, sorted_logits
        )

    @staticmethod
    def _logits_to_tokens_with_repetition(
            logits,
            greedy=True,
            temperature=1.0,
            top_k=None,
            top_p=None,
            generator=None,
            min_tokens_to_keep=1,
            num_samples=1,
            return_probs=False,
            repetition_penalty=None,
            repetition_penalty_penalty=1.0,
            past_tokens=None,
    ):
        """
        增强版本：支持重复惩罚和多次采样
        """
        if greedy:
            result = logits.argmax(dim=-1)
            if return_probs:
                probs = torch.softmax(logits, dim=-1)
                max_probs = probs.max(dim=-1)[0]
                return result, max_probs
            return result

        if temperature <= 0:
            result = logits.argmax(dim=-1)
            if return_probs:
                probs = torch.softmax(logits, dim=-1)
                max_probs = probs.max(dim=-1)[0]
                return result, max_probs
            return result

        logits = logits / temperature

        # 应用重复惩罚
        if repetition_penalty and past_tokens is not None:
            logits = TCRFlowInference._apply_repetition_penalty(
                logits,
                past_tokens,
                penalty=repetition_penalty_penalty
            )

        # 应用采样过滤
        logits = TCRFlowInference._apply_sampling_filters(
            logits,
            top_k=top_k,
            top_p=top_p,
            min_tokens_to_keep=min_tokens_to_keep
        )

        # 计算概率
        probs = torch.softmax(logits, dim=-1)
        B, L, V = probs.shape

        if torch.isinf(logits).all() or (probs.sum(dim=-1) == 0).any():
            result = logits.argmax(dim=-1)
            if return_probs:
                return result, torch.ones_like(result, dtype=torch.float32) / V
            return result

        # 多次采样
        if num_samples == 1:
            idx = torch.multinomial(
                probs.reshape(B * L, V),
                1,
                generator=generator
            )
            result = idx.view(B, L)
        else:
            idx = torch.multinomial(
                probs.reshape(B * L, V),
                num_samples,
                generator=generator,
                replacement=True
            )
            result = idx.view(B, num_samples, L)

        if return_probs:
            if num_samples == 1:
                sampled_probs = torch.gather(
                    probs.reshape(B * L, V),
                    -1,
                    result.reshape(B * L, 1)
                ).view(B, L)
            else:
                sampled_probs = torch.gather(
                    probs.reshape(B * L, V).unsqueeze(1).expand(-1, num_samples, -1),
                    -1,
                    result.reshape(B * num_samples * L, 1)
                ).view(B, num_samples, L)
            return result, sampled_probs

        return result

    @staticmethod
    def _apply_repetition_penalty(logits, past_tokens, penalty=1.0):
        """对已生成的 token 施加重复惩罚"""
        if penalty == 1.0:
            return logits

        unique_tokens = torch.unique(past_tokens)
        logits[..., unique_tokens] = logits[..., unique_tokens] / penalty
        return logits

    '''@staticmethod
    def _logits_to_tokens(
        logits, greedy=True, temperature=1.0, top_k=False, generator=None,
    ):
        if greedy:
            return logits.argmax(dim=-1)
        logits = logits / max(temperature, 1e-6)
        if top_k:
            v, _ = torch.topk(logits, top_k, dim=-1)
            thresh = v[..., -1:].expand_as(logits)
            logits = torch.where(logits < thresh,
                                 torch.full_like(logits, float("-inf")), logits)
        probs = torch.softmax(logits, dim=-1)
        B, L, V = probs.shape
        idx = torch.multinomial(probs.reshape(B * L, V), 1, generator=generator)
        return idx.view(B, L)'''


AAID = {1:'A', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'K', 10:'L',
        11:'M', 12:'N', 13:'P', 14:'Q', 15:'R', 16:'S', 17:'T', 18:'V', 19:'W', 20:'Y',
        }
def id_to_aa(pred_ids: torch.Tensor):
    pred_aas = []
    if pred_ids.ndim == 3:
        for ind in range(pred_ids.shape[0]):
            ids = pred_ids[ind]  # .tolist()
            rep_top_sampling_res = []
            for id in ids:
                t = ''
                for i in id:
                    if i.item() == 21 or i.item() == 0:
                        continue
                    elif i.item() == 22:
                        break
                    else:
                        t = t + AAID[i.item()]
                rep_top_sampling_res.append(t)
            pred_aas.append(list(set(rep_top_sampling_res)))
    elif pred_ids.ndim == 2:
        for ind in range(pred_ids.shape[0]):
            ids = pred_ids[ind].tolist()
            t = ''
            for i in ids:
                if i == 21 or i == 0:
                    continue
                elif i == 22:
                    break
                else:
                    t = t + AAID[i]
            pred_aas.append([t])
    return pred_aas


def main():
    '''import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint .pt 路径")
    parser.add_argument("--config", type=str, required=True, help="config 文件路径（与训练一致）")
    parser.add_argument("--output_dir", type=str, default="./inference_out")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--solver", type=str, default="dopri5",
                        choices=["dopri5", "rk4", "euler"])
    parser.add_argument("--use_sde", action="store_true")
    parser.add_argument("--sde_gamma", type=float, default=0.3)
    args = parser.parse_args()'''

    # ===== 1. 加载 config =====
    # 这里假设你工程里有一个加载 config 的入口；按需替换

    sampling_config = Sampling_config()#load_config(args.config)
    model_config = Model_config()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # ===== 2. 构建模型 + 加载权重 =====
    from Model.TCRFlow_rope import TCRFlow                    # 你给的模型文件
    model = TCRFlow(model_config).to(device)

    ckpt = torch.load('/home/gaoletao/TCRFlow/ckpt/models_0903_config5_unseen/best_model.pt', weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    '''if missing:
        print(f"[WARN] missing keys: {missing}")
    if unexpected:
        print(f"[WARN] unexpected keys: {unexpected}")'''
    model.eval()



    '''# ===== 3. 构建数据加载器（与 train 一致） =====
    from your_dataset_module import build_eval_loader   # ← 改成你工程
    eval_loader = build_eval_loader(config, batch_size=args.batch_size)

    # ===== 4. tokenizer（用于把 token id 解回字符串） =====
    from your_tokenizer_module import get_tokenizer     # ← 改成你工程
    tokenizer = get_tokenizer(config)
    pad_id = getattr(config, "pad_token_id", 0)
    eos_id = getattr(config, "eos_token_id", None)

    # ===== 5. 推理循环 =====
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "generated.jsonl")
    f_out = open(out_path, "w", encoding="utf-8")
    '''
    sample_id = 0
    eval_loader = BatchLoader("/home/gaoletao/TCRFlow/TrainingData/validation_eval_batches_len")

    n_seed = sampling_config.n_seed
    seed_start = random.randrange(42, 20260903)

    flowed_results = []

    sampler = TCRFlowInference(
        model=model,
        config=sampling_config,
        device=device,
    )

    for batch in eval_loader:

        eval_val_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        val_epitopes = eval_val_batch['Epitopes']
        #val_alleles = eval_val_batch['Alleles']
        tar_len = eval_val_batch['target_length']#torch.randint(10, 20, (len(val_epitopes), )).to(device)
        cond_emb       = eval_val_batch["cond"]                    # 训练时叫 cond
        cond_seq_mask  = eval_val_batch["cond_seq_mask"]
        #target_attn    = eval_val_batch.get("attention_mask", None)   # 可选

        seed_end = seed_start + n_seed

        pred_ids = sampler.sample_with_seed_range(batch_size=cond_emb.shape[0],
                                                  tar_len=tar_len,
                                                  cond=cond_emb,
                                                  cond_mask=cond_seq_mask,
                                                  seed_start = seed_start,
                                                  seed_end = seed_end,
                                                  seq_length=model_config.max_length,
                                                  latent_dim=model_config.target_encoder_dim,
                                                  )#(bs*valid_repeat_num, top_sampling_num, max_len)
                              #attn_mask=target_attn)
        pred_cdr3b = id_to_aa(pred_ids)#list of list, len: bs*valid_repeat_num, per element: a list of top_sampling_num cdrs
        val_epitopes = val_epitopes * n_seed

        flowed_results.extend([r for r in zip(pred_cdr3b, val_epitopes)])

        seed_start = seed_end

    flowed_results_df = pd.DataFrame(flowed_results, columns=['flowed_cdr3b', 'Epitope'])
    flowed_results_df.to_csv('/home/gaoletao/TCRFlow/unseen_topk_pmhcs_inference/models_0903_config5_flowed_len_greedy.csv', index=False)
        # 按 EOS 截断
        #if eos_id is not None:
        #    pred_ids = mask_after_eos(pred_ids, eos_token_id=eos_id, pad_token_id=pad_id)

    '''for i in range(pred_ids.shape[0]):
        ids = pred_ids[i].tolist()
        cdr3b = id_to_aa(ids)
        epitope = val_epitopes[i]
        flowed_cdrs_per_epitope[epitope].append(cdr3b)
        record = {
            "generated_ids": ids,
            "generated": text,
        }
        # 如果 batch 里有 reference / context，一并写出
        for k in ("input", "target", "epitope", "tcr"):
            if k in batch and isinstance(batch[k], (list, tuple)):
                record[k] = batch[k][i]
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        sample_id += 1

        if sample_id >= args.num_samples:
            break

    if sample_id >= args.num_samples:
        break

f_out.close()
print(f"[Done] saved {sample_id} samples to {out_path}")'''



if __name__ == "__main__":
    main()