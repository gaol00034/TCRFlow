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
    @torch.no_grad()
    def _denoiser_call(self, z, t, tar_len, cond, cond_mask):
        x_pred,_ = self.model(z, t, tar_len, cond, cond_mask, decoder_step_active=False)
        B = z.shape[0]
        one_minus_t = (1.0 - t).clamp(min=self.t_eps).view(B, 1, 1)
        v = (x_pred - z) / one_minus_t
        return v, x_pred

    @torch.no_grad()
    def _vector_field(self, z, t, tar_len=None, cond=None, cond_mask=None) -> torch.Tensor:
        v, _ = self._denoiser_call(z, t, tar_len, cond, cond_mask)  # , attn_mask)
        return v

    def _diffusion_g2(self, t: torch.Tensor, churn: float) -> torch.Tensor:

        g2 = 2.0 * churn * (1.0 - t)
        if (g2 < 0).any():
            print(f"[WARN] g^2 appears negative value，check churn={churn} and t")
        return g2.clamp(min=0.0)
   
    @torch.no_grad()
    def _step_euler(self, z, t_cur, t_next, tar_len, cond, cond_mask):
        dt = (t_next - t_cur)
        v = self._vector_field(z, t_cur, tar_len, cond, cond_mask)
        z_next = z + dt[0]*v
        return z_next


    @torch.no_grad()
    def invert_to_noise(
            self,
            x_data: torch.Tensor,
            tar_len: torch.Tensor,
            cond: torch.Tensor,
            cond_mask: torch.Tensor,
            num_steps: int = 50,
    ) -> torch.Tensor:
        ts = torch.linspace(1.0 - self.t_eps, self.t_eps, num_steps + 1, device=self.device)
        z = x_data.clone()
        B = z.shape[0]

        for i in range(num_steps):
            t_cur = ts[i].expand(B)
            t_next = ts[i + 1].expand(B)
            z = self._step_euler(
                z, t_cur, t_next, tar_len, cond, cond_mask
            )
        return z

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
        ts = torch.linspace(self.t_eps, 1.0 - self.t_eps,
                            num_steps + 1, device=self.device)

        z = z_noise.clone()
        B = z.shape[0]

        for i in range(num_steps):
            t_cur = ts[i].expand(B)
            t_next = ts[i + 1].expand(B)
            z = self._step_euler(
                z, t_cur, t_next, tar_len, cond, cond_mask
            )
        return z

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
            mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, float('-inf'))
            logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        all_inf_per_pos = torch.isinf(logits).all(dim=-1)          # (B, L)
        zero_prob_per_pos = (probs.sum(dim=-1) == 0)                # (B, L)
        if all_inf_per_pos.any() or zero_prob_per_pos.any():
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

    @torch.no_grad()
    def reverse_map_to_noise(
            self,
            x_latent: torch.Tensor,
            cond: torch.Tensor,
            cond_mask: torch.Tensor,
            tar_len: torch.Tensor,
            num_reverse_steps: int = 50,
    ) -> torch.Tensor:
        z_noise = self.inference.invert_to_noise(
            x_latent, tar_len, cond, cond_mask, num_steps=num_reverse_steps
        )
        return z_noise
    @staticmethod
    def _variance_preserving_perturb(
            z: torch.Tensor,
            scale: float,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        s = float(np.clip(scale, 0.0, 1.0))
        eps = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)
        return np.sqrt(1.0 - s ** 2) * z + s * eps

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

        for _ in range(10):
            t = torch.full((B,), 0.5, device=self.device)
            _, x_pred = self.inference._denoiser_call(z, t, tar_len, cond, None)
            z = z + 0.01 * (x_pred - z)
        return z
        
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
        model = self.inference.model
        B = z_final.shape[0]
        t_one = torch.ones(B, device=self.device, dtype=z_final.dtype)
        _, logits = model(z_final, t_one, tar_len, cond, cond_mask, decoder_step_active=True)

        tokens = self.inference.sample_tokens(
            logits, temperature=temperature, top_k=top_k, top_p=top_p,
            generator=generator,
        )
        return tokens
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
        z_noise = self.reverse_map_to_noise(
            ref_lantent, cond, cond_mask, tar_len, num_reverse_steps=num_steps
        )
        recon_tokens = self._forward_sample(
            z_noise, cond, cond_mask, tar_len,
            generator=None, num_steps=num_steps,
            temperature=1e-4,
            top_k=1, top_p=None,
        )
        result = recon_tokens.clone()
        for i in range(recon_tokens.shape[0]):
            row = recon_tokens[i]
            mask = (row == 22)
            if mask.any():
                pos = torch.nonzero(mask)[0, 0].item()
                result[i, pos + 1:] = 0

        match = (result == ref_tokens.to(self.device)).float()
        return {
            "token_accuracy": match.mean().item(),
            "seq_exact_match": (match.mean(dim=-1) == 1.0).float().mean().item(),
        }

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
                    noise_seed = self._alternative_noise_estimation(
                        ref_tensor, cond[:batch_size], tar_len[:batch_size]
                    )
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
    epitope = eval_unit["epitope"]
    ep_cond = eval_unit["epitope_cond"].to(device)
    cond_mask_1d = eval_unit["cond_mask"].to(device)
    ref_latent = eval_unit["ref_latent"].to(device)
    ref_lens = eval_unit["ref_lens"].to(device)
    references = eval_unit["references"]
    N_ref = ref_latent.size(0)

    print(f"[Epitope={epitope}] N_ref={N_ref}, scales={perturbation_scales}, "
          f"variants_per_scale={variants_per_scale}")

    all_noise_seeds = []
    for st in range(0, N_ref, ref_chunk_size):
        ed = min(st + ref_chunk_size, N_ref)
        B = ed - st

        #if ep_cond.dim() == 2:  # (L_ep, D)
        cond_b = ep_cond.expand(B, -1, -1).contiguous()
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
    variants_by_scale: Dict[float, List[str]] = {s: [] for s in perturbation_scales}
    records: List[Dict] = []

    for scale in perturbation_scales:
        for v_idx in range(variants_per_scale):
            seed_val = hash((epitope, float(scale), v_idx)) & 0xFFFFFFFF
            gen = torch.Generator(device=device).manual_seed(seed_val)
            for st in range(0, N_ref, ref_chunk_size):
                ed = min(st + ref_chunk_size, N_ref)
                B = ed - st

                #if ep_cond.dim() == 2:
                cond_b = ep_cond.expand(B, -1, -1).contiguous()
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


def main_with_reference_guidance():

    all_records: List[Dict] = []
    all_metrics: List[Dict] = []

    for idx in range(len(dataset)):
        eval_unit = dataset[idx]


        sampler = ReferenceGuidedSampler(
            inference_engine=inference,
            reference_sequences=eval_unit["references"],
            reference_latent=eval_unit["ref_latent"],
            epitope_sequences=[eval_unit["epitope"]] * len(eval_unit["references"]),
            device=device,
        )

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

if __name__ == "__main__":
    main_with_reference_guidance()
