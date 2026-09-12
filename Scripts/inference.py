# inference.py

from typing import Optional, Callable

import pandas as pd
import torch
import torch.nn as nn

from config.configs import Sampling_config, Model_config, Train_config
from Dataprocessing.batch_loader import BatchLoader
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
            u = torch.linspace(0.0, 1.0, self.config.num_sampling_steps + 1)
            t = 1.0 - torch.cos(0.5 * torch.pi * u)
            return t.clamp(self.t_eps, 1.0 - self.t_eps)

        raise ValueError(f"Unknown time_schedule: {self.config.time_schedule}")

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


    @torch.no_grad()
    def _step_euler(self, z, t_cur, t_next, tar_len, cond, cond_mask):
        dt = (t_next - t_cur)
        v = self._vector_field(z, t_cur.expand(z.shape[0]), tar_len, cond, cond_mask)
        return z + dt * v

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


        v_pred = self._vector_field(z, t_batch, tar_len, cond_seq, cond_seq_mask)

        # Euler update
        z_next = z_back + (t_next - t_back) * v_pred

        return z_next, v_pred


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

    @torch.no_grad()
    def decode_tokens(self, z, tar_len=None, cond=None, cond_mask=None) -> torch.Tensor:
        B = z.shape[0]
        t_one = torch.ones(B, device=self.device, dtype=z.dtype)
        _, logits = self.model(z, t_one, tar_len, cond, cond_mask, decoder_step_active=True)
        return logits

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
        noise_scale = float(getattr(self.config, "denoiser_noise_scale", 1.0))
        z = torch.randn(
            batch_size, seq_length, latent_dim,
            generator=generator, device=self.device,
        ).to(self.device) * noise_scale

        t_span = self.get_sampling_steps(generator)

        z_clean = self.integrate(
            z, t_span, tar_len,
            cond=cond, cond_mask=cond_mask,# attn_mask=attn_mask,
            generator=generator,
        )

        logits = self.decode_tokens(z_clean, tar_len=tar_len, cond=cond, cond_mask=cond_mask)#, attn_mask=attn_mask)

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
        all_tokens = []

        for seed in range(seed_start, seed_end):
            print(seed)
            generator = torch.Generator(device=self.device).manual_seed(seed)

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
            num_samples=1, 
            return_probs=False,
    ):
        if greedy:
            print('greedy')
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

        logits = TCRFlowInference._apply_sampling_filters(
            logits,
            top_k=top_k,
            top_p=top_p,
            min_tokens_to_keep=min_tokens_to_keep
        )

        probs = torch.softmax(logits, dim=-1)
        B, L, V = probs.shape

        if torch.isinf(logits).all() or (probs.sum(dim=-1) == 0).any():
            result = logits.argmax(dim=-1)
            if return_probs:
                return result, torch.ones_like(result, dtype=torch.float32) / V
            return result


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

            result = idx.view(B, L, num_samples).transpose(1, 2)
            #print(result.shape)

        if return_probs:
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

        if top_k is not None and top_k > 0 and (top_p is None or top_p <= 0):
            return TCRFlowInference._apply_top_k(logits, top_k)

        if top_p is not None and 0 < top_p < 1 and (top_k is None or top_k <= 0):
            return TCRFlowInference._apply_top_p(logits, top_p, min_tokens_to_keep)

        if top_k is not None and top_k > 0 and top_p is not None and 0 < top_p < 1:
            logits = TCRFlowInference._apply_top_k(logits, top_k)
            logits = TCRFlowInference._apply_top_p(logits, top_p, min_tokens_to_keep)
            return logits

        return logits

    @staticmethod
    def _apply_top_k(logits, k):
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


        if repetition_penalty and past_tokens is not None:
            logits = TCRFlowInference._apply_repetition_penalty(
                logits,
                past_tokens,
                penalty=repetition_penalty_penalty
            )


        logits = TCRFlowInference._apply_sampling_filters(
            logits,
            top_k=top_k,
            top_p=top_p,
            min_tokens_to_keep=min_tokens_to_keep
        )


        probs = torch.softmax(logits, dim=-1)
        B, L, V = probs.shape

        if torch.isinf(logits).all() or (probs.sum(dim=-1) == 0).any():
            result = logits.argmax(dim=-1)
            if return_probs:
                return result, torch.ones_like(result, dtype=torch.float32) / V
            return result


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

        if penalty == 1.0:
            return logits

        unique_tokens = torch.unique(past_tokens)
        logits[..., unique_tokens] = logits[..., unique_tokens] / penalty
        return logits


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

    sampling_config = Sampling_config()#load_config(args.config)
    model_config = Model_config()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    from Model.TCRFlow_rope import TCRFlow             
    model = TCRFlow(model_config).to(device)

    ckpt = torch.load('.../TCRFlow/ckpt/model/best_model.pt', weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()



    sample_id = 0
    eval_loader = BatchLoader(".../TCRFlow/TrainingData/validation_eval_batches")

    n_seed = sampling_config.n_seed
    seed_start = random.randrange(42, 20260912)

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
        cond_emb       = eval_val_batch["cond"]     
        cond_seq_mask  = eval_val_batch["cond_seq_mask"]
        #target_attn    = eval_val_batch.get("attention_mask", None)

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
    flowed_results_df.to_csv('.../TCRFlow/inference/output.csv', index=False)




if __name__ == "__main__":
    main()
