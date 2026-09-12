class ReferenceGuidedSampler:
    def __init__(
            self,
            inference_engine: TCRFlowInference,
            reference_sequences: List[str],
            reference_latent: torch.Tensor,
            epitope_sequences: List[str],
            device: str = "cuda:0",
            token_to_id: Optional[Dict] = None,
            id_to_token: Optional[Dict] = None,
    ):
        assert len(reference_sequences) == len(epitope_sequences), \

        self.inference = inference_engine
        self.device = torch.device(device)
        self.reference_seqs = reference_sequences
        self.reference_latent = reference_latent
        self.epitope_seqs = epitope_sequences

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

        if tokens.dim() == 3:
            B, num_samples, L = tokens.shape
            tokens = tokens.reshape(B * num_samples, L)

        for row in tokens.tolist():
            seq = ""
            for token_id in row:
                token_id = int(token_id)
                if token_id == 0:
                    continue
                aa = self.id_to_token.get(token_id, 'X')
                seq += aa

            if not seq:
                seq = 'X' * (len(row) if hasattr(row, '__len__') else 1)

            out.append(seq)

        return out

    def _tokens_to_latent(self, tokens: torch.Tensor) -> torch.Tensor:
        model = self.inference.model

        if hasattr(model, 'token_embedding'):
            return model.token_embedding(tokens.to(self.device))
        elif hasattr(model, 'embed_tokens'):
            return model.embed_tokens(tokens.to(self.device))
        elif hasattr(model, 'tcr_embedding'):
            return model.tcr_embedding(tokens.to(self.device))
        else:
            raise AttributeError(
                "model did not supply token_embedding / embed_tokens / tcr_embedding，"
                "cannot process latent reverse. check model structure or use _alternative_noise_estimation."
            )

    @torch.no_grad()
    def reverse_map_to_noise(
            self,
            x_latent: torch.Tensor,
            cond: torch.Tensor,
            cond_mask: torch.Tensor,
            tar_len: torch.Tensor,
            num_reverse_steps: int = 50,
    ) -> torch.Tensor:
        t_span = torch.linspace(
            1.0 - self.inference.t_eps,
            self.inference.t_eps,
            num_reverse_steps + 1,
            device=self.device
        )

        z = x_latent.clone()
        B = z.shape[0]

        for i in range(num_reverse_steps):
            t_cur = t_span[i].expand(B)
            t_next = t_span[i + 1].expand(B)

            z = self.inference._step_euler(
                z, t_cur, t_next, tar_len, cond, cond_mask
            )

        return z

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
            num_samples: int = 1,
    ) -> torch.Tensor:
        t_span = self.inference.get_sampling_steps(generator)
        z_clean = self.inference.integrate(
            z, t_span, tar_len,
            cond=cond, cond_mask=cond_mask,
            generator=generator,
        )

        logits = self.inference.decode_tokens(
            z_clean, tar_len=tar_len,
            cond=cond, cond_mask=cond_mask
        )

        tokens = self.inference._logits_to_tokens(
            logits,
            greedy=False,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
            num_samples=num_samples,
        )

        return tokens

    @torch.no_grad()
    def verify_roundtrip(
            self,
            ref_tokens: torch.Tensor,
            cond: torch.Tensor,
            cond_mask: torch.Tensor,
            tar_len: torch.Tensor,
            num_steps: int = 50,
    ) -> Dict[str, float]:
        ref_latent = self._tokens_to_latent(ref_tokens)

        z_noise = self.reverse_map_to_noise(
            ref_latent, cond, cond_mask, tar_len,
            num_reverse_steps=num_steps
        )

        recon_tokens = self._forward_sample(
            z_noise, cond, cond_mask, tar_len,
            generator=None, num_steps=num_steps,
            temperature=1e-4,  # 近似 argmax 保证确定性
            top_k=1, top_p=None,
            num_samples=1,
        )

        match = (recon_tokens == ref_tokens.to(self.device)).float()
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
            temperature: float = 1.0,
            top_k: Optional[int] = 5,
            top_p: Optional[float] = 0.9,
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
            print(f"Processing reference {ref_idx + 1}/{len(self.reference_seqs)}: {ref_seq}")

            ref_tensor = torch.tensor([ref_tokens], device=self.device).repeat(batch_size, 1)

            if use_alternative:
                noise_seed = self._alternative_noise_estimation(
                    ref_tensor, cond[:batch_size], tar_len[:batch_size]
                )
            else:
                try:
                    ref_latent = self._tokens_to_latent(ref_tensor)

                    noise_seed = self.reverse_map_to_noise(
                        ref_latent,
                        cond[:batch_size],
                        cond_mask[:batch_size] if cond_mask is not None else None,
                        tar_len[:batch_size],
                        num_reverse_steps,
                    )
                except AttributeError as e:
                    print(f"[WARN] {e}\n rollback to alternative")
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
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            num_samples=1,
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
                            "temperature": temperature,
                            "top_k": top_k,
                            "top_p": top_p,
                        })

        return all_sequences, all_metadata
