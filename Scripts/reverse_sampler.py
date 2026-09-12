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
            device: str = "cuda:0",
            token_to_id: Optional[Dict] = None,
            id_to_token: Optional[Dict] = None,
    ):
        assert len(reference_sequences) == len(epitope_sequences), \
            "reference 与 epitope 数量必须一致"

        self.inference = inference_engine
        self.device = torch.device(device)
        self.reference_seqs = reference_sequences
        self.reference_latent = reference_latent
        self.epitope_seqs = epitope_sequences

        # AA -> ID 映射
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
        """将序列字符串编码为 token ID 列表"""
        return [self.token_to_id.get(aa, 0) for aa in seq]

    def _decode_sequences(self, tokens: torch.Tensor) -> List[str]:
        """
        将 token IDs 转换为序列字符串

        Args:
            tokens: (B, L) 或 (B, num_samples, L) 的 token ID 张量

        Returns:
            List[str]: 解码后的序列列表
        """
        out = []

        # 处理多维输入 (B, num_samples, L) 或 (B, L)
        if tokens.dim() == 3:
            # (B, num_samples, L) -> 展平为 (B*num_samples, L)
            B, num_samples, L = tokens.shape
            tokens = tokens.reshape(B * num_samples, L)

        # 遍历每个序列
        for row in tokens.tolist():
            # 构建序列，过滤掉 padding token (0)
            seq = ""
            for token_id in row:
                token_id = int(token_id)
                if token_id == 0:  # padding token，跳过
                    continue
                # 使用 id_to_token 映射，如果找不到则用 'X' 代替
                aa = self.id_to_token.get(token_id, 'X')
                seq += aa

            # 如果序列为空，使用 'X' 填充
            if not seq:
                seq = 'X' * (len(row) if hasattr(row, '__len__') else 1)

            out.append(seq)

        return out

    # -----------------------------------------------------------------
    # 通过 model 的 token embedding 得到 latent
    # -----------------------------------------------------------------
    def _tokens_to_latent(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        将 token IDs 映射到 latent 空间

        Args:
            tokens: (B, L) token ID 张量

        Returns:
            torch.Tensor: (B, L, latent_dim) latent 表示
        """
        model = self.inference.model

        # 尝试获取 token embedding 层
        if hasattr(model, 'token_embedding'):
            return model.token_embedding(tokens.to(self.device))
        elif hasattr(model, 'embed_tokens'):
            return model.embed_tokens(tokens.to(self.device))
        elif hasattr(model, 'tcr_embedding'):
            return model.tcr_embedding(tokens.to(self.device))
        else:
            raise AttributeError(
                "模型未提供 token_embedding / embed_tokens / tcr_embedding，"
                "无法进行 latent 反演。请检查模型结构或使用 _alternative_noise_estimation。"
            )

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
        """
        将参考序列的 latent 表示反向映射到噪声空间

        Args:
            x_latent: (B, L, latent_dim) 参考序列的 latent 表示
            cond: (B, cond_dim) 条件信息
            cond_mask: (B, cond_dim) 条件 mask
            tar_len: (B,) 目标序列长度
            num_reverse_steps: 反向积分步数

        Returns:
            torch.Tensor: (B, L, latent_dim) 噪声种子
        """
        # 注意：inference_opus2_benchmark.py 中 t=0=noise, t=1=clean
        # 反向映射是从 t=1-eps 到 t=eps
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

            # 使用 Euler 方法进行反向积分
            z = self.inference._step_euler(
                z, t_cur, t_next, tar_len, cond, cond_mask
            )

        return z

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
        """
        当模型没有 token embedding 时的备选方案
        使用随机噪声 + 少量迭代
        """
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
            num_samples: int = 1,
    ) -> torch.Tensor:
        """
        从噪声种子正向采样生成 token 序列

        Args:
            z: (B, L, latent_dim) 噪声种子
            cond: (B, cond_dim) 条件信息
            cond_mask: (B, cond_dim) 条件 mask
            tar_len: (B,) 目标序列长度
            generator: 随机数生成器
            num_steps: 正向积分步数
            temperature: 采样温度
            top_k: Top-K 采样参数
            top_p: Top-P 采样参数
            num_samples: 每个噪声种子的采样次数

        Returns:
            torch.Tensor: (B, L) 或 (B, num_samples, L) token IDs
        """
        # 正向积分
        t_span = self.inference.get_sampling_steps(generator)
        z_clean = self.inference.integrate(
            z, t_span, tar_len,
            cond=cond, cond_mask=cond_mask,
            generator=generator,
        )

        # 解码得到 logits
        logits = self.inference.decode_tokens(
            z_clean, tar_len=tar_len,
            cond=cond, cond_mask=cond_mask
        )

        # 采样 token
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

    # -----------------------------------------------------------------
    # Round-trip 一致性验证
    # -----------------------------------------------------------------
    @torch.no_grad()
    def verify_roundtrip(
            self,
            ref_tokens: torch.Tensor,
            cond: torch.Tensor,
            cond_mask: torch.Tensor,
            tar_len: torch.Tensor,
            num_steps: int = 50,
    ) -> Dict[str, float]:
        """
        验证 decode(invert(encode(x))) ≈ x
        返回 token 重合率
        """
        # 获取 latent 表示
        ref_latent = self._tokens_to_latent(ref_tokens)

        # 反向映射到噪声
        z_noise = self.reverse_map_to_noise(
            ref_latent, cond, cond_mask, tar_len,
            num_reverse_steps=num_steps
        )

        # 正向采样
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
            temperature: float = 1.0,
            top_k: Optional[int] = 5,
            top_p: Optional[float] = 0.9,
    ) -> Tuple[List[str], List[Dict]]:
        """
        生成 reference-guided 变体

        Args:
            batch_data: 包含 cond, cond_mask, tar_len 的字典
            batch_size: 每个 reference 的批次大小
            perturbation_scales: 扰动强度列表
            num_variants_per_ref: 每个扰动强度的变体数量
            num_reverse_steps: 反向积分步数
            num_forward_steps: 正向积分步数
            use_alternative: 是否使用备选噪声估计方法
            temperature: 采样温度
            top_k: Top-K 采样参数
            top_p: Top-P 采样参数

        Returns:
            Tuple[List[str], List[Dict]]: 生成的序列和元数据
        """
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

            # ---- 反向映射 ----
            if use_alternative:
                noise_seed = self._alternative_noise_estimation(
                    ref_tensor, cond[:batch_size], tar_len[:batch_size]
                )
            else:
                try:
                    # 获取 latent 表示
                    ref_latent = self._tokens_to_latent(ref_tensor)

                    noise_seed = self.reverse_map_to_noise(
                        ref_latent,
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

                    # 扰动噪声种子
                    z_perturbed = self._variance_preserving_perturb(
                        noise_seed, scale=scale, generator=generator
                    )

                    try:
                        # 正向采样
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

                        # 解码序列
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