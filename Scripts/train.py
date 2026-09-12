import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# 定义模型（假设同时支持denoiser和decoder两种模式）
'''class ELFModel(nn.Module):
    def __init__(self, self.config):
        super().__init__()
        self.self.config = self.config

        # 共享的主干网络
        self.shared_backbone = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
        )

        # Denoiser头：预测velocity（连续值）
        self.denoiser_head = nn.Linear(self.config.hidden_dim, self.config.latent_dim)

        # Decoder头：预测token logits（离散值）
        self.decoder_head = nn.Linear(self.config.hidden_dim, self.config.vocab_size)

    def forward(self, z, t, decoder_step_active=False, **kwargs):
        """
        前向传播
        Args:
            z: 输入latent (B, S, D)
            t: 时间步 (B, 1, 1)
            decoder_step_active: 是否处于decoder模式
        """
        # 共享的前向传播
        hidden = self.shared_backbone(z)

        if decoder_step_active:
            # Decoder模式：输出logits
            logits = self.decoder_head(hidden)
            # 模拟denoiser分支也需要的输出（实际未使用）
            v_pred = torch.zeros_like(z)
            return v_pred, logits
        else:
            # Denoiser模式：输出velocity
            v_pred = self.denoiser_head(hidden)
            logits = torch.zeros(z.shape[0], z.shape[1], self.self.config.vocab_size, device=z.device)
            return v_pred, logits'''

class Trainer:
    def __init__(
        self,
        model,
        logger,
        configs,
        save_dir = '/home/gaoletao/TCRFlow/ckpt/',
        #criterion,手动计算ce和mse loss
        device="cuda:0",
        save_model_name="model.pt",
        is_progress_bar=True,
        early_stopping=True,
        patience=10,
    ):

        self.early_stopping = early_stopping
        self.patience = patience
        self.device = device
        self.model = model.to(self.device)

        self.save_dir = save_dir
        self.save_model_path = os.path.join(self.save_dir, save_model_name)
        self.is_progress_bar = is_progress_bar
        self.logger = logger
        self.logger.info("Training Device: {}".format(self.device))

        self.config = configs

        self.best_val_loss = 10

        # 优化器
        if self.config.optimizer == "adamw":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, betas=(self.config.adam_b1, self.config.adam_b2), weight_decay=self.config.weight_decay)
        else:
            self.muon_params = [p for p in self.model.parameters() if p.requires_grad and p.ndim == 2]
            self.adamw_params = [p for p in self.model.parameters() if p.requires_grad and p.ndim != 2]
            self.optimizer = optim.Muon(self.muon_params, lr=self.config.muon_learning_rate, weight_decay=self.config.weight_decay)
            self.adam_optimizer = optim.Adam(self.adamw_params, lr=self.config.learning_rate, betas=(self.config.adam_b1, self.config.adam_b2), weight_decay=self.config.weight_decay)



    def train_step(
            self,
            batch_data: dict,
    ):
        """
            执行单个训练步（包含完整的伯努利采样流程）

            Args:
                model: ELF模型
                batch: 包含input_ids, attention_mask等数据的batch
                self.config: 训练配置
                optimizer: 优化器
                loss_mask: 损失掩码

            Returns:
                metrics: 训练指标字典
            """

        # 1. 伯努利采样：决定当前batch走哪个分支
        # 生成0-1之间的随机数
        bernoulli_sample = torch.rand(1).item()

        is_decoder_step = bernoulli_sample < self.config.decoder_prob

        attention_mask = batch_data["attention_mask"]
        target_len = batch_data["target_length"]
        cond = batch_data["cond"]
        cond_seq_mask = batch_data["cond_seq_mask"]
        decoder_target = batch_data['mapped_target_input_inds']
        loss_mask = attention_mask
        batch_size, seq_length = batch_data["mapped_target_input_inds"].shape[0], batch_data["mapped_target_input_inds"].shape[1]
        # 2. 前向传播和损失计算（根据采样结果）
        if is_decoder_step:
            # ========== Decoder分支 ==========
            # 生成decoder专用的噪声latent（t=1附近）

            # 使用logit-normal分布采样噪声强度


            decoder_z_vals = torch.randn(batch_size * seq_length, device=batch_data["mapped_target_input_inds"].device)

            decoder_z_vals = decoder_z_vals * self.config.decoder_p_std + self.config.decoder_p_mean
            decoder_lambda_t = torch.sigmoid(decoder_z_vals).reshape(batch_size, seq_length, 1)
            #print(f"decoder_lambda_t: min={decoder_lambda_t.min():.4f}, max={decoder_lambda_t.max():.4f}, "
            #      f"mean={decoder_lambda_t.mean():.4f}")

            # 添加噪声
            decoder_noise = torch.randn_like(batch_data["clean_x"]) * self.config.decoder_noise_scale
            decoder_z = decoder_lambda_t * batch_data["clean_x"] + (1 - decoder_lambda_t) * decoder_noise
            #print(f"decoder_z: min={decoder_z.min():.4f}, max={decoder_z.max():.4f}, "
            #      f"mean={decoder_z.mean():.4f}, std={decoder_z.std():.4f}")

            decoder_input = decoder_z#F.layer_norm(decoder_z, normalized_shape=[decoder_z.shape[-1]])#decoder_z

            # 固定t=1
            decoder_t = torch.ones(batch_size, device=batch_data["mapped_target_input_inds"].device)

            # 模型前向（decoder模式）
            _, decoder_logits = self.model(
                decoder_input, decoder_t, target_len, cond, cond_seq_mask, attention_mask,
                decoder_step_active=True,
            )





            # 计算交叉熵损失
            log_probs = F.log_softmax(decoder_logits.float(), dim=-1)
            '''print(f"dtype={decoder_target.dtype}, shape={decoder_target.shape}, "
                  f"min={decoder_target.min().item()}, max={decoder_target.max().item()}, "
                  f"vocab_size={23}, "
                  f"log_probs.shape={log_probs.shape}")'''

            ce_loss = -torch.gather(log_probs, -1, decoder_target.unsqueeze(-1)).squeeze(-1)

            # 应用损失掩码并取平均
            masked_ce = (ce_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            loss = masked_ce
            l2_loss_val = torch.tensor(0.0, device=batch_data["mapped_target_input_inds"].device)
            ce_loss_val = masked_ce

        else:
            # ========== Denoiser分支 ==========
            # 采样随机时间步t
            # 使用logit-normal分布（与原始代码一致）
            t = torch.randn(batch_size, device=batch_data["mapped_target_input_inds"].device)
            t = t * self.config.denoiser_p_std + self.config.denoiser_p_mean
            t_expand = torch.sigmoid(t).reshape(batch_size, 1, 1)  # (B, 1, 1)

            # 添加噪声
            noise = torch.randn_like(batch_data["clean_x"]) * self.config.denoiser_noise_scale
            denoiser_z = t_expand * batch_data["clean_x"] + (1-t_expand) * noise#(1 - t_expand) * batch_data["clean_x"] + t_expand * noise

            # 计算velocity目标
            t_eps = self.config.t_eps
            v_target = (batch_data["clean_x"] - denoiser_z) / (1 - t_expand).clamp(min=t_eps)

            '''# 自条件化逻辑
                if self.config.self_cond_prob > 0:
                    # 生成自条件化掩码
                    use_self_cond_mask = (torch.rand(batch_size) < self.config.self_cond_prob).float()
                    use_self_cond_mask = use_self_cond_mask.reshape(-1, 1, 1)

                    # 第一次预测（无自条件化）
                    z_input_init = denoiser_z
                    net_out_init, _ = self.model(z_input_init, t, decoder_step_active=False)
                    _, x_pred_init = net_out_to_v_x(net_out_init, denoiser_z, t, t_eps)

                    # 用第一次预测作为自条件化输入
                    x_pred_cond = x_pred_init * use_self_cond_mask
                    denoiser_input = torch.cat([denoiser_z, x_pred_cond], dim=-1)
                else:
                    '''
            denoiser_input = denoiser_z

            # 模型前向（denoiser模式）
            x_pred, _ = self.model(denoiser_input, t, target_len, cond, cond_seq_mask, attention_mask, decoder_step_active=False)
            v_pred = (x_pred - denoiser_z) / (1 - t_expand).clamp(min=t_eps)
            # 计算L2损失
            per_dim_loss = (v_pred - v_target) ** 2
            l2_loss = (per_dim_loss.mean(dim=-1) * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            '''print('l2')
            print(l2_loss)'''
            loss = l2_loss
            l2_loss_val = l2_loss
            ce_loss_val = torch.tensor(0.0, device=batch_data["mapped_target_input_inds"].device)

        # 3. 梯度累积（关键部分）
        # 将损失除以累积步数，使得最终梯度是平均值
        loss = loss / self.config.grad_accum_steps

        # 反向传播（梯度会累加到参数的.grad属性中）
        loss.backward()

        # 4. 记录当前步的训练指标
        metrics = {
                "loss": loss.item() * self.config.grad_accum_steps,  # 恢复原始loss用于日志
                "l2_loss": l2_loss_val.item(),
                "ce_loss": ce_loss_val.item(),
                "branch": "decoder" if is_decoder_step else "denoiser",
            }

        return metrics


    def train(self, train_loader, val_loader=None):
        """完整的训练循环，包含伯努利采样和梯度累积"""

        # EMA模型（可选）
        '''ema_model = None
            if self.config.use_ema:
                ema_model = ELFModel(self.config).to(device)
                ema_model.load_state_dict(model.state_dict())'''

        # 梯度累积计数器
        self.optimizer.zero_grad()
        #self.adam_optimizer.zero_grad()

        for epoch in range(self.config.epochs):

            # Epoch级别的统计变量
            epoch_loss = 0.0
            epoch_l2_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_decoder_loss = 0.0
            epoch_denoiser_loss = 0.0
            decoder_steps = 0
            denoiser_steps = 0


            for step, batch in enumerate(train_loader):
                # 将数据移到GPU
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # 执行训练步（包含伯努利采样和反向传播）
                metrics = self.train_step(
                    batch_data=batch,
                )

                # 累加epoch统计
                epoch_loss += metrics["loss"]
                epoch_l2_loss += metrics["l2_loss"]
                epoch_ce_loss += metrics["ce_loss"]

                if metrics["branch"] == "decoder":
                    epoch_decoder_loss += metrics["loss"]
                    decoder_steps += 1
                else:
                    epoch_denoiser_loss += metrics["loss"]
                    denoiser_steps += 1

                # 梯度累积：每grad_accum_steps步更新一次参数
                if (step + 1) % self.config.grad_accum_steps == 0:
                    # 梯度裁剪（可选）
                    if self.config.grad_clip_norm > 0:
                        #torch.nn.utils.clip_grad_norm_(self.muon_params, self.config.grad_clip_norm)
                        #torch.nn.utils.clip_grad_norm_(self.adamw_params, self.config.grad_clip_norm)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

                    # 更新参数
                    self.optimizer.step()
                    #self.adam_optimizer.step()

                    # 更新EMA
                    '''if self.config.use_ema:
                            for ema_param, param in zip(ema_model.parameters(), self.model.parameters()):
                                ema_param.data.mul_(self.config.ema_decay).add_(
                                    param.data, alpha=1 - self.config.ema_decay
                                )'''

                    # 清零梯度
                    self.optimizer.zero_grad()
                    #self.adam_optimizer.zero_grad()


            # ========== 计算Epoch级别的统计量 ==========
            num_batches = len(train_loader)
            avg_loss = epoch_loss / num_batches
            avg_l2_loss = epoch_l2_loss / num_batches
            avg_ce_loss = epoch_ce_loss / num_batches

            # 按分支统计（考虑采样概率的加权平均）
            decoder_prob = self.config.decoder_prob
            denoiser_prob = 1.0 - decoder_prob

            # 实际分支比例
            actual_decoder_ratio = decoder_steps / num_batches if num_batches > 0 else 0
            actual_denoiser_ratio = denoiser_steps / num_batches if num_batches > 0 else 0

            # 加权合并的loss（按照实际采样比例）
            if decoder_steps > 0 and denoiser_steps > 0:
                weighted_l2 = (avg_l2_loss * denoiser_steps + 0 * decoder_steps) / (
                        denoiser_steps + decoder_steps)
                weighted_ce = (0 * denoiser_steps + avg_ce_loss * decoder_steps) / (
                        denoiser_steps + decoder_steps)
            else:
                weighted_l2 = avg_l2_loss if denoiser_steps > 0 else 0
                weighted_ce = avg_ce_loss if decoder_steps > 0 else 0

            # 按期望概率重新缩放的loss（与ELF一致）
            expected_decoder_loss = epoch_decoder_loss / decoder_prob if decoder_steps > 0 and decoder_prob > 0 else 0
            expected_denoiser_loss = epoch_denoiser_loss / denoiser_prob if denoiser_steps > 0 and denoiser_prob > 0 else 0
            expected_total_loss = expected_decoder_loss + expected_denoiser_loss

            # 日志记录
            self.logger.info("=" * 70)
            self.logger.info(f"Epoch {epoch + 1}/{self.config.epochs} Summary:")
            self.logger.info(f"  Total Loss: {avg_loss:.6f}")
            self.logger.info(f"  L2 Loss (Denoiser): {avg_l2_loss:.6f}")
            self.logger.info(f"  CE Loss (Decoder): {avg_ce_loss:.6f}")
            self.logger.info(
                f"  Branch distribution: Decoder={decoder_steps}/{num_batches} ({actual_decoder_ratio * 100:.1f}%), "
                f"Denoiser={denoiser_steps}/{num_batches} ({actual_denoiser_ratio * 100:.1f}%)")
            self.logger.info(f"  Expected loss (scaled by prob): Total={expected_total_loss:.6f}, "
                            f"Decoder={expected_decoder_loss:.6f}, Denoiser={expected_denoiser_loss:.6f}")
            self.logger.info("=" * 70)

            # 可选：保存检查点
            if (epoch + 1) % self.config.save_freq == 0:
                checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                self.save_checkpoint(epoch, avg_loss, checkpoint_path)

            # 验证（如果提供验证集）
            if val_loader is not None:
                val_loss = self.validate(val_loader, epoch)
                self.logger.info(f"  Validation Loss: {val_loss:.6f}")

                # Early stopping
                if self.early_stopping:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        best_model = self.save_checkpoint(epoch, val_loss, os.path.join(self.save_dir, "best_model.pt"))
                        self.logger.info(f"  New best model saved! (val_loss: {val_loss:.6f})")
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.patience:
                            self.logger.info(f"Early stopping triggered after epoch {epoch + 1}")
                            break

        # 保存最终模型
        self.save_checkpoint(self.config.epochs, avg_loss, self.save_model_path)
        self.logger.info(f"Training completed! Final model saved to {self.save_model_path}")

        return self.model#, ema_model

    @torch.no_grad()
    def validate(self, val_loader, epoch):
        """验证函数"""
        self.model.eval()
        total_val_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 简化的验证：只使用denoiser分支
            attention_mask = batch["attention_mask"]
            target_len = batch["target_length"]
            cond = batch["cond"]
            cond_seq_mask = batch["cond_seq_mask"]
            loss_mask = attention_mask
            batch_size = batch["mapped_target_input_inds"].shape[0]

            # 采样时间步
            t = torch.randn(batch_size, device=self.device)
            t = t * self.config.denoiser_p_std + self.config.denoiser_p_mean
            t_expanded = torch.sigmoid(t).reshape(batch_size, 1, 1)

            # 添加噪声
            noise = torch.randn_like(batch["clean_x"])
            #denoiser_z = (1 - t_expanded) * batch["clean_x"] + t_expanded * noise
            denoiser_z = t_expanded * batch["clean_x"] + (1 - t_expanded) * noise

            # 计算velocity目标
            v_target = (batch["clean_x"] - denoiser_z) / (1 - t_expanded).clamp(min=self.config.t_eps)

            # 模型预测
            x_pred, _ = self.model(denoiser_z, t, target_len, cond, cond_seq_mask, attention_mask, decoder_step_active=False)
            v_pred = (x_pred - denoiser_z) / (1 - t_expanded).clamp(min=self.config.t_eps)

            # 计算损失
            per_dim_loss = (v_pred - v_target) ** 2
            l2_loss = (per_dim_loss.mean(dim=-1) * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

            total_val_loss += l2_loss.item()
            num_batches += 1

        self.model.train()
        return total_val_loss / num_batches if num_batches > 0 else 0

    def save_checkpoint(self, epoch, loss, path):
        """保存检查点"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
        }
        torch.save(checkpoint, path)
        return path
