import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Trainer:
    def __init__(
        self,
        model,
        logger,
        configs,
        save_dir = '.../TCRFlow/ckpt/',
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
        
        bernoulli_sample = torch.rand(1).item()

        is_decoder_step = bernoulli_sample < self.config.decoder_prob

        attention_mask = batch_data["attention_mask"]
        target_len = batch_data["target_length"]
        cond = batch_data["cond"]
        cond_seq_mask = batch_data["cond_seq_mask"]
        decoder_target = batch_data['mapped_target_input_inds']
        loss_mask = attention_mask
        batch_size, seq_length = batch_data["mapped_target_input_inds"].shape[0], batch_data["mapped_target_input_inds"].shape[1]
        if is_decoder_step:
            decoder_z_vals = torch.randn(batch_size * seq_length, device=batch_data["mapped_target_input_inds"].device)

            decoder_z_vals = decoder_z_vals * self.config.decoder_p_std + self.config.decoder_p_mean
            decoder_lambda_t = torch.sigmoid(decoder_z_vals).reshape(batch_size, seq_length, 1)
            decoder_noise = torch.randn_like(batch_data["clean_x"]) * self.config.decoder_noise_scale
            decoder_z = decoder_lambda_t * batch_data["clean_x"] + (1 - decoder_lambda_t) * decoder_noise

            decoder_input = decoder_z#F.layer_norm(decoder_z, normalized_shape=[decoder_z.shape[-1]])#decoder_z

            decoder_t = torch.ones(batch_size, device=batch_data["mapped_target_input_inds"].device)

            _, decoder_logits = self.model(
                decoder_input, decoder_t, target_len, cond, cond_seq_mask, attention_mask,
                decoder_step_active=True,
            )


            log_probs = F.log_softmax(decoder_logits.float(), dim=-1)

            ce_loss = -torch.gather(log_probs, -1, decoder_target.unsqueeze(-1)).squeeze(-1)

            masked_ce = (ce_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            loss = masked_ce
            l2_loss_val = torch.tensor(0.0, device=batch_data["mapped_target_input_inds"].device)
            ce_loss_val = masked_ce

        else:
            t = torch.randn(batch_size, device=batch_data["mapped_target_input_inds"].device)
            t = t * self.config.denoiser_p_std + self.config.denoiser_p_mean
            t_expand = torch.sigmoid(t).reshape(batch_size, 1, 1)  # (B, 1, 1)

            noise = torch.randn_like(batch_data["clean_x"]) * self.config.denoiser_noise_scale
            denoiser_z = t_expand * batch_data["clean_x"] + (1-t_expand) * noise#(1 - t_expand) * batch_data["clean_x"] + t_expand * noise

            t_eps = self.config.t_eps
            v_target = (batch_data["clean_x"] - denoiser_z) / (1 - t_expand).clamp(min=t_eps)
            denoiser_input = denoiser_z

            x_pred, _ = self.model(denoiser_input, t, target_len, cond, cond_seq_mask, attention_mask, decoder_step_active=False)
            v_pred = (x_pred - denoiser_z) / (1 - t_expand).clamp(min=t_eps)

            per_dim_loss = (v_pred - v_target) ** 2
            l2_loss = (per_dim_loss.mean(dim=-1) * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

            loss = l2_loss
            l2_loss_val = l2_loss
            ce_loss_val = torch.tensor(0.0, device=batch_data["mapped_target_input_inds"].device)

        loss = loss / self.config.grad_accum_steps

        loss.backward()

        metrics = {
                "loss": loss.item() * self.config.grad_accum_steps,
                "l2_loss": l2_loss_val.item(),
                "ce_loss": ce_loss_val.item(),
                "branch": "decoder" if is_decoder_step else "denoiser",
            }

        return metrics


    def train(self, train_loader, val_loader=None):
        self.optimizer.zero_grad()

        for epoch in range(self.config.epochs):

            epoch_loss = 0.0
            epoch_l2_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_decoder_loss = 0.0
            epoch_denoiser_loss = 0.0
            decoder_steps = 0
            denoiser_steps = 0


            for step, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                metrics = self.train_step(
                    batch_data=batch,
                )

                epoch_loss += metrics["loss"]
                epoch_l2_loss += metrics["l2_loss"]
                epoch_ce_loss += metrics["ce_loss"]

                if metrics["branch"] == "decoder":
                    epoch_decoder_loss += metrics["loss"]
                    decoder_steps += 1
                else:
                    epoch_denoiser_loss += metrics["loss"]
                    denoiser_steps += 1

                if (step + 1) % self.config.grad_accum_steps == 0:
                    if self.config.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    #self.adam_optimizer.zero_grad()


            num_batches = len(train_loader)
            avg_loss = epoch_loss / num_batches
            avg_l2_loss = epoch_l2_loss / num_batches
            avg_ce_loss = epoch_ce_loss / num_batches

            decoder_prob = self.config.decoder_prob
            denoiser_prob = 1.0 - decoder_prob

            actual_decoder_ratio = decoder_steps / num_batches if num_batches > 0 else 0
            actual_denoiser_ratio = denoiser_steps / num_batches if num_batches > 0 else 0

            if decoder_steps > 0 and denoiser_steps > 0:
                weighted_l2 = (avg_l2_loss * denoiser_steps + 0 * decoder_steps) / (
                        denoiser_steps + decoder_steps)
                weighted_ce = (0 * denoiser_steps + avg_ce_loss * decoder_steps) / (
                        denoiser_steps + decoder_steps)
            else:
                weighted_l2 = avg_l2_loss if denoiser_steps > 0 else 0
                weighted_ce = avg_ce_loss if decoder_steps > 0 else 0

            expected_decoder_loss = epoch_decoder_loss / decoder_prob if decoder_steps > 0 and decoder_prob > 0 else 0
            expected_denoiser_loss = epoch_denoiser_loss / denoiser_prob if denoiser_steps > 0 and denoiser_prob > 0 else 0
            expected_total_loss = expected_decoder_loss + expected_denoiser_loss

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

            if (epoch + 1) % self.config.save_freq == 0:
                checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                self.save_checkpoint(epoch, avg_loss, checkpoint_path)

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

        self.save_checkpoint(self.config.epochs, avg_loss, self.save_model_path)
        self.logger.info(f"Training completed! Final model saved to {self.save_model_path}")

        return self.model

    @torch.no_grad()
    def validate(self, val_loader, epoch):
        self.model.eval()
        total_val_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            attention_mask = batch["attention_mask"]
            target_len = batch["target_length"]
            cond = batch["cond"]
            cond_seq_mask = batch["cond_seq_mask"]
            loss_mask = attention_mask
            batch_size = batch["mapped_target_input_inds"].shape[0]

            t = torch.randn(batch_size, device=self.device)
            t = t * self.config.denoiser_p_std + self.config.denoiser_p_mean
            t_expanded = torch.sigmoid(t).reshape(batch_size, 1, 1)

            noise = torch.randn_like(batch["clean_x"])
            denoiser_z = t_expanded * batch["clean_x"] + (1 - t_expanded) * noise

            v_target = (batch["clean_x"] - denoiser_z) / (1 - t_expanded).clamp(min=self.config.t_eps)

            x_pred, _ = self.model(denoiser_z, t, target_len, cond, cond_seq_mask, attention_mask, decoder_step_active=False)
            v_pred = (x_pred - denoiser_z) / (1 - t_expanded).clamp(min=self.config.t_eps)

            per_dim_loss = (v_pred - v_target) ** 2
            l2_loss = (per_dim_loss.mean(dim=-1) * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

            total_val_loss += l2_loss.item()
            num_batches += 1

        self.model.train()
        return total_val_loss / num_batches if num_batches > 0 else 0

    def save_checkpoint(self, epoch, loss, path):
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
