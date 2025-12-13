from trl import RLOOTrainer
import torch
import numpy as np
from trl.models.utils import disable_gradient_checkpointing
from trl.trainer.utils import pad
from trl.data_utils import (
    is_conversational,
)
from accelerate.utils import gather, gather_object

class CustomRLOO(RLOOTrainer):
    def __init__(
        self,
        model=None,
        reward_funcs=None,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        reward_processing_classes=None,
        callbacks=None,
        optimizers=(None,None),
        peft_config=None,
    ):

        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            )


    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]


        prompt_ids_list, completion_ids_list = self._generate(prompts)

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size


        forward_kwargs = {}

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        # When gradient checkpointing is enabled with use_reentrant=True (default), calling the model inside a
        # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
        # Temporarily disable checkpointing to avoid this warning during inference.
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            # Compute the per-token log probabilities for the current model
            old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.model,
                prompt_completion_ids,
                attention_mask,
                logits_to_keep,
                batch_size,
                num_images=0,
                **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
            )
            old_logps = (old_per_token_logps * completion_mask).sum(1)  # mask out padding and tokens after EOS

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=0,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=0,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text, strict=True):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                if isinstance(bootstrap, list):  # for VLM, the format might be [{"type": "text", "text": "..."}]
                    assert len(bootstrap) == 1 and bootstrap[0]["type"] == "text"
                    bootstrap = bootstrap[0]["text"]
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Apply reward clipping if specified
        if self.reward_clip_range:
            rewards = rewards.clamp(min=self.reward_clip_range[0], max=self.reward_clip_range[1])

        # Include the KL penalty in the reward
        if self.beta != 0.0:
            per_token_kl = old_per_token_logps - ref_per_token_logps
            # Apply sequence-level KL penalty to rewards (sum KL across tokens first, then apply to each sequence)
            kl = (per_token_kl * completion_mask).sum(-1)
            kl = gather(kl)  # rewards are gathered, so kl must be too
            rewards = rewards - self.beta * kl

        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        grouped_rewards = rewards.view(-1, num_generations)
        mean_grouped_rewards = grouped_rewards.mean(dim=1)
        if num_generations > 1:
            std_rewards = grouped_rewards.std(dim=1)
        else:  # this case doesn't occur during training, but could in eval when num_generations_eval=1
            std_rewards = torch.zeros_like(mean_grouped_rewards)
        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))

        alpha = old_per_token_logps.view(-1,num_generations,old_per_token_logps.size(-1)) # (num_prompts,num_generations,seq_len)
        alpha = alpha.sum(dim=-1) # log-likelihood sequence wise 
        alpha = torch.softmax(alpha, dim=1) # (num_prompts,num_generations)
        
        # RLOO advantages computation
        grouped_sum = (alpha*grouped_rewards).sum(dim=1, keepdim=True)  # (num_prompts, 1)
        if num_generations > 1:
            baselines = (grouped_sum - (alpha*grouped_rewards)) / (num_generations - 1)  # (num_prompts, num_generations)
            baselines = baselines.view(-1)  # Flatten back to match rewards shape
            advantages = rewards - baselines
        else:  # this case doesn't occur during training, but could in eval when num_generations_eval=1
            advantages = torch.zeros_like(rewards)

        # Normalize advantages
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Calculate and log the mean KL divergence between current and reference model
        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            #std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            #self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())


        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_logps": old_logps,
            "advantages": advantages,
        }
        return output

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        logps = (per_token_logps * completion_mask).sum(1)  # mask out padding and tokens after EOS
        old_logps = inputs["old_logps"]
        log_ratio = logps - old_logps

        # Compute the loss
        advantages = inputs["advantages"]
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_sequence_loss1 = coef_1 * advantages
        per_sequence_loss2 = coef_2 * advantages
        per_sequence_loss = -torch.min(per_sequence_loss1, per_sequence_loss2)
        loss = per_sequence_loss.mean()

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        # Entropy
        mean_entropy = (entropies * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
        is_region_clipped = is_low_clipped | is_high_clipped
        gathered_low_clip = self.accelerator.gather(is_low_clipped.float().mean())
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        #self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(is_high_clipped.float().mean())
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        #self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(is_region_clipped.float().mean())
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss


