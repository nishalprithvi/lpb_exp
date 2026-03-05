from typing import Dict
import torch
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator


class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_scheduler: DDPMScheduler,
        horizon,
        obs_dim,
        action_dim,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_local_cond=False,
        obs_as_global_cond=False,
        pred_action_steps_only=False,
        oa_step_convention=False,
        **kwargs,
    ):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention

        # Keep default behavior identical when guidance_mode='none'.
        self.guidance_mode = kwargs.pop("guidance_mode", "none")
        self.obstacle_guidance_scale = float(kwargs.pop("obstacle_guidance_scale", 0.0))
        self.obstacle_margin = float(kwargs.pop("obstacle_margin", 10.0))
        self.guidance_start_timestep = int(kwargs.pop("guidance_start_timestep", 10))
        self.obstacle_center = kwargs.pop("obstacle_center", None)
        self.obstacle_radius = kwargs.pop("obstacle_radius", None)
        self.kwargs = kwargs
        self.reset_guidance_debug()

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def reset_guidance_debug(self):
        self.guidance_debug = {
            "enabled": False,
            "total_steps": 0,
            "eligible_steps": 0,
            "applied_steps": 0,
            "none_grad_steps": 0,
            "zero_grad_steps": 0,
            "obstacle_loss_sum": 0.0,
            "obstacle_loss_count": 0,
            "grad_norm_sum": 0.0,
            "grad_norm_max": 0.0,
        }

    def get_guidance_debug_summary(self):
        d = dict(self.guidance_debug)
        loss_count = max(int(d.get("obstacle_loss_count", 0)), 1)
        applied = max(int(d.get("applied_steps", 0)), 1)
        eligible = max(int(d.get("eligible_steps", 0)), 1)
        d["obstacle_loss_mean"] = float(d.get("obstacle_loss_sum", 0.0) / loss_count)
        d["grad_norm_mean"] = float(d.get("grad_norm_sum", 0.0) / applied) if d.get("applied_steps", 0) > 0 else 0.0
        d["guidance_applied_ratio"] = float(d.get("applied_steps", 0) / eligible) if d.get("eligible_steps", 0) > 0 else 0.0
        return d

    def set_obstacle_guidance(self, center, radius, enabled=True):
        if center is None or radius is None or not enabled:
            self.obstacle_center = None
            self.obstacle_radius = None
            self.guidance_debug["enabled"] = False
            return
        self.obstacle_center = torch.as_tensor(center, dtype=torch.float32, device=self.device).view(1, 1, -1)
        self.obstacle_radius = torch.as_tensor(float(radius), dtype=torch.float32, device=self.device)
        self.guidance_debug["enabled"] = True

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]

            if self.guidance_mode == "obstacle" and self.obstacle_center is not None and self.obstacle_radius is not None:
                trajectory = trajectory.detach().requires_grad_()

            model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)

            if self.guidance_mode == "obstacle" and self.obstacle_center is not None and self.obstacle_radius is not None:
                self.guidance_debug["total_steps"] += 1
                if t < self.guidance_start_timestep:
                    self.guidance_debug["eligible_steps"] += 1
                    trajectory0 = scheduler.step(model_output, t, trajectory).pred_original_sample
                    action_traj = self.normalizer["action"].unnormalize(trajectory0[..., :self.action_dim])
                    center = self.obstacle_center.to(device=trajectory.device, dtype=trajectory.dtype)
                    radius = self.obstacle_radius.to(device=trajectory.device, dtype=trajectory.dtype)
                    center_dim = center.shape[-1]
                    action_pos = action_traj[..., :center_dim]
                    dist = torch.linalg.norm(action_pos - center, dim=-1)
                    safe_r = radius + torch.as_tensor(float(self.obstacle_margin), device=trajectory.device, dtype=trajectory.dtype)
                    obstacle_loss = torch.relu(safe_r - dist).pow(2).mean()

                    self.guidance_debug["obstacle_loss_sum"] += float(obstacle_loss.detach().item())
                    self.guidance_debug["obstacle_loss_count"] += 1

                    obstacle_grad = torch.autograd.grad(obstacle_loss, trajectory, retain_graph=False, allow_unused=True)[0]
                    if obstacle_grad is None:
                        self.guidance_debug["none_grad_steps"] += 1
                    else:
                        grad_norm = float(torch.linalg.norm(obstacle_grad).detach().item())
                        if grad_norm <= 1e-12:
                            self.guidance_debug["zero_grad_steps"] += 1
                        else:
                            self.guidance_debug["applied_steps"] += 1
                            self.guidance_debug["grad_norm_sum"] += grad_norm
                            self.guidance_debug["grad_norm_max"] = max(float(self.guidance_debug["grad_norm_max"]), grad_norm)
                            alpha = torch.as_tensor(float(self.obstacle_guidance_scale), device=trajectory.device, dtype=trajectory.dtype) * (1 - scheduler.alphas_cumprod[t]).sqrt()
                            trajectory = trajectory.detach() - alpha * obstacle_grad

            trajectory = scheduler.step(
                model_output,
                t,
                trajectory,
                generator=generator,
                **kwargs,
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "obs" in obs_dict
        assert "past_action" not in obs_dict
        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            local_cond = torch.zeros(size=(B, T, Do), device=device, dtype=dtype)
            local_cond[:, :To] = nobs[:, :To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            global_cond = nobs[:, :To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs[:, :To]
            cond_mask[:, :To, Da:] = True

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[..., Da:]
            obs_pred = self.normalizer["obs"].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:, start:end]
            result["action_obs_pred"] = action_obs_pred
            result["obs_pred"] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch["obs"]
        action = nbatch["action"]

        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            local_cond = obs
            local_cond[:, self.n_obs_steps :, :] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:, : self.n_obs_steps, :].reshape(obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:, start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=trajectory.device).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        pred = self.model(noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss
