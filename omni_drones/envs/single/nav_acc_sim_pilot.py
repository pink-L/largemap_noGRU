"""Experimental acceleration-level PPO hover task with a firmware-style inner loop."""
from __future__ import annotations

import math

import torch
from tensordict import TensorDict

from omni_drones.controllers.acceleration_to_rpyt import (
    AccelerationToRPYT,
    AccelerationToRPYTConfig,
)
from omni_drones.controllers.crazyflie_firmware_pid import (
    CrazyflieFirmwarePID,
    CrazyflieFirmwarePIDConfig,
)
from omni_drones.envs.isaac_env import AgentSpec
from omni_drones.envs.single.hover import Hover
from omni_drones.utils.torch import (
    euler_to_quaternion,
    quat_rotate_inverse,
    quaternion_to_euler,
)
from omni_drones.utils.torchrl.compat import (
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)


class NavAccSimPilot(Hover):
    """Obstacle-free simulation pilot; its checkpoints are not NavAcc-v1 checkpoints."""

    OBSERVATION_DIM = 80
    ACTIVE_OBSERVATION_DIM = 16

    def __init__(self, cfg, headless):
        self._validate_frozen_config(cfg)
        task = cfg.task
        self.target_xyz = tuple(float(x) for x in task.target)
        self.goal_position_scale = float(task.observation.goal_position_scale)
        self.velocity_scale = float(task.observation.velocity_scale)
        self.gyro_scale = float(task.observation.gyro_scale)
        self.train_position_half_range = float(task.reset.train_position_half_range_m)
        self.eval_position_half_range = float(task.reset.eval_position_half_range_m)
        self.roll_pitch_half_range_rad = math.radians(
            float(task.reset.roll_pitch_half_range_deg)
        )
        self.yaw_uniform_rad = tuple(float(x) for x in task.reset.yaw_uniform_rad)
        self.arena_xy = float(task.termination.arena_xy_m)
        self.crash_height = float(task.termination.crash_height_m)
        self.ceiling_height = float(task.termination.ceiling_height_m)
        self.terminal_penalty = float(task.reward.terminal_penalty)
        super().__init__(cfg, headless)

        self.target_pos = torch.tensor([self.target_xyz], device=self.device)
        self.firmware_pid = self._make_firmware_pid()
        self.acceleration_to_rpyt = self._make_acceleration_to_rpyt()
        shape3 = (self.num_envs, 1, 3)
        shape4 = (self.num_envs, 1, 4)
        self.current_a_nom = torch.zeros(shape3, device=self.device)
        self.previous_a_nom = torch.zeros_like(self.current_a_nom)
        self.previous_a_nom_valid = torch.zeros(
            self.num_envs, 1, dtype=torch.bool, device=self.device
        )
        self.held_rpyt = torch.zeros(shape4, device=self.device)
        self.last_rotor_command = torch.zeros(shape4, device=self.device)
        self.outer_substep_count = torch.zeros(
            self.num_envs, 1, dtype=torch.long, device=self.device
        )
        self.total_substep_count = torch.zeros_like(self.outer_substep_count)

    def _make_firmware_pid(self):
        """Legacy factory; a distinct physical task may override explicitly."""
        return CrazyflieFirmwarePID(CrazyflieFirmwarePIDConfig.cf21bl_2025_02())

    def _make_acceleration_to_rpyt(self):
        return AccelerationToRPYT(AccelerationToRPYTConfig.cf21bl_simulation_reference())

    def _reset_firmware_motor_command(self, env_ids, attitude_quat_wxyz):
        """Legacy force-fraction reset and [-1,1] command encoding."""
        initial_force_fraction = self.drone.throttle[env_ids].square()
        self.firmware_pid.reset(
            env_ids, attitude_quat_wxyz=attitude_quat_wxyz,
            motor_state=initial_force_fraction,
        )
        return 2.0 * initial_force_fraction - 1.0

    @staticmethod
    def _validate_frozen_config(cfg) -> None:
        errors = []
        if cfg.task.name != "NavAccSimPilot":
            errors.append("task.name must be NavAccSimPilot")
        if cfg.task.drone_model.name != "Crazyflie" or cfg.task.drone_model.controller is not None:
            errors.append("NavAccSimPilot requires Crazyflie with controller=null")
        if float(cfg.sim.dt) != 0.002 or int(cfg.sim.substeps) != 5:
            errors.append("NavAccSimPilot requires sim.dt=0.002 and sim.substeps=5")
        if int(cfg.env.max_episode_length) != 1000:
            errors.append("NavAccSimPilot requires a 1000-step (10 s) episode")
        if list(cfg.task.action.acceleration_scale_m_s2) != [2.0, 2.0, 2.0]:
            errors.append("unexpected experimental acceleration scale")
        if list(cfg.task.target) != [0.0, 0.0, 1.5]:
            errors.append("NavAccSimPilot target must be [0,0,1.5]")
        if list(cfg.task.reset.yaw_uniform_rad) != [-math.pi, math.pi]:
            errors.append("NavAccSimPilot reset yaw must be uniform on [-pi,pi]")
        if float(cfg.task.reset.roll_pitch_half_range_deg) != 3.0:
            errors.append("NavAccSimPilot reset roll/pitch range must be +/-3 degrees")
        if (
            float(cfg.task.termination.crash_height_m) != 0.15
            or float(cfg.task.termination.arena_xy_m) != 3.0
            or float(cfg.task.termination.ceiling_height_m) != 3.0
        ):
            errors.append("unexpected NavAccSimPilot termination boundary")
        if errors:
            raise ValueError("; ".join(errors))

    def _set_specs(self):
        self.observation_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {
                        "observation": UnboundedContinuousTensorSpec(
                            (1, self.OBSERVATION_DIM), device=self.device
                        ),
                        "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device),
                    }
                )
            }
        ).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"action": UnboundedContinuousTensorSpec((1, 4), device=self.device)}
                )
            }
        ).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"reward": UnboundedContinuousTensorSpec((1, 1), device=self.device)}
                )
            }
        ).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics"),
        )

        scalar = lambda: UnboundedContinuousTensorSpec((1,), device=self.device)
        stats_spec = CompositeSpec(
            {
                "return": scalar(),
                "episode_len": scalar(),
                "pos_error": scalar(),
                "speed": scalar(),
                "crash": scalar(),
                "oob": scalar(),
            }
        ).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

        info_spec = CompositeSpec(
            {
                "drone_state": UnboundedContinuousTensorSpec((1, 13), device=self.device),
                "yaw_hold_W": UnboundedContinuousTensorSpec((1, 1), device=self.device),
                "a_nom": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                "rpyt_api": UnboundedContinuousTensorSpec((1, 4), device=self.device),
                "rpyt_tilt_rad": UnboundedContinuousTensorSpec((1, 1), device=self.device),
                "rpyt_thrust_saturated": DiscreteTensorSpec(
                    2, (1, 1), dtype=torch.bool, device=self.device
                ),
                "rotor_command": UnboundedContinuousTensorSpec((1, 4), device=self.device),
                "motor_pwm": UnboundedContinuousTensorSpec((1, 4), device=self.device),
                "pid_rate_output": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                "firmware_yaw_reference_deg": UnboundedContinuousTensorSpec(
                    (1, 1), device=self.device
                ),
                "substeps_last_outer": UnboundedContinuousTensorSpec(
                    (1,), device=self.device, dtype=torch.long
                ),
                "substeps_total": UnboundedContinuousTensorSpec(
                    (1,), device=self.device, dtype=torch.long
                ),
                "termination_reason": UnboundedContinuousTensorSpec(
                    (1,), device=self.device, dtype=torch.long
                ),
            }
        ).expand(self.num_envs).to(self.device)
        self.observation_spec["info"] = info_spec
        self.info = info_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        count = env_ids.numel()
        half_range = (
            self.train_position_half_range if self.training else self.eval_position_half_range
        )
        pos = self.target_pos.expand(count, 1, 3).clone()
        pos += torch.empty((count, 1, 3), device=self.device).uniform_(
            -half_range, half_range
        )
        rpy = torch.empty((count, 1, 3), device=self.device)
        rpy[..., :2].uniform_(-self.roll_pitch_half_range_rad, self.roll_pitch_half_range_rad)
        rpy[..., 2].uniform_(*self.yaw_uniform_rad)
        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids].unsqueeze(1), rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)

        target_world = self.target_pos.expand(count, 3) + self.envs_positions[env_ids]
        target_rot = euler_to_quaternion(
            torch.zeros((count, 3), device=self.device)
        )
        self.target_vis.set_world_poses(target_world, target_rot, env_indices=env_ids)

        if not self.firmware_pid._initialized and count != self.num_envs:
            raise RuntimeError("first NavAccSimPilot reset must cover every environment")
        initial_rotor = self._reset_firmware_motor_command(env_ids, rot)
        yaw_hold = quaternion_to_euler(rot)[..., 2:3]
        hover_rpyt, hover_diagnostics = self.acceleration_to_rpyt(
            torch.zeros((count, 1, 3), device=self.device), yaw_hold
        )
        if not bool(hover_diagnostics["valid"].all()):
            raise RuntimeError("reset hover RPYT is invalid")
        # TensorDict scalar assignment creates one integer source for every
        # leaf and fails for this intentionally mixed float/bool/long spec.
        for value in self.info.values(True, True):
            value[env_ids] = torch.zeros_like(value[env_ids])
        self.info["yaw_hold_W"][env_ids] = yaw_hold
        self.current_a_nom[env_ids] = 0
        self.previous_a_nom[env_ids] = 0
        self.previous_a_nom_valid[env_ids] = False
        self.held_rpyt[env_ids] = hover_rpyt
        self.info["rpyt_api"][env_ids] = hover_rpyt
        self.info["rpyt_tilt_rad"][env_ids] = hover_diagnostics["tilt_rad"]
        self.info["rpyt_thrust_saturated"][env_ids] = hover_diagnostics[
            "thrust_saturated"
        ]
        self.last_rotor_command[env_ids] = initial_rotor
        self.info["rotor_command"][env_ids] = initial_rotor
        self.outer_substep_count[env_ids] = 0
        self.total_substep_count[env_ids] = 0
        for value in self.stats.values(True, True):
            value[env_ids] = torch.zeros_like(value[env_ids])

    def _pre_sim_step(self, tensordict):
        rpyt = tensordict[("agents", "action")]
        if rpyt.shape != (self.num_envs, 1, 4):
            raise ValueError(f"base action must be [B,1,4] RPYT, got {tuple(rpyt.shape)}")
        if not torch.isfinite(rpyt).all():
            raise RuntimeError("non-finite RPYT reached NavAccSimPilot")
        first_substep = self.outer_substep_count == 0
        if bool(first_substep.all()):
            self.current_a_nom.copy_(tensordict[("info", "a_nom")])
            self.held_rpyt.copy_(rpyt)
            self.info["a_nom"].copy_(self.current_a_nom)
            self.info["rpyt_api"].copy_(rpyt)
            self.info["rpyt_tilt_rad"].copy_(tensordict[("info", "rpyt_tilt_rad")])
            self.info["rpyt_thrust_saturated"].copy_(
                tensordict[("info", "rpyt_thrust_saturated")]
            )
        elif not torch.equal(rpyt, self.held_rpyt):
            raise RuntimeError("RPYT changed inside a five-substep outer step")

        state = self.drone.get_state()
        quat = state[..., 3:7]
        gyro_body = quat_rotate_inverse(quat, state[..., 10:13])
        rotor, diagnostics = self.firmware_pid.step(rpyt, quat, gyro_body)
        if not torch.isfinite(rotor).all():
            raise RuntimeError("firmware PID produced non-finite rotor commands")
        self.last_rotor_command.copy_(rotor)
        self.info["rotor_command"].copy_(rotor)
        self.info["motor_pwm"].copy_(diagnostics["motor_pwm"])
        self.info["pid_rate_output"].copy_(diagnostics["rate_pid_output_float"])
        self.info["firmware_yaw_reference_deg"].copy_(
            diagnostics["firmware_yaw_reference_deg"]
        )
        self.effort = self.drone.apply_action(rotor)
        self.outer_substep_count += 1
        self.total_substep_count += 1

    def _post_sim_step(self, tensordict):
        if not bool((self.outer_substep_count == self.substeps).all()):
            raise RuntimeError("NavAccSimPilot did not execute exactly five inner-loop steps")
        self.info["substeps_last_outer"].copy_(self.outer_substep_count)
        self.info["substeps_total"].copy_(self.total_substep_count)
        self.outer_substep_count.zero_()

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()
        self.info["drone_state"].copy_(self.drone_state[..., :13])
        self.rpos = self.target_pos - self.drone_state[..., :3]
        quat = self.drone_state[..., 3:7]
        velocity_W = self.drone_state[..., 7:10]
        gyro_B = quat_rotate_inverse(quat, self.drone_state[..., 10:13])
        yaw_hold = self.info["yaw_hold_W"]
        time_fraction = (self.progress_buf / self.max_episode_length).clamp(0, 1)
        active = torch.cat(
            (
                self.rpos / self.goal_position_scale,
                quat,
                velocity_W / self.velocity_scale,
                gyro_B / self.gyro_scale,
                torch.sin(yaw_hold),
                torch.cos(yaw_hold),
                time_fraction[:, None, None],
            ),
            dim=-1,
        )
        if active.shape[-1] != self.ACTIVE_OBSERVATION_DIM:
            raise RuntimeError("NavAccSimPilot active observation is not 16-wide")
        obstacle_slots = torch.zeros(
            self.num_envs, 1, self.OBSERVATION_DIM - self.ACTIVE_OBSERVATION_DIM,
            device=self.device, dtype=active.dtype,
        )
        obs = torch.cat((active, obstacle_slots), dim=-1)
        return TensorDict(
            {
                "agents": {"observation": obs, "intrinsics": self.drone.intrinsics},
                # Every collector frame owns an immutable telemetry snapshot.
                # Returning the live TensorDict aliases all rollout frames to
                # the most recent command/state.
                "info": self.info.clone(),
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        pos_error = torch.linalg.vector_norm(self.rpos, dim=-1)
        speed = torch.linalg.vector_norm(self.drone_state[..., 7:10], dim=-1)
        delta_a = torch.where(
            self.previous_a_nom_valid.unsqueeze(-1),
            self.current_a_nom - self.previous_a_nom,
            torch.zeros_like(self.current_a_nom),
        )
        reward = (
            2.0 * torch.exp(-4.0 * pos_error.square())
            + torch.exp(-speed.square())
            - 0.01 * self.current_a_nom.square().sum(-1)
            - 0.01 * delta_a.square().sum(-1)
        )
        pos = self.drone_state[..., :3]
        crash = pos[..., 2] < self.crash_height
        oob = (
            (pos[..., :2].abs() > self.arena_xy).any(-1)
            | (pos[..., 2] > self.ceiling_height)
        )
        nonfinite = ~torch.isfinite(self.drone_state).all(-1)
        terminated = crash | oob | nonfinite
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        reward = reward + terminated.to(reward.dtype) * self.terminal_penalty

        reason = torch.zeros_like(self.info["termination_reason"])
        reason = torch.where(crash, torch.ones_like(reason), reason)
        reason = torch.where(oob, torch.full_like(reason, 2), reason)
        reason = torch.where(nonfinite, torch.full_like(reason, 3), reason)
        reason = torch.where(truncated & ~terminated, torch.full_like(reason, 4), reason)
        self.info["termination_reason"].copy_(reason)

        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(-1)
        self.stats["pos_error"].copy_(pos_error)
        self.stats["speed"].copy_(speed)
        self.stats["crash"].copy_(crash.float())
        self.stats["oob"].copy_(oob.float())
        self.previous_a_nom.copy_(self.current_a_nom)
        self.previous_a_nom_valid.fill_(True)

        return TensorDict(
            {
                "agents": {"reward": reward.unsqueeze(-1)},
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
                "info": self.info.clone(),
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )
