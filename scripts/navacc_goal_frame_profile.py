"""Independent fixed-goal/SI/dyn10 profile; no legacy checkpoint migration."""
from copy import deepcopy
from pathlib import Path
import hashlib,json
import torch
from omegaconf import OmegaConf
MODE='public21_goal_fixed_si_yaw144_dyn10_v1'
POLICY_IDENTITY='navacc_s80d20_short30_arrival80_savedlr_goal_fixed_si_yaw144_dyn10_actor213_v1'
CRITIC_SCHEMA='public21_goal_fixed_si_yaw144_dyn10_critic213_v1'
CONTRACT={'schema':MODE,'goal_frame':'episode_goal_frame_v1','action_frame':'goal_fixed_v1','dynamic_schema':'uG3_dxy_dz_vrelG3_radius_height_SI_v1','dynamic_slot_dim':10,'velocity_scale':1.,'gyro_body_scale':1.,'quaternion':'wxyz_left_inverse_unit_lex_positive_v1','static_lidar_frame':'actual_yaw_only','history_authority':'selected_request_world','horizon_s':'course_context_registered_32_36_40_45'}

def is_goal_config(cfg):return cfg.task.observation.get('mode')==MODE


def validate_config(cfg):
 from largemap_no_gru.config import validate_config as validate
 return validate(cfg)

def frame_binding(frame,*,start_W=None,goal_W=None,task_epoch=None):
 # Native caller supplies independently held final reset missions/epochs.
 if bool(frame.valid.any()) and any(x is None for x in (start_W,goal_W,task_epoch)):raise ValueError('valid G capture needs independent actual mission binding')
 return {'start_W':(frame.start if start_W is None else start_W).detach().cpu().clone(),'goal_W':(frame.goal if goal_W is None else goal_W).detach().cpu().clone(),'task_epoch':(frame.task_epoch if task_epoch is None else task_epoch).detach().cpu().clone()}

def validate_goal_runtime(runtime):
 from navacc_goal_frame import GoalFrame
 state=runtime['goal_frame'];binding=runtime['goal_frame_missions'];n=runtime['reset_epochs'].numel()
 if set(binding)!={'start_W','goal_W','task_epoch'}:raise ValueError('goal-frame mission binding keys differ')
 frame=GoalFrame(n,device='cpu',dtype=state['theta'].dtype)
 frame.load_state_dict(state,expected_start=binding['start_W'],expected_goal=binding['goal_W'],expected_task_epoch=binding['task_epoch'])
 if bool((frame.valid & (frame.task_epoch!=runtime['reset_epochs'])).any()):raise ValueError('goal-frame epoch not bound to runtime reset epochs')
 return frame






