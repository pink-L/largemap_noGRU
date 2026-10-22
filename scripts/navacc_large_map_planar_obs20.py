from navacc_large_map_hold48 import OBS_MODE as HOLD_MODE,OBS_SCHEMA as HOLD_SCHEMA
from pathlib import Path
import torch
MODE = 'public20_yaw_v1'
OBSERVATION_SCHEMA = 'true_worldgoal_public20_prevselectedrequest_yawlidar36x4_near2cvthreat2_v1'
POLICY_IDENTITY = 'navacc_planar_public20_yawlidar_actor212_publiccritic212_v1'
CRITIC_SCHEMA = 'structured212_public20_no_private_distance_v1'
GOAL_SCALE_MODE = 'public20_xy1_z1_logd_yaw_v1'
GOAL_SCALE_SCHEMA = 'worldgoal_xymin1_z1_logdistance1_public20_prevselectedrequest_yawlidar36x4_near2cvthreat2_v1'
GOAL_SCALE_POLICY_IDENTITY = 'navacc_planar_public20_xy1_z1_logd_yaw_actor212_publiccritic212_v1'
GOAL_SCALE_CRITIC_SCHEMA = 'structured212_public20_xy1_z1_logd_no_private_distance_v1'
GOAL_FRAME_MODE="public21_goal_fixed_si_yaw144_dyn10_v1"
GOAL_FRAME_SCHEMA=GOAL_FRAME_MODE
PUBLIC20_MODES = (MODE, GOAL_SCALE_MODE, HOLD_MODE, GOAL_FRAME_MODE)
PUBLIC20_SCHEMAS = (OBSERVATION_SCHEMA, GOAL_SCALE_SCHEMA, HOLD_SCHEMA, GOAL_FRAME_SCHEMA)
SCHEMA_BY_MODE = dict(zip(PUBLIC20_MODES, PUBLIC20_SCHEMAS))
REF_DISTANCE_M = (50.**2*2 + 4.5**2)**.5
SCALES = [.8,.8,.5]
PHASES = {'corr003':'obs20_actor20m_corr003_s350_d0_local10m_v1',
          'corr030':'obs20_actor20m_corr030_s350_d0_local10m_v1'}
COEFFICIENTS = {'corr003':.03,'corr030':.30}
EVAL_PROTOCOL = 'fixed_first_terminal80s_obs20_actor20m_corr_ab_local10m_v1'
PARENT_SHA256 = '537242c259a6ebfd16b9a4362dc165381a8ca2fbec2f29013d6fac8869a6de83'
def validate_observation_config(cfg):
    o = cfg.task.observation
    if o.get("mode")==GOAL_FRAME_MODE:
        from navacc_goal_frame_profile import validate_config
        validate_config(cfg);return
    required = {'mode':MODE, 'schema':OBSERVATION_SCHEMA, 'state_dim':20,
                'goal_position_scale_m':REF_DISTANCE_M, 'goal_distance_scale_m':REF_DISTANCE_M,
                'previous_request_scales_m_s2':SCALES, 'static_frame':'actual_yaw_only',
                'static_encoding':'4-clamp(distance,0,4)',
                'missing_previous_request':'zero_placeholder_valid_false_telemetry_only'}
    if o.get('mode') == MODE and cfg.algo.get('structured_observation_mode') not in (None, MODE):
        raise ValueError('legacy public20 observation/encoder mode mismatch')
    if o.get('mode') in (GOAL_SCALE_MODE, HOLD_MODE):
        required.pop('goal_position_scale_m');required.pop('goal_distance_scale_m')
        required.update(mode=GOAL_SCALE_MODE,schema=GOAL_SCALE_SCHEMA,
            goal_xy_min_norm_m=1.,goal_z_scale_m=1.,goal_distance_log_scale_m=1.,
            goal_position_scale=1.,goal_xy_encoding='delta_xy/divide_clamped_xy_norm',
            goal_distance_encoding='log1p_true_3d_distance')
        if o.get('mode') == HOLD_MODE:
            required.update(mode=HOLD_MODE,schema=HOLD_SCHEMA,state_dim=21)
        if cfg.algo.get('structured_observation_mode') != o.get('mode'):
            raise ValueError('goal-scale observation/encoder mode mismatch')
        if o.get('goal_position_scale_m') is not None or o.get('goal_distance_scale_m') is not None:
            raise ValueError('goal-scale mode rejects obsolete linear goal scales')
    for key, expected in required.items():
        value = o.get(key)
        if key == 'previous_request_scales_m_s2' and value is not None:value=list(value)
        if value != expected:raise ValueError('obs20 config mismatch ' + key)
    if o.guidance_mode != 'global_goal_v1' or int(cfg.algo.structured_state_dim) != (21 if o.get('mode') == HOLD_MODE else 20):
        raise ValueError('obs20 requires global guidance and encoder20')

def publish_selected_request(base, request, valid=None):
    if not getattr(base, '_planar_public20', False):return
    if valid is None:valid=torch.ones(base.num_envs,device=request.device,dtype=torch.bool)
    valid=valid.reshape(-1)
    if request.shape != (base.num_envs,3) or valid.shape != (base.num_envs,):
        raise ValueError('selected request history shape mismatch')
    valid=valid & torch.isfinite(request).all(-1)
    base._planar_previous_selected_request.copy_(torch.where(valid[:,None], request, torch.zeros_like(request)))
    base._planar_previous_selected_request_valid.copy_(valid)
    if 'planar_previous_selected_request_valid' in base.info.keys():
        base.info['planar_previous_selected_request_valid'].copy_(valid[:,None])

def assert_native_observation(td,base,policy):
    from omni_drones.learning.ppo.navacc_v4_structured import OBS_KEYS
    width=21 if getattr(base,"_hold48",False) else 20
    fused=192+width
    expected=((base.num_envs,1,width),(base.num_envs,1,36,4),(base.num_envs,1,4,10 if getattr(base,"_goal_frame",None) is not None else 8))
    for key,shape in zip(OBS_KEYS,expected):
        value=td[key]
        if tuple(value.shape)!=shape or value.device!=torch.device(base.device) or value.dtype!=torch.float32 or not bool(torch.isfinite(value).all()):
            raise RuntimeError('obs20 real observation contract differs '+str(key))
    if list(policy.critic.in_keys)!=list(OBS_KEYS):raise RuntimeError('obs20 critic consumes private observations')
    first=policy.actor.state_dict()['module.0.module.head.0.weight']
    if first.shape!=(256,fused) or policy.critic.module.head[0].in_features!=fused:raise RuntimeError('obs20 real fused width differs')
    _history=base._planar_previous_selected_request
    if getattr(base,"_goal_frame",None) is not None:
        from navacc_goal_frame import world_to_goal
        _history=world_to_goal(_history,base._goal_frame.theta)
    torch.testing.assert_close(td[OBS_KEYS[0]][:,0,17:20],_history / td[OBS_KEYS[0]].new_tensor(SCALES),rtol=0,atol=0)
    return {'device':str(base.device),'observation_shapes':[list(x) for x in expected],
            'actor_fused_dim':fused,'critic_fused_dim':fused,'critic_in_keys':[list(x) for x in policy.critic.in_keys],
            'previous_selected_request_matches_state':True}
