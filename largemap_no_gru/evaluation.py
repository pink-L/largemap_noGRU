"""Original first-terminal/parking evaluation; no campaign loader."""
import torch,numpy as np,math,time,hashlib
from pathlib import Path
from .checkpoint import atomic_json,sha as sha256_file
class _CrossEvalActionNoise:
    """Paired request-indexed standard normal stream for random eval actions.

    Draws exactly (num_envs, 1, width) standard normals per tick from
    SeedSequence([action_seed]); consumption is independent of lane
    terminations, so the same (request, tick) receives identical noise in
    every random cell and every checkpoint run.
    """
    def __init__(self, seed, num_envs, device, width=3):
        import numpy as _np
        if seed is None:
            raise ValueError("random action mode requires an action seed")
        self._rng = _np.random.default_rng(_np.random.SeedSequence([int(seed)]))
        self.num_envs = int(num_envs); self.device = device; self.width = width
        self.rolling_sha256=hashlib.sha256();self.draw_count=0
    def draw(self):
        sample = self._rng.standard_normal((self.num_envs, 1, self.width))
        self.rolling_sha256.update(sample.astype(np.float32).tobytes());self.draw_count+=1
        return torch.as_tensor(sample, dtype=torch.float32, device=self.device)

def _apply_random_eval_action(policy, td, action_noise):
    """Replace the MODE action with a paired-draw sample of the SAME policy.

    PPOPolicy.__call__ excludes loc/scale and pre-sets z_policy to the MODE
    action; therefore re-run the actor parameter module for loc/scale, draw z'
    via the paired noise stream, refresh log_prob, and keep
    z_policy == action so the action transform's equality guard stays
    satisfied.
    """
    policy.actor.module[0](td)  # ModuleList wraps the structured actor
    loc = td["loc"]; scale = torch.clamp_min(td["scale"], 1e-6)
    z = loc + scale * action_noise.draw()
    td.set(("agents", "action"), z)
    td.set("sample_log_prob", torch.distributions.Independent(
        torch.distributions.Normal(loc, scale), 1).log_prob(z))
    td.set(("agents", "z_policy"), z.clone())
    td.exclude("loc", "scale", inplace=True)

def run_fixed_zero_evaluation(env, policy, base, max_steps, progress_path=None,
                              action_mode="deterministic", action_seed=None,
                              diag_ids=None, diag_path=None, diag_tag=None, diag_ids_path=None, value_trace=None, diagnostic_loop_limit=None):
    """No collector, probe or resets after initial fixed requests; first done locks.

    ``diag_ids`` adds a read-only per-tick recorder for the selected lanes: the
    executed trajectory is unchanged (the recorder only reads post-step info that
    the loop already consumes), and the run stays a selected-lane diagnostic that
    is never reported as a formal 1024-cohort outcome."""
    from torchrl.envs.utils import step_mdp, set_exploration_type, ExplorationType
    diag = Hold48DiagnosticRecorder(diag_ids) if diag_ids else None
    from torchrl.envs.utils import step_mdp, set_exploration_type, ExplorationType
    hold48=getattr(base,'_hold48',False)
    hold36=getattr(base,'_hold36',False)
    if hold48 and max_steps!=(int(base.cfg.task.large_map.course_context.horizon_s)*100 if base.cfg.task.large_map.get("episode_mode")=="obstacle_course_continuous_v1" else 5000 if base.cfg.task.large_map.get("episode_mode")=="hold50_continuous_v1" else 3000 if hold36 else 4800):raise ValueError('finite hold evaluation horizon differs')
    if diagnostic_loop_limit is not None:
        from navacc_goal_curriculum import active as curriculum_active
        if diagnostic_loop_limit not in (32,64) or not hold36 or max_steps!=(int(base.cfg.task.large_map.course_context.horizon_s)*100 if base.cfg.task.large_map.get("episode_mode")=="obstacle_course_continuous_v1" else 5000 if base.cfg.task.large_map.get("episode_mode")=="hold50_continuous_v1" else 3000) or not curriculum_active(base.cfg):raise ValueError('short eval loop only explicit curriculum diagnostic32/64')
    actual_loop_steps=max_steps if diagnostic_loop_limit is None else diagnostic_loop_limit
    hold_end={k:torch.zeros(base.num_envs,device=base.device) for k in ('raw_in_goal_end','safe_reach_goal_end','ever_reached','hold_time_s','first_reach_time_s','first_reach_speed')}
    policy.eval()
    if action_mode not in ("deterministic", "random"):
        raise ValueError("eval action mode must be deterministic or random")
    action_noise = (_CrossEvalActionNoise(action_seed, base.num_envs, base.device)
                    if action_mode == "random" else None)
    td=env.reset()
    e=base.num_envs
    locked=torch.zeros(e,dtype=torch.bool,device=base.device)
    arrival=torch.zeros_like(locked);failure=torch.zeros_like(locked)
    tail_low_speed_steps=torch.zeros(e,dtype=torch.long,device=base.device)
    distance0=td["info","large_goal_distance_m"].reshape(e).clone()
    distance=distance0.clone();speed=torch.zeros_like(distance)
    arrival_velocity=torch.zeros(e,3,device=base.device)
    arrival_distance=torch.full((e,),float('nan'),device=base.device)
    fallback=torch.zeros(e,dtype=torch.long,device=base.device)
    steps=torch.zeros_like(fallback)
    reason=torch.zeros_like(fallback)
    episode_return=torch.full((e,),float('nan'),device=base.device)
    flags={k:torch.zeros_like(locked) for k in ("collision_static","collision_dynamic","crash","oob","timeout","nonfinite")}
    passive_counts={k:torch.zeros(e,device=base.device,dtype=torch.long) for k in ('static_encounter_steps','dynamic_encounter_steps','cbf_active_steps','cbf_active_row_total','cbf_correction_steps','cbf_row_failed_steps','cbf_selected_static_steps','cbf_selected_dynamic_steps')}
    minimum={f'{kind}_{axes}':torch.full((e,),float('inf'),device=base.device) for kind in ('static','dynamic') for axes in ('3d','xy')}
    saturation=torch.zeros(e,dtype=torch.long,device=base.device)
    chosen_saturation=torch.zeros_like(saturation);chosen_saturation_observed=torch.zeros_like(saturation)
    from navacc_large_map_planar_unbounded import protocol_of as _eval_protocol_of, PROTOCOL as _eval_unbounded_protocol
    chosen_saturation_propagated=(_eval_protocol_of(getattr(base,'cfg',{}))==_eval_unbounded_protocol)
    nominal_accel_sum=torch.zeros(e,device=base.device)
    commanded_accel_sum=torch.zeros(e,device=base.device)
    goal_dir_speed_sum=torch.zeros(e,device=base.device)
    goal_xy=torch.as_tensor(np.asarray([goal[:2] for goal in base.large_scene.goals]),device=base.device,dtype=torch.float32)
    for kind in ('static','dynamic'):
        defined=td.get(('info',f'planar_{kind}_clearance_defined'),None)
        if defined is not None:
            valid=defined.reshape(e).bool()
            for axes in ('3d','xy'):
                value=td['info',f'planar_{kind}_clearance_{axes}_m'].reshape(e)
                minimum[f'{kind}_{axes}'][valid]=value[valid]
    from navacc_correction_eval_metrics import CorrectionMetrics
    correction_metrics=CorrectionMetrics(e,base.device)
    def passive_row(i):
        return {**correction_metrics.row(i),**{k:int(v[i]) for k,v in passive_counts.items()},**{f'minimum_{k}_net_clearance_m':float(v[i]) if bool(torch.isfinite(v[i])) else None for k,v in minimum.items()}}
    started=time.monotonic();parking_resets=0
    with set_exploration_type(ExplorationType.MODE):
        for tick in range(actual_loop_steps):
            policy(td)
            if action_noise is not None:
                _apply_random_eval_action(policy, td, action_noise)
            if value_trace is not None:value_trace.before(td,policy)
            transition=env.step(td)
            nxt=transition["next"]
            active=~locked
            correction_metrics.observe(nxt,active)
            if value_trace is not None:value_trace.after(tick,nxt,active)
            distance[active]=nxt["info","large_goal_distance_m"].reshape(e)[active]
            speed[active]=nxt["stats","speed"].reshape(e)[active]
            fallback += (active & nxt["info","planar_cbf_fallback"].reshape(e)).long()
            cbf_active=nxt['info','planar_cbf_active_count'].reshape(e).long()
            passive_counts['cbf_active_steps']+=(active&(cbf_active>0)).long()
            chosen=nxt.get(('info','planar_cbf_selected_count'),None)
            kind=nxt.get(('info','planar_cbf_selected_is_static'),None)
            if chosen is not None and kind is not None:
                selected=active&(chosen.reshape(e)>0)
                passive_counts['cbf_selected_static_steps']+=(selected&kind.reshape(e)).long()
                passive_counts['cbf_selected_dynamic_steps']+=(selected&~kind.reshape(e)).long()
            passive_counts['cbf_active_row_total']+=torch.where(active,cbf_active,0)
            correction=nxt.get(('info','planar_reward_cbf_correction'),None)
            if correction is not None:passive_counts['cbf_correction_steps']+=(active&(correction.reshape(e)<-1e-12)).long()
            failed=~nxt['info','planar_cbf_domain_valid'].reshape(e)|~nxt['info','planar_cbf_selected_row_feasible'].reshape(e)|~nxt['info','planar_cbf_selected_row_pass'].reshape(e)
            passive_counts['cbf_row_failed_steps']+=(active&(cbf_active>0)&failed).long()
            for kind in ('static','dynamic'):
                encounter=nxt.get(('info',f'planar_{kind}_encounter_center3m_count'),None)
                defined=nxt.get(('info',f'planar_{kind}_clearance_defined'),None)
                if encounter is not None:passive_counts[f'{kind}_encounter_steps']+=(active&(encounter.reshape(e)>0)).long()
                if defined is not None:
                    valid=active&defined.reshape(e).bool()
                    for axes in ('3d','xy'):
                        value=nxt['info',f'planar_{kind}_clearance_{axes}_m'].reshape(e)
                        minimum[f'{kind}_{axes}'][valid]=torch.minimum(minimum[f'{kind}_{axes}'][valid],value[valid])
            steps += active.long()
            sat=nxt.get(('info','planar_cbf_candidate_thrust_saturated'),None)
            if sat is not None:
                saturation += (active & sat.reshape(e).bool()).long()
            chosen_sat=nxt.get(('info','rpyt_thrust_saturated'),None)
            if chosen_saturation_propagated and chosen_sat is not None:
                chosen_saturation_observed += active.long()
                chosen_saturation += (active & chosen_sat.reshape(e).bool()).long()
            nom=nxt.get(('info','planar_policy_nominal_acceleration'),None)
            if nom is not None:
                nominal_accel_sum[active]+=torch.linalg.norm(nom.reshape(e,3),dim=-1)[active]
            cmd=nxt.get(('info','planar_commanded_acceleration'),None)
            if cmd is not None:
                commanded_accel_sum[active]+=torch.linalg.norm(cmd.reshape(e,3),dim=-1)[active]
            state=nxt.get(('info','drone_state'),None)
            if state is not None:
                st=state.reshape(e,-1); vel=st[:,7:10]; pos=st[:,:3]
                from navacc_obstacle_course_state import update_tail_low_speed
                tail_low_speed_steps=update_tail_low_speed(tail_low_speed_steps,active,distance,vel)
                delta=goal_xy-pos[:,:2]
                nd=torch.linalg.norm(delta,dim=-1,keepdim=True)
                unit=torch.where(nd>1e-6,delta/torch.clamp_min(nd,1e-6),torch.zeros_like(delta))
                goal_dir_speed_sum[active]+=(vel[:,:2]*unit).sum(-1)[active]
            done=nxt["done"].reshape(e)&active
            if diag is not None:
                diag.record(tick, e, active, nxt, goal_xy, base.device)
            from navacc_large_map_planar_logging import capture_first_completed_returns
            capture_first_completed_returns(episode_return,nxt,done)
            if hold48:
                for k in hold_end:hold_end[k][done]=nxt['stats',k].reshape(e)[done]
            arrival_event=done & nxt["stats","arrival"].reshape(e).bool()
            if bool(arrival_event.any()):
                st_arr=nxt.get(('info','drone_state'),None)
                if st_arr is None:raise RuntimeError("arrival truth speed requires drone_state on the terminal step")
                vel_arr=st_arr.reshape(e,-1)[:,7:10]
                arrival_velocity[arrival_event]=vel_arr[arrival_event].detach()
                arrival_distance[arrival_event]=distance[arrival_event]
            arrival |= arrival_event
            failure |= done & ~nxt["stats","arrival"].reshape(e).bool()
            reason[done]=nxt["info","termination_reason"].reshape(e)[done].long()
            for key in flags: flags[key][done]=nxt["stats",key].reshape(e)[done].bool()
            locked |= done
            if progress_path is not None and (bool(done.any()) or tick%20==0):
                atomic_json(progress_path,dict(tail_low_speed_hold_criterion={"radius_m":.5,"speed_3d_max_m_s":.3,"required_s":5.,"requests":e},
        last5s_low_speed_hold_count=int(((tail_low_speed_steps>=500)&locked&(steps==max_steps)&(reason==10)).sum()),
        requests=e,completed=int(locked.sum()),arrivals=int(arrival.sum()),failures=int(failure.sum()),censored=int((~locked).sum()),tick=tick+1,
                    requests_detail=[dict(id=i,completed=bool(locked[i]),arrival=bool(arrival[i]),termination_reason=int(reason[i]),final_distance_m=float(distance[i]),steps=int(steps[i]),**passive_row(i),**{k:bool(v[i]) for k,v in flags.items()}) for i in range(e)]))
            if bool(locked.all()):break
            td=step_mdp(transition,keep_other=True)
            # Park terminal lanes without resampling any active request. All
            # first results remain locked; shared clock is not reset.
            if bool(locked.any()):
                ids=locked.nonzero().reshape(-1)
                parking_resets+=int(ids.numel())
                base._reset_idx(ids)
                fresh=base._compute_state_and_obs()
                td.set(("info","drone_state"),fresh["info","drone_state"])
                td.set(("info","large_goal_distance_m"),fresh["info","large_goal_distance_m"])
                td.set(("agents","observation"),fresh["agents","observation"])
                td.set("is_init",locked[:,None])
    def _eval_percentile_stats(values):
        arr=[float(v) for v in values]
        if not arr:return {"count":0,"mean":None,"median":None,"p90":None,"p95":None,"max":None}
        ordered=sorted(arr)
        def _pick(p):
            if len(ordered)==1:return ordered[0]
            position=(len(ordered)-1)*p;low=math.floor(position);high=math.ceil(position)
            return ordered[low]+(ordered[high]-ordered[low])*(position-low)
        return {"count":len(arr),"mean":sum(arr)/len(arr),"median":_pick(.5),"p90":_pick(.9),"p95":_pick(.95),"max":max(arr)}
    _arrival_ids=[i for i in range(e) if bool(arrival[i])]
    arrival_speed_stats={"unit":"m/s","scope":"first_terminal_arrival_truth_world_velocity_before_reset",
        "speed_3d":_eval_percentile_stats([float(torch.linalg.norm(arrival_velocity[i]).item()) for i in _arrival_ids]),
        "speed_xy":_eval_percentile_stats([float(torch.linalg.norm(arrival_velocity[i,:2]).item()) for i in _arrival_ids]),
        "velocity_z_signed":_eval_percentile_stats([float(arrival_velocity[i,2].item()) for i in _arrival_ids]),
        "abs_velocity_z":_eval_percentile_stats([float(abs(arrival_velocity[i,2].item())) for i in _arrival_ids]),
        "arrival_distance_m":_eval_percentile_stats([float(arrival_distance[i].item()) for i in _arrival_ids if math.isfinite(float(arrival_distance[i]))]),
        "sample_count":len(_arrival_ids)}
    _failure_ids=[i for i in range(e) if bool(failure[i])]
    failure_final_speed_stats={"unit":"m/s","scope":"completed_failure_lanes_terminal_frame_speed_scalar_separate_from_arrival_denominator",
        "speed":_eval_percentile_stats([float(speed[i]) for i in _failure_ids])}
    diag_receipt=None
    if diag is not None:
        diag_receipt=diag.receipt(diag_tag,diag_ids_path,sha256_file=sha256_file)
        if diag_path is not None:
            atomic_json(Path(diag_path),dict(diag_receipt,columns_data=diag.rows))
            diag_receipt['trace_file']=str(Path(diag_path).name)
    return dict(schema=("curriculum_censored_eval_smoke_v1" if diagnostic_loop_limit is not None else "fixed_obstacle_course_hold_end_v1" if base.cfg.task.large_map.get("episode_mode")=="obstacle_course_continuous_v1" else "fixed_finite50s_hold_end_v1" if base.cfg.task.large_map.get("episode_mode")=="hold50_continuous_v1" else "fixed_finite30s_hold_end_v1" if hold36 else "fixed_finite48s_hold_end_v1" if hold48 else "navacc_fixed_first_terminal_eval_v1"),
        **({'diagnostic_smoke':True,'formal_evaluation':False,'actual_loop_steps':actual_loop_steps,'configured_horizon_steps':max_steps,'not_navigation_reliability_evidence':True} if diagnostic_loop_limit is not None else {}),
        **({'hold48_diagnostic_trace':diag_receipt} if diag_receipt is not None else {}),
        **({'hold48':{'registered_count':e,'raw_in_goal_end_count':int(hold_end['raw_in_goal_end'].sum()),
            'safe_reach_goal_end_count':int(hold_end['safe_reach_goal_end'].sum()),'ever_reached_count':int(hold_end['ever_reached'].sum()),
            'arrival_end':float(hold_end['safe_reach_goal_end'].mean()),'metrics_per_request':{k:v.cpu().tolist() for k,v in hold_end.items()}}} if hold48 else {}),
        cbf_correction_metrics=correction_metrics.summary(),
        actual_physics_ticks=tick+1,
        requests=e,completed=int(locked.sum()),
        arrivals=int(arrival.sum()),failures=int(failure.sum()),censored=int((~locked).sum()),
        arrival_rate=float(arrival.float().mean()),elapsed_wall_s=time.monotonic()-started,
        fixed_scene_sha256=base.large_scene.sha256(),max_steps=max_steps,
        requests_detail=[dict(id=i,completed=bool(locked[i]),arrival=bool(arrival[i]),
            initial_distance_m=float(distance0[i]),final_distance_m=float(distance[i]),
            tail_low_speed_hold_s=float(tail_low_speed_steps[i])*.01,last5s_low_speed_hold=bool(tail_low_speed_steps[i]>=500 and locked[i] and steps[i]==max_steps and reason[i]==10),
            final_speed_m_s=float(speed[i]),
            arrival_vx_m_s=(float(arrival_velocity[i,0].item()) if bool(arrival[i]) else None),
            arrival_vy_m_s=(float(arrival_velocity[i,1].item()) if bool(arrival[i]) else None),
            arrival_vz_m_s=(float(arrival_velocity[i,2].item()) if bool(arrival[i]) else None),
            arrival_speed_3d_m_s=(float(torch.linalg.norm(arrival_velocity[i]).item()) if bool(arrival[i]) else None),
            arrival_speed_xy_m_s=(float(torch.linalg.norm(arrival_velocity[i,:2]).item()) if bool(arrival[i]) else None),
            arrival_distance_at_arrival_m=(float(arrival_distance[i].item()) if bool(arrival[i]) and math.isfinite(float(arrival_distance[i])) else None),
            episode_return=(float(episode_return[i]) if bool(locked[i]) and bool(torch.isfinite(episode_return[i])) else None),fallback_steps=int(fallback[i]),steps=int(steps[i]),termination_reason=int(reason[i]),
            saturation_steps=int(saturation[i]),
            saturation_steps_semantics='candidate converter thrust saturation; historical field retained',
            chosen_execution_thrust_saturation_steps=(int(chosen_saturation[i]) if int(steps[i])>0 and int(chosen_saturation_observed[i])==int(steps[i]) else None),
            chosen_execution_thrust_saturation_observed_steps=int(chosen_saturation_observed[i]),
            nominal_accel_mean_norm=(float(nominal_accel_sum[i]/steps[i]) if int(steps[i])>0 else None),
            commanded_accel_mean_norm=(float(commanded_accel_sum[i]/steps[i]) if int(steps[i])>0 else None),
            goal_dir_speed_mean=(float(goal_dir_speed_sum[i]/steps[i]) if int(steps[i])>0 else None),
            **passive_row(i),**{k:bool(v[i]) for k,v in flags.items()}) for i in range(e)],
        action_noise_sha256=(action_noise.rolling_sha256.hexdigest()if action_noise else None),action_noise_ticks=(action_noise.draw_count if action_noise else 0),
        action_mode=action_mode,action_seed=(int(action_seed) if action_seed is not None else None),
        action_noise_protocol=("paired request-indexed standard normal; per-tick (num_envs,1,3) draws from SeedSequence([action_seed]); constant consumption independent of terminations" if action_mode=="random" else None),
        arrival_speed_stats=arrival_speed_stats,failure_final_speed_stats=failure_final_speed_stats,
        training_state_consumed=False,denominator_preserved=True,parking_resets=parking_resets)


def noise_prefix_sha(seed,num_envs,ticks):
    """Independent CPU oracle prefix; no physical ticks or policy/RNG mutation."""
    if not isinstance(ticks,int)or not 0<ticks<=3200:raise ValueError('Invalid oracle prefix length')
    rng=np.random.default_rng(np.random.SeedSequence([int(seed)]));h=hashlib.sha256()
    for _ in range(ticks):h.update(rng.standard_normal((num_envs,1,3)).astype(np.float32).tobytes())
    return h.hexdigest()

def validate_noise_receipt(result,seed,num_envs,full_oracle):
    ticks=result['action_noise_ticks']
    expected=full_oracle if ticks==3200 else noise_prefix_sha(seed,num_envs,ticks)
    if result['action_noise_sha256']!=expected:raise RuntimeError('Action noise oracle prefix mismatch')
    return {'status':'PASS','actual_physics_ticks':ticks,'oracle_scope':'full3200'if ticks==3200 else'actual_tick_prefix','no_extra_physics':True}
