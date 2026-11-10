"""Collect and validate rule-based MetaDrive expert demonstrations."""
import numpy as np

from datasets.srairl.expert_builder import ExpertTransition


from datasets.srairl.teacher_controller import MetaDriveExpertController, TEACHER_VERSION


def _world_role_states(env):
    result=[]
    for role in ('elcv','rfv','rlv_current','rlv_target'):
        vehicle=env.world.role_vehicle(role)
        pos=np.asarray(vehicle.position,dtype=np.float64)[:2]
        vel=np.asarray(vehicle.velocity,dtype=np.float64)[:2]
        result.append([pos[0],pos[1],vel[0],vel[1]])
    return np.asarray(result,dtype=np.float64)


def _reference_role_states(env):
    """Store all actors in the same generated ELCV path/Frenet frame."""
    roles=env._roles_live()
    return np.asarray([roles[name] for name in ('elcv','rfv','rlv_current','rlv_target')],dtype=np.float64)


def _role_lane_indices(env):
    output=[]
    for role in ('elcv','rfv','rlv_current','rlv_target'):
        vehicle=env.world.role_vehicle(role)
        try:
            index=vehicle.lane_index
            output.append('/'.join(str(part) for part in index) if index is not None else 'uninitialized')
        except (AttributeError,RuntimeError):
            # MetaDrive assigns a navigation lane on the first physics tick.
            # The initial state row precedes warm-up and carries no lane ID.
            output.append('uninitialized')
    return output


def _measured_acceleration(vehicle, before_velocity, before_heading, dt):
    heading=float(before_heading)
    tangent=np.asarray([np.cos(heading),np.sin(heading)])
    normal=np.asarray([-tangent[1],tangent[0]])
    delta=(np.asarray(vehicle.velocity,dtype=np.float64)[:2]-before_velocity)/float(dt)
    return np.asarray([delta@tangent,delta@normal],dtype=np.float32)


def validate_expert_episode(raw, minimum_future_steps=100, minimum_history_steps=20):
    """Check synchronized four-role trajectories and complete temporal support."""
    errors=[]
    states=np.asarray(raw.get('role_states',[]),dtype=np.float64)
    reference=np.asarray(raw.get('reference_states_frenet',[]),dtype=np.float64)
    lane_indices=raw.get('lane_indices',[])
    times=np.asarray(raw.get('times_s',[]),dtype=np.float64)
    desired=np.asarray(raw.get('desired_acceleration',[]),dtype=np.float64)
    mapped=np.asarray(raw.get('metadrive_actions',[]),dtype=np.float64)
    measured=np.asarray(raw.get('actual_acceleration',[]),dtype=np.float64)
    tracking=np.asarray(raw.get('tracking_errors',[]),dtype=np.float64)
    n=len(times)
    if states.shape!=(n,4,4):errors.append('four_role_state_shape')
    if reference.shape!=(n,4,4) or len(lane_indices)!=n:errors.append('four_role_reference_alignment')
    if not np.isfinite(states).all() or not np.isfinite(times).all():errors.append('non_finite_or_missing_state')
    if n<minimum_history_steps+minimum_future_steps+1:errors.append('insufficient_trajectory_length')
    if len(times)>1 and (np.any(np.diff(times)<=0) or not np.allclose(np.diff(times),.1,atol=1e-5)):
        errors.append('unaligned_or_discontinuous_timestamps')
    continuity_max_error_m=0.;continuity_worst_role=None;continuity_worst_frame=None
    if states.shape==(n,4,4) and n>1:
        displacement=np.diff(states[:,:,:2],axis=0)
        average_velocity=.5*(states[:-1,:,2:4]+states[1:,:,2:4])*.1
        continuity_error=np.linalg.norm(displacement-average_velocity,axis=-1)
        worst=np.unravel_index(int(np.argmax(continuity_error)),continuity_error.shape)
        continuity_max_error_m=float(continuity_error[worst]);continuity_worst_role=raw['role_order'][worst[1]]
        continuity_worst_frame=int(worst[0])
        if continuity_max_error_m>2.0:
            errors.append('discontinuous_role_trajectory')
    action_steps=max(0,n-1-int(raw.get('warmup_steps',minimum_history_steps)))
    if desired.shape!=(action_steps,2,2) or mapped.shape!=desired.shape or measured.shape!=desired.shape:
        errors.append('action_alignment')
    elif not all(np.isfinite(x).all() for x in (desired,mapped,measured)):
        errors.append('non_finite_action')
    elif np.any(np.abs(desired)>1.5+1e-6):
        errors.append('desired_action_outside_calibrated_domain')
    completed=raw.get('lane_change_completion_index')
    if completed is not None and reference.shape==(n,4,4) and 0<=int(completed)<n:
        longitudinal=reference[int(completed),:,1]
        if not longitudinal[1]<longitudinal[0]<longitudinal[3]:
            errors.append('target_slot_not_preserved')
    if not raw.get('lane_change_success',False):errors.append('lane_change_not_successful')
    if raw.get('collision',False):errors.append('collision')
    if raw.get('out_of_road',False):errors.append('out_of_road')
    if completed is None or n-1-int(completed)<minimum_future_steps:errors.append('incomplete_future_after_lane_change')
    if measured.size and (np.max(np.abs(measured[:,:,0]))>8. or np.max(np.abs(measured[:,:,1]))>8.):
        errors.append('implausible_measured_acceleration')
    if 'tracking_errors' in raw and (tracking.shape!=(action_steps,2,2) or not np.isfinite(tracking).all()):
        errors.append('tracking_error_alignment')
    # Require RFV's measured response to vary with the interaction rather than
    # accepting an unchanged/frozen background actor as a demonstration.
    if (states.shape==(n,4,4) and desired.shape==(action_steps,2,2)
            and measured.shape==(action_steps,2,2) and len(measured)>10):
        observed_speed=np.linalg.norm(states[int(raw.get('warmup_steps',minimum_history_steps)):,1,2:4],axis=1)
        requested=desired[:,1,0]
        response_requested=bool(np.max(np.abs(requested))>.1)
        if response_requested and float(np.ptp(measured[:,1,0]))<.02 and float(np.ptp(observed_speed))<.05:
            errors.append('rfv_actuation_response_missing')
    return dict(valid=not errors,errors=errors,continuity_max_error_m=continuity_max_error_m,
                continuity_worst_role=continuity_worst_role,continuity_worst_frame=continuity_worst_frame)


def collect_metadrive_expert(env, episodes, seed, callback=None, *, vary_initial_state=False):
    """Collect physical episodes and restore caller state even after an error."""
    base=env.interaction_scenario
    previous=bool(getattr(env,'_expert_collection',False))
    try:
        return _collect_metadrive_expert(env,episodes,seed,callback,vary_initial_state=vary_initial_state)
    finally:
        env.interaction_scenario=base
        try:env.world.set_interaction_scenario(base)
        finally:env._expert_collection=previous


def _collect_metadrive_expert(env, episodes, seed, callback=None, *, vary_initial_state=False):
    """Collect paired transitions and full-world trajectory records.

    Each accepted episode contains 20 real lane-tracking warm-up frames, then a complete
    lane change followed by 100 real simulator frames. Invalid episodes remain
    available to the caller for quarantine but never enter the training data.
    """
    transitions=[];episode_rows=[];raw_episodes=[]
    base_interaction=env.interaction_scenario
    env._expert_collection=True
    dt=float(env.dt);history_steps=20;future_steps=100
    for episode in range(int(episodes)):
        if callback and not callback(dict(state='running',step=episode,total=int(episodes),episodes=episode)):
            break
        if vary_initial_state:
            from dataclasses import replace
            rng=np.random.default_rng(int(seed)+episode)
            spec=replace(base_interaction,
                rfv_x=base_interaction.elcv_x-float(rng.uniform(12.,24.)),
                rlv_current_x=base_interaction.elcv_x+float(rng.uniform(28.,48.)),
                rlv_target_x=base_interaction.elcv_x+float(rng.uniform(22.,42.)),
                rfv_speed=float(np.clip(base_interaction.rfv_speed+rng.uniform(-1.,1.),1.,25.)),
                rlv_current_speed=float(np.clip(base_interaction.rlv_current_speed+rng.uniform(-1.,1.),1.,25.)),
                rlv_target_speed=float(np.clip(base_interaction.rlv_target_speed+rng.uniform(-1.,1.),1.,25.)))
            env.interaction_scenario=spec.validate(env.lane_num)
            env.world.set_interaction_scenario(spec)
        observations,_=env.reset(seed=int(seed)+episode)
        controller=MetaDriveExpertController(env)
        done=False;step=0;collisions=0;offroad=0;distance=0.;started=False;completed=False
        initial_gap=float(env._initial_gap);final_gap=initial_gap;time_to_lane_change=None
        initial_rfv_speed=float(env._rfv_initial_speed);final_rfv_speed=initial_rfv_speed;rfv_min_accel=0.
        raw_states=[_world_role_states(env)];reference_states=[_reference_role_states(env)]
        lane_indices=[_role_lane_indices(env)];times=[-history_steps*dt]
        controller.set_warmup(True)
        for warm_step in range(history_steps):
            observations,warm_info=env.step({role:controller.action(role) for role in ('elcv','rfv')})
            raw_states.append(_world_role_states(env));reference_states.append(_reference_role_states(env))
            lane_indices.append(_role_lane_indices(env));times.append((warm_step+1-history_steps)*dt)
            collisions+=int(bool(warm_info.get('metadrive_primary_collision')))
            offroad+=int(bool(warm_info.get('metadrive_out_of_road')))
            if warm_info.get('metadrive_primary_collision') or warm_info.get('metadrive_out_of_road'):
                done=True;break
        if not done:
            env.begin_evaluation_phase(history_steps)
            controller.set_warmup(False)
            initial_gap=float(env._initial_gap);final_gap=initial_gap
            initial_rfv_speed=float(env._rfv_initial_speed);final_rfv_speed=initial_rfv_speed
        # Warm-up initializes temporal history; exclude it from interaction
        # response acceleration statistics.
        env._rfv_min_accel=0.
        episode_transitions=[];desired_actions=[];mapped_actions=[];measured_actions=[];tracking_errors=[]
        completion_index=None;post_completion=0;events=[];teacher_phases=[];interrupted=False
        while not done and (completion_index is None or post_completion<future_steps):
            state=observations['elcv'].copy()
            actions={role:controller.action(role) for role in ('elcv','rfv')}
            if controller.phase=='refused':
                done=True;break
            tracking_errors.append(np.asarray([controller.tracking_error(env.world.role_vehicle(role),role)
                                               for role in ('elcv','rfv')],dtype=np.float32))
            teacher_phases.append(controller.phase)
            vehicles={role:env.world.role_vehicle(role) for role in ('elcv','rfv')}
            before={role:(np.asarray(vehicles[role].velocity,dtype=np.float64)[:2].copy(),
                          float(vehicles[role].heading_theta)) for role in vehicles}
            mapped={role:np.asarray(env._to_metadrive_action(vehicles[role],actions[role]),dtype=np.float32) for role in vehicles}
            next_observations,info=env.step(actions)
            next_state=next_observations['elcv'].copy();done=bool(info['done'])
            actual=np.stack([_measured_acceleration(vehicles[role],before[role][0],before[role][1],dt)
                             for role in ('elcv','rfv')])
            episode_transitions.extend(ExpertTransition(state.copy(),actions[role].copy(),next_state.copy(),done,
                    role,f"metadrive-{env.scenario_type}-seed{int(seed)+episode}",step) for role in ('elcv','rfv'))
            desired_actions.append(np.stack([actions['elcv'],actions['rfv']]))
            mapped_actions.append(np.stack([mapped['elcv'],mapped['rfv']]))
            measured_actions.append(actual)
            raw_states.append(_world_role_states(env));reference_states.append(_reference_role_states(env))
            lane_indices.append(_role_lane_indices(env));times.append((step+1)*dt)
            step+=1;collisions+=int(bool(info['metadrive_primary_collision']));offroad+=int(bool(info['metadrive_out_of_road']))
            started|=bool(info['lane_change_started']);completed|=bool(info['lane_change_completed'])
            if info['lane_change_completed'] and completion_index is None:completion_index=len(raw_states)-1
            elif completion_index is not None:post_completion+=1
            if info['lane_change_started'] and not any(event['type']=='maneuver_start' for event in events):
                events.append({'type':'maneuver_start','time_s':float(info['time'])})
            if info['lane_change_completed'] and completion_index==len(raw_states)-1:
                events.append({'type':'maneuver_complete','time_s':float(info['time'])})
            if info['metadrive_primary_collision']:events.append({'type':'collision','time_s':float(info['time'])})
            if info['metadrive_out_of_road']:events.append({'type':'out_of_road','time_s':float(info['time'])})
            final_gap=float(info['target_gap_m']);time_to_lane_change=info.get('time_to_lane_change')
            final_rfv_speed=float(info['rfv_speed_mps']);rfv_min_accel=min(rfv_min_accel,float(info['rfv_min_accel_mps2']))
            distance=float(info['distance_travelled_m']);observations=next_observations
            if callback and not callback(dict(state='running',step=episode,total=int(episodes),episodes=episode,
                    episode_step=step,global_steps=(len(transitions)+len(episode_transitions))//2,
                    traffic_vehicle_count=info['traffic_vehicle_count'],teacher_phase=controller.phase)):
                interrupted=True;break
        task_success=bool(info.get('merge_success',completed and not collisions and not offroad)) if step else False
        raw=dict(role_order=['elcv','rfv','rlv_current','rlv_target'],role_states=np.asarray(raw_states),
            reference_states_frenet=np.asarray(reference_states),lane_indices=lane_indices,
            times_s=np.asarray(times),desired_acceleration=np.asarray(desired_actions,dtype=np.float32).reshape(-1,2,2),
            metadrive_actions=np.asarray(mapped_actions,dtype=np.float32).reshape(-1,2,2),actual_acceleration=np.asarray(measured_actions,dtype=np.float32).reshape(-1,2,2),
            tracking_errors=np.asarray(tracking_errors,dtype=np.float32).reshape(-1,2,2),
            lane_change_completion_index=completion_index,lane_change_success=task_success,
            collision=bool(collisions),out_of_road=bool(offroad),seed=int(seed)+episode,scenario=env.scenario_type,
            warmup_steps=history_steps,minimum_future_steps=future_steps)
        validation=validate_expert_episode(raw,future_steps,history_steps)
        from dataclasses import asdict
        from env.tasks.srairl_interaction import ROAD_GEOMETRY_CONFIG, REFERENCE_FRAME_VERSION
        raw['initial_interaction']=asdict(env.interaction_scenario)
        raw['initial_variation_version']='gap-relative-speed-v1' if vary_initial_state else 'canonical'
        raw['reference_frame_version']=REFERENCE_FRAME_VERSION
        raw['road_geometry']=dict(ROAD_GEOMETRY_CONFIG)
        raw['teacher_version']=TEACHER_VERSION
        raw['teacher_config']=controller.configuration()
        raw['teacher_phases']=teacher_phases
        raw['teacher_events']=controller.events
        raw['teacher_gap_diagnostics']=controller.gap_diagnostics
        raw['teacher_merge_gap_requested']=controller.merge_gap_requested
        raw['teacher_emergency_braking_count']=controller.emergency_braking_count
        raw['teacher_command_saturation_count']=controller.command_saturation_count
        outcome=('interrupted' if interrupted else 'collision' if collisions else 'out_of_road' if offroad else
            controller.refusal_reason if controller.phase=='refused' else 'accepted' if validation['valid'] else
            'incomplete_maneuver' if not task_success else 'trajectory_quality_rejected')
        raw['teacher_outcome']=outcome
        if controller.phase=='refused':
            validation['errors'].append('teacher_'+str(controller.refusal_reason));validation['valid']=False
        raw['action_calibration_revision']=getattr(env.action_adapter,'calibration_revision',None)
        raw['time_origin_version']='post-history-t0-v1'
        raw['warmup_version']='gap-aware-history-20-frames-v2'
        raw['split_version']='scenario_grouped_seed_70_15_15_v1'
        raw['speed_config']={key:value for key,value in asdict(env.interaction_scenario).items() if key.endswith('_speed')}
        raw['validation']=validation
        raw['events']=events
        raw['quality_metrics']={'lane_change_success':task_success,'collision':bool(collisions),
            'out_of_road':bool(offroad),'continuous_path_error_m':validation['continuity_max_error_m'],
            'minimum_future_steps_after_maneuver':None if completion_index is None else len(times)-1-completion_index}
        raw_episodes.append(raw)
        if validation['valid']:
            if len(episode_transitions)>=2:
                for index in (-2,-1):
                    final=episode_transitions[index]
                    episode_transitions[index]=ExpertTransition(final.state,final.action,final.next_state,True,
                        final.agent,final.scenario_id,final.step_index)
            transitions.extend(episode_transitions)
        episode_rows.append(dict(episode=episode+1,seed=int(seed)+episode,scenario=env.scenario_type,steps=step,
            distance_travelled_m=distance,collision=bool(collisions),out_of_road=bool(offroad),
            lane_change_started=started,lane_change_completed=completed,
            merge_success=task_success,time_to_lane_change_s=time_to_lane_change,
            accepted=validation['valid'],rejection_reasons=validation['errors'],
            teacher_version=TEACHER_VERSION,teacher_outcome=outcome,teacher_final_phase=controller.phase,
            teacher_wait_s=next((event['time_s'] for event in controller.events if event.get('phase')=='executing'),float(env.time)),
            teacher_emergency_braking_count=controller.emergency_braking_count,
            warmup_steps=history_steps,post_lane_change_steps=post_completion,
            initial_target_gap_m=initial_gap,final_target_gap_m=final_gap,target_gap_change_m=final_gap-initial_gap,
            delta_v_rfv_mps=final_rfv_speed-initial_rfv_speed,rfv_min_accel_mps2=rfv_min_accel))
        if callback and not callback(dict(state='running',step=episode+1,total=int(episodes),episodes=episode+1,
                global_steps=len(transitions)//2,episode_steps=step,collision=bool(collisions),out_of_road=bool(offroad))):break
    return transitions,episode_rows,raw_episodes


def save_raw_episode(path, raw):
    """Persist a single raw simulator episode without Python object arrays."""
    from pathlib import Path
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,role_order=np.asarray(raw['role_order'],dtype='U32'),
        role_states=np.asarray(raw['role_states'],dtype=np.float64),times_s=np.asarray(raw['times_s'],dtype=np.float64),
        reference_states_frenet=np.asarray(raw['reference_states_frenet'],dtype=np.float64),
        lane_indices=np.asarray(raw['lane_indices'],dtype='U64'),
        desired_acceleration=np.asarray(raw['desired_acceleration'],dtype=np.float32),
        metadrive_actions=np.asarray(raw['metadrive_actions'],dtype=np.float32),
        actual_acceleration=np.asarray(raw['actual_acceleration'],dtype=np.float32),
        tracking_errors=np.asarray(raw['tracking_errors'],dtype=np.float32),
        metadata=np.asarray(__import__('json').dumps({k:v for k,v in raw.items() if k not in {
            'role_order','role_states','reference_states_frenet','lane_indices','times_s',
            'desired_acceleration','metadrive_actions','actual_acceleration','tracking_errors'}},
            ensure_ascii=False,allow_nan=False)))
