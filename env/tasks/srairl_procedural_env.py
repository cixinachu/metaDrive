"""Procedural MetaDrive interaction environment for the two SR-AIRL actors."""
from types import SimpleNamespace
import numpy as np

from env.tasks.srairl_env import SRAIRLEnv
from env.adapters.srairl_observation import SRAIRLObservationAdapter
from env.tasks.srairl_metadrive_world import SRAIRLProceduralWorld


class SRAIRLProceduralEnv(SRAIRLEnv):
    """Two policy-controlled vehicles with arbitrary simulator traffic.

    SR-AIRL keeps its upstream two-actor ELCV/RFV formulation. MetaDrive may
    add multiple IDM background vehicles, but those are not learned policies.
    """

    def __init__(self, scenario_type="straight", lane_num=3, traffic_density=.15,
                 seed=0, horizon=1000, view_mode="topdown", config=None, traffic_seed=None,
                 interaction_scenario=None, rfv_response="policy",initial_state_samples=None):
        from env.tasks.srairl_interaction import SRAIRLInteractionScenario
        if interaction_scenario is None:
            interaction_scenario=SRAIRLInteractionScenario()
            if scenario_type in ('curve','merge'):
                from dataclasses import replace
                # The generated C and ramp blocks cannot be safely tracked at
                # canonical highway speeds inside the calibrated action range.
                interaction_scenario=replace(interaction_scenario,elcv_speed=6.,rfv_speed=6.,
                    rlv_current_speed=6.,rlv_target_speed=6.)
        self.interaction_scenario=interaction_scenario.validate(lane_num)
        self.initial_state_samples=[dict(item) for item in (initial_state_samples or [])]
        if self.initial_state_samples:
            if scenario_type=='merge':raise ValueError('NGSIM mainline samples cannot initialize a ramp-merge map')
            for item in self.initial_state_samples:SRAIRLInteractionScenario.from_ngsim_initial_state(item).validate(lane_num)
        self._initial_state_rng=np.random.default_rng(int(seed)+0x5A17)
        self.current_initial_state=None
        if rfv_response not in {'policy','fixed_speed','scripted_yield','scripted_gap_closing'}:
            raise ValueError(f'unsupported RFV response: {rfv_response}')
        self.rfv_response=rfv_response
        self.seed=int(seed)
        self.scenario_type=scenario_type
        self.lane_num=int(lane_num)
        self._is_merge=scenario_type=='merge'
        self.scene_source="procedural"
        self._warmup_policies={}
        self._episode_horizon=int(horizon)
        self.max_steps=self._episode_horizon
        states=np.asarray([
            [-.2,8.,0.,18.,0.],[-.1,8.,0.,18.,0.],
            [self.max_steps*.1,200.,0.,18.,0.],[(self.max_steps+1)*.1,202.,0.,18.,0.]
        ],dtype=float)
        tracks={str(i):{"id":str(i),"length":4.5,"width":1.8,"states":states.copy()} for i in range(4)}
        placeholder=SimpleNamespace(scenario_id=f"metadrive-{scenario_type}",elcv_id="0",rfv_id="1",
            rlv_current_id="2",rlv_target_id="3",tracks=tracks,start_time=0.,end_time=(self.max_steps+1)*.1)
        world=SRAIRLProceduralWorld(scenario_type,lane_num,traffic_density,view_mode,seed,horizon+20,traffic_seed,
                                   self.interaction_scenario)
        super().__init__(placeholder,dynamics_mode="metadrive_physics",config=config,metadrive_world=world)
        self.observer=SRAIRLObservationAdapter(self.config.history_len,
            pos_lat_scale=self.config.pos_lat_scale,pos_long_scale=self.config.pos_long_scale,
            vel_scale=self.config.vel_scale,vel_offset=self.config.vel_offset)
        self._previous_progress=0.
        self._last_info={}

    @staticmethod
    def _local_state(vehicle, reference_lane):
        x,y=map(float,vehicle.position[:2]);vx,vy=map(float,vehicle.velocity[:2])
        longitudinal,lateral=reference_lane.local_coordinates((x,y))
        heading=float(reference_lane.heading_theta_at(longitudinal))
        tangent=np.asarray([np.cos(heading),np.sin(heading)])
        normal=np.asarray([-tangent[1],tangent[0]])
        velocity=np.asarray([vx,vy])
        return np.asarray([float(lateral),float(longitudinal),float(velocity@tangent),float(velocity@normal)])

    @staticmethod
    def _path_state(vehicle, lane_path):
        """Project a vehicle onto a continuous generated lane path."""
        position=np.asarray(vehicle.position,dtype=np.float64)[:2]
        velocity=np.asarray(vehicle.velocity,dtype=np.float64)[:2]
        best=None;offset=0.
        for lane in lane_path:
            local_s,lateral=lane.local_coordinates(position)
            clipped=float(np.clip(local_s,0.,lane.length))
            center=np.asarray(lane.position(clipped,0.),dtype=np.float64)
            distance=float(np.linalg.norm(position-center))
            candidate=(distance,offset+clipped,float(lateral),lane,clipped)
            if best is None or candidate[0]<best[0]:best=candidate
            offset+=float(lane.length)
        _,longitudinal,lateral,lane,s=best
        heading=float(lane.heading_theta_at(s));tangent=np.asarray([np.cos(heading),np.sin(heading)])
        normal=np.asarray([-tangent[1],tangent[0]])
        return np.asarray([lateral,longitudinal,float(velocity@tangent),float(velocity@normal)])

    def _vehicle_pool(self):
        vehicles=list(self.world.agents.values())
        vehicles.extend(v for v in getattr(self.world,'interaction_vehicles',{}).values() if v not in vehicles)
        manager=self.world.engine.traffic_manager
        vehicles.extend(v for v in getattr(manager,"traffic_vehicles",[]) if v not in vehicles)
        return vehicles

    def _roles_live(self):
        ego=self.world.role_vehicle("elcv")
        other=self.world.role_vehicle("rfv")
        path=self._merge_reference_path if self._is_merge else self._source_lane_path
        def state(vehicle):
            result=self._path_state(vehicle,path)
            result[1]-=self._world_path_offset
            return result
        ego_state=state(ego);other_state=state(other)
        current=state(self.world.role_vehicle('rlv_current'))
        target=state(self.world.role_vehicle('rlv_target'))
        return {"elcv":ego_state,"rfv":other_state,"rlv_current":current,"rlv_target":target}

    def reset(self,seed=None):
        self._close_warmup_policies()
        self._active_seed=self.seed if seed is None else int(seed)
        self.time=0.;self.steps=0;self.max_steps=self._episode_horizon
        self._done=False;self._metadrive_collision=False;self._metadrive_out_of_road=False
        if self.initial_state_samples:
            from env.tasks.srairl_interaction import SRAIRLInteractionScenario
            if seed is not None:self._initial_state_rng=np.random.default_rng(int(seed)+0x5A17)
            self.current_initial_state=self.initial_state_samples[int(self._initial_state_rng.integers(len(self.initial_state_samples)))]
            self.interaction_scenario=SRAIRLInteractionScenario.from_ngsim_initial_state(self.current_initial_state).validate(self.lane_num)
            self.world.set_interaction_scenario(self.interaction_scenario)
        else:self.current_initial_state=None
        # MetaDrive interprets reset(seed=...) as an absolute scenario index
        # in [start_seed, start_seed + num_scenarios). Seed every episode
        # explicitly so another map cannot leak global RNG state into it.
        stale=[v.id for v in getattr(self.world,'interaction_vehicles',{}).values()]
        if stale:self.world.engine.clear_objects(stale,force_destroy=True)
        self.world.interaction_vehicles={}
        _,world_info=self.world.reset(seed=int(self._active_seed))
        if self._is_merge:
            lanes=self.world.current_map.road_network.graph['1r0_0_']['1r0_1_']
            self._source_lane=self._target_lane=lanes[2]
            self.world._interaction_s_offset=0.
        else:
            lane_a,lane_b=self.world.role_vehicle('elcv').lane_index[:2]
            road_lanes=self.world.current_map.road_network.graph[lane_a][lane_b]
            self._source_lane=road_lanes[self.interaction_scenario.source_lane]
            self._target_lane=road_lanes[self.interaction_scenario.target_lane]
        self.world.spawn_interaction_roles()
        if self._is_merge:
            ego=self.world.role_vehicle('elcv')
            self._merge_reference_path=self.world._lane_path(ego.lane)
            self._world_path_offset=self._path_state(ego,self._merge_reference_path)[1]-self.interaction_scenario.elcv_x
        else:
            self._source_lane_path=self.world._interaction_lane_paths[self.interaction_scenario.source_lane]
            self._target_lane_path=self.world._interaction_lane_paths[self.interaction_scenario.target_lane]
            self._world_path_offset=float(self.world._interaction_s_offset)
        self.world.set_role_colors()
        roles=self._roles_live()
        obs=self.observer.reset(roles)
        self.distance_travelled_m=0.
        self._last_position=np.asarray(self.world.role_vehicle("elcv").position[:2],dtype=float)
        self.vehicles={"0":dict(zip(("lat","lon","v_long","v_lat"),roles["elcv"])),
                       "1":dict(zip(("lat","lon","v_long","v_lat"),roles["rfv"]))}
        self._lane_change_started=False;self._lane_change_completed=False
        self._lane_change_stable=0;self._lane_change_start_time=None;self._lane_change_completion_time=None
        self._merge_success=False
        self._rfv_initial_speed=float(np.linalg.norm(self.world.role_vehicle('rfv').velocity))
        self._fixed_rfv_speed=self._rfv_initial_speed
        self._rfv_min_accel=0.
        self._initial_gap=float(roles['rlv_target'][1]-roles['rfv'][1])
        if self._is_merge:
            ego=self.world.role_vehicle('elcv');rfv=self.world.role_vehicle('rfv')
            self._merge_initial_route=(ego.lane_index[:2],rfv.lane_index[:2])
            self._merge_start_x=float(ego.position[0])
            self._merge_complete_x=None
            self._evaluation_start_s=self._path_state(ego,self._merge_reference_path)[1]-self._world_path_offset
            self._evaluation_end_s=self._evaluation_start_s+85.
        else:
            self._evaluation_start_s=float(self.interaction_scenario.elcv_x)
            self._evaluation_end_s=float(self.interaction_scenario.interaction_end_x)
        self._last_info={"done":False,"time":0.,"distance_travelled_m":0.,"tracking_reward":0.,
                         "collision_proxy_any":False,"controlled_collision_proxy":False,
                         "metadrive_primary_collision":False,"metadrive_out_of_road":False,
                         "minimum_distance_m":None,"traffic_vehicle_count":max(0,len(self._vehicle_pool())-4)}
        if self.frame_callback:self.frame_callback(self,self._last_info)
        return {"elcv":obs.copy(),"rfv":obs.copy()},dict(scenario_id=self.scenario.scenario_id,
            dynamics_mode="metadrive_physics",scene_source="procedural",**world_info)

    def begin_evaluation_phase(self, warmup_steps=20):
        """Start measured episode time after collecting physical history frames."""
        if self._done:
            raise RuntimeError("cannot start evaluation after a terminated warm-up")
        if int(warmup_steps) < 1:
            raise ValueError("procedural evaluation requires at least one warm-up step")
        self._close_warmup_policies()
        ego=self.world.role_vehicle("elcv");rfv=self.world.role_vehicle("rfv")
        self.steps=0;self.time=0.;self._done=False
        self.max_steps+=int(warmup_steps)
        self.distance_travelled_m=0.
        self._last_position=np.asarray(ego.position[:2],dtype=float)
        self._previous_progress=0.
        self._lane_change_started=False;self._lane_change_completed=False
        self._lane_change_stable=0;self._lane_change_start_time=None;self._lane_change_completion_time=None
        self._merge_success=False
        self._metadrive_collision=False;self._metadrive_out_of_road=False
        self._rfv_initial_speed=float(np.linalg.norm(rfv.velocity))
        self._fixed_rfv_speed=self._rfv_initial_speed
        self._rfv_min_accel=0.
        roles=self._roles_live()
        self._initial_gap=float(roles["rlv_target"][1]-roles["rfv"][1])
        if self._is_merge:
            self._merge_initial_route=(ego.lane_index[:2],rfv.lane_index[:2])
            self._merge_start_x=float(ego.position[0])
            self._merge_complete_x=None
            self._evaluation_start_s=self._path_state(ego,self._merge_reference_path)[1]-self._world_path_offset
            self._evaluation_end_s=self._evaluation_start_s+85.
        else:
            ego_s=float(self._roles_live()['elcv'][1])
            self._evaluation_start_s=float(ego_s)
            span=float(self.interaction_scenario.interaction_end_x-self.interaction_scenario.elcv_x)
            self._evaluation_end_s=self._evaluation_start_s+span
        self._last_info={"done":False,"time":0.,"distance_travelled_m":0.,
                         "tracking_reward":0.,"collision_proxy_any":False,
                         "controlled_collision_proxy":False,"metadrive_primary_collision":False,
                         "metadrive_out_of_road":False,"minimum_distance_m":None,
                         "traffic_vehicle_count":max(0,len(self._vehicle_pool())-4),
                         "evaluation_start_s":self._evaluation_start_s,
                         "warmup_steps":int(warmup_steps)}
        return dict(evaluation_start_s=self._evaluation_start_s,
                    evaluation_end_s=self._evaluation_end_s,warmup_steps=int(warmup_steps))

    def _close_warmup_policies(self):
        for policy in self._warmup_policies.values():
            policy.destroy()
        self._warmup_policies={}

    def warmup_step(self):
        """Advance controlled vehicles with MetaDrive IDM for physical history."""
        from metadrive.policy.idm_policy import IDMPolicy
        roles=("elcv","rfv")
        for index,role in enumerate(roles):
            if role not in self._warmup_policies:
                vehicle=self.world.role_vehicle(role)
                policy=IDMPolicy(vehicle,self._active_seed+index)
                policy.enable_lane_change=False
                target_speed={'curve':6.,'merge':6.}.get(self.scenario_type,22.)
                policy.target_speed=target_speed*3.6
                self._warmup_policies[role]=policy
        controls={role:np.asarray(self._warmup_policies[role].act(),dtype=np.float32)
                  for role in roles}
        self._warmup_control_override=controls
        try:
            return self.step({role:np.zeros(2,dtype=float) for role in roles})
        finally:
            self._warmup_control_override=None

    def step(self,actions):
        if self._done:raise RuntimeError("call reset after the interaction episode ends")
        if set(actions)!={"elcv","rfv"}:raise ValueError("actions must contain exactly 'elcv' and 'rfv'")
        policy_actions={role:np.asarray(actions[role],dtype=float).copy() for role in ('elcv','rfv')}
        rfv_speed_before=float(np.linalg.norm(self.world.role_vehicle('rfv').velocity))
        ego=self.world.role_vehicle('elcv');target_lane=self._target_lane
        reference_path=self._merge_reference_path if self._is_merge else self._source_lane_path
        ego_source_longitudinal=self._path_state(ego,reference_path)[1]-self._world_path_offset
        if self.rfv_response=='fixed_speed':policy_actions['rfv']=np.asarray([0.,0.])
        elif self.rfv_response=='scripted_yield' and self._lane_change_started:policy_actions['rfv'][0]=-1.
        elif self.rfv_response=='scripted_gap_closing' and self._lane_change_started:policy_actions['rfv'][0]=.75
        override=getattr(self,"_warmup_control_override",None)
        controls=override if override is not None else {
            role:self._to_metadrive_action(self.world.role_vehicle(role),policy_actions[role])
            for role in ("elcv","rfv")}
        _,_,terminated,truncated,world_info=self.world.step(controls)
        if self.rfv_response=='fixed_speed':
            vehicle=self.world.role_vehicle('rfv');lane=vehicle.lane
            s,_=lane.local_coordinates(vehicle.position[:2]);heading=float(lane.heading_theta_at(s))
            vehicle.set_velocity((self._fixed_rfv_speed*np.cos(heading),self._fixed_rfv_speed*np.sin(heading)))
        self.steps+=1;self.time=self.steps*self.dt
        roles=self._roles_live();obs=self.observer.append(roles)
        rfv_speed_after=float(np.linalg.norm(self.world.role_vehicle('rfv').velocity))
        rfv_actual_accel=(rfv_speed_after-rfv_speed_before)/max(self.dt,1e-6)
        self._rfv_min_accel=min(self._rfv_min_accel,rfv_actual_accel)
        if self._is_merge:
            merge_node=ego.lane_index[0] in ('1r0_0_','1r0_1_')
            if not self._lane_change_started and ego_source_longitudinal>=self._evaluation_start_s+10.:
                self._lane_change_started=True;self._lane_change_start_time=self.time
            in_shared_target_lane=merge_node and ego.lane_index[2]==2
            self._lane_change_stable=self._lane_change_stable+1 if self._lane_change_started and in_shared_target_lane else 0
        else:
            lateral_from_source=abs(self._path_state(ego,self._source_lane_path)[0])
            if not self._lane_change_started and lateral_from_source>=.25 and self._evaluation_start_s<=ego_source_longitudinal<=self._evaluation_end_s:
                self._lane_change_started=True;self._lane_change_start_time=self.time
            ego_target_lateral=self._path_state(ego,self._target_lane_path)[0]
            self._lane_change_stable=self._lane_change_stable+1 if self._lane_change_started and abs(float(ego_target_lateral))<=self.interaction_scenario.success_lateral_tolerance_m else 0
        if self._lane_change_stable>=self.interaction_scenario.success_stable_steps and not self._lane_change_completed:
            self._lane_change_completed=True;self._lane_change_completion_time=self.time
            if self._is_merge:
                self._merge_complete_x=float(ego.position[0]);self._merge_success=True
        ego_zone_s=ego_source_longitudinal
        position=np.asarray(self.world.role_vehicle("elcv").position[:2],dtype=float)
        self.distance_travelled_m+=float(np.linalg.norm(position-self._last_position));self._last_position=position
        tracking_reward=float(np.clip(roles["elcv"][2]/30.,-1.,1.))
        agents=[self.world.role_vehicle("elcv"),self.world.role_vehicle("rfv")]
        collision=any(bool(v.crash_vehicle or v.crash_object) for v in agents)
        out_of_road=any(bool(v.on_yellow_continuous_line or not v.on_lane) for v in agents)
        self._metadrive_collision|=collision;self._metadrive_out_of_road|=out_of_road
        pool=self._vehicle_pool();distances=[];ttcs=[]
        for actor in agents:
            for other in pool:
                if other is actor:continue
                d=float(np.linalg.norm(np.asarray(actor.position[:2])-np.asarray(other.position[:2])))
                distances.append(d)
                actor_state=self._path_state(actor,reference_path)
                other_state=self._path_state(other,reference_path)
                lateral_clearance=abs(other_state[0]-actor_state[0])
                if lateral_clearance>(actor.WIDTH+other.WIDTH)*.5+.75:continue
                gap=other_state[1]-actor_state[1];closing=actor_state[2]-other_state[2]
                if gap>0. and closing>1e-6:ttcs.append(max(0.,gap-(actor.LENGTH+other.LENGTH)*.5)/closing)
        zone_end=self._evaluation_end_s
        zone_start=self._evaluation_start_s
        zone_exited=ego_zone_s>=zone_end
        merge_success=bool(self._lane_change_completed and (not self._is_merge or self._merge_success) and
                            not collision and not out_of_road)
        # Expert collection keeps the simulator running after task-zone exit so
        # the saved demonstration includes a real 10 s future tail.
        done=self.steps>=self.max_steps or (zone_exited and not getattr(self,"_expert_collection",False)) or collision or out_of_road or bool(terminated.get("__all__",False))
        self._done=bool(done)
        info={"done":self._done,"time":self.time,"scenario_id":self.scenario.scenario_id,
              "tracking_reward":tracking_reward,"distance_travelled_m":self.distance_travelled_m,
              "collision_proxy_any":collision,"controlled_collision_proxy":bool(
                  np.linalg.norm(np.asarray(agents[0].position[:2])-np.asarray(agents[1].position[:2]))<
                  (agents[0].LENGTH+agents[1].LENGTH)*.5),
              "metadrive_primary_collision":collision,"metadrive_out_of_road":out_of_road,
              "minimum_distance_m":min(distances) if distances else None,
              "minimum_ttc_s":min(ttcs) if ttcs else None,
              "traffic_vehicle_count":max(0,len(pool)-4),"scene_source":"procedural",
              "scenario_type":self.scenario_type,"terminated":terminated,"truncated":truncated,
              "initialization_protocol":"ngsim_distribution" if self.current_initial_state else "canonical",
              "initial_state_time_s":None if self.current_initial_state is None else float(self.current_initial_state.get('time_s',0.)),
              "initial_state_dataset_id":None if self.current_initial_state is None else self.current_initial_state.get('dataset_id'),
              "lane_change_started":self._lane_change_started,"lane_change_completed":self._lane_change_completed,
              "merge_success":merge_success,
              "lane_change_success":merge_success,
              "interaction_zone_exited":zone_exited,
              "lane_change_start_time":self._lane_change_start_time,
              "time_to_lane_change":None if self._lane_change_completion_time is None or self._lane_change_start_time is None else self._lane_change_completion_time-self._lane_change_start_time,
              "route_completion":float(np.clip((ego_zone_s-zone_start)/(zone_end-zone_start),0.,1.)),
              "rfv_speed_mps":rfv_speed_after,
              "delta_v_rfv_mps":rfv_speed_after-rfv_speed_before,
              "rfv_min_accel_mps2":self._rfv_min_accel,
              "rfv_longitudinal_accel_mps2":float(policy_actions['rfv'][0]),
              "target_gap_m":float(roles['rlv_target'][1]-roles['rfv'][1]),
              "target_gap_change_m":float(roles['rlv_target'][1]-roles['rfv'][1]-self._initial_gap),
              "merge_route_valid":bool(self._merge_initial_route[0][0].startswith('1r1_') and
                  self._merge_initial_route[1][0]=='>>>' and self._lane_change_completed) if self._is_merge else None,
              "merge_shared_lane":bool(self.world.role_vehicle('elcv').lane_index[0] in ('1r0_0_','1r0_1_') and
                  self.world.role_vehicle('elcv').lane_index[2]==2) if self._is_merge else None}
        self._last_info=info
        if self.frame_callback:self.frame_callback(self,info)
        return {"elcv":obs.copy(),"rfv":obs.copy()},info

    def close(self):
        self._close_warmup_policies()
        if self.world is not None:self.world.close()
