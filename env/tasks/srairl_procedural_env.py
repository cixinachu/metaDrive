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
        self.interaction_scenario=(interaction_scenario or SRAIRLInteractionScenario()).validate(lane_num)
        self.initial_state_samples=[dict(item) for item in (initial_state_samples or [])]
        if self.initial_state_samples:
            if scenario_type=='merge':raise ValueError('NGSIM mainline samples cannot initialize a ramp-merge map')
            for item in self.initial_state_samples:SRAIRLInteractionScenario.from_ngsim_initial_state(item).validate(lane_num)
        self._initial_state_rng=np.random.default_rng(int(seed)+0x5A17)
        self.current_initial_state=None
        if rfv_response not in {'policy','fixed_speed','scripted_yield','scripted_gap_closing'}:
            raise ValueError(f'unsupported RFV response: {rfv_response}')
        self.rfv_response=rfv_response
        self.scenario_type=scenario_type
        self.lane_num=int(lane_num)
        self._is_merge=scenario_type=='merge'
        self.scene_source="procedural"
        self.max_steps=int(horizon)
        states=np.asarray([
            [-.2,8.,0.,18.,0.],[-.1,8.,0.,18.,0.],
            [self.max_steps*.1,200.,0.,18.,0.],[(self.max_steps+1)*.1,202.,0.,18.,0.]
        ],dtype=float)
        tracks={str(i):{"id":str(i),"length":4.5,"width":1.8,"states":states.copy()} for i in range(4)}
        placeholder=SimpleNamespace(scenario_id=f"metadrive-{scenario_type}",elcv_id="0",rfv_id="1",
            rlv_current_id="2",rlv_target_id="3",tracks=tracks,start_time=0.,end_time=(self.max_steps+1)*.1)
        world=SRAIRLProceduralWorld(scenario_type,lane_num,traffic_density,view_mode,seed,horizon,traffic_seed,
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

    def _vehicle_pool(self):
        vehicles=list(self.world.agents.values())
        vehicles.extend(v for v in getattr(self.world,'interaction_vehicles',{}).values() if v not in vehicles)
        manager=self.world.engine.traffic_manager
        vehicles.extend(v for v in getattr(manager,"traffic_vehicles",[]) if v not in vehicles)
        return vehicles

    def _roles_live(self):
        ego=self.world.role_vehicle("elcv")
        other=self.world.role_vehicle("rfv")
        if self._is_merge:
            def global_state(vehicle):
                x,y=map(float,vehicle.position[:2]);vx,vy=map(float,vehicle.velocity[:2])
                return np.asarray([y,x,vx,vy],dtype=float)
            return {"elcv":global_state(ego),"rfv":global_state(other),
                    "rlv_current":global_state(self.world.role_vehicle('rlv_current')),
                    "rlv_target":global_state(self.world.role_vehicle('rlv_target'))}
        lane=self._source_lane
        ego_state=self._local_state(ego,lane)
        other_state=self._local_state(other,lane)
        current=self._local_state(self.world.role_vehicle('rlv_current'),lane)
        target=self._local_state(self.world.role_vehicle('rlv_target'),lane)
        return {"elcv":ego_state,"rfv":other_state,"rlv_current":current,"rlv_target":target}

    def reset(self,seed=None):
        self.time=0.;self.steps=0;self._done=False;self._metadrive_collision=False;self._metadrive_out_of_road=False
        if self.initial_state_samples:
            from env.tasks.srairl_interaction import SRAIRLInteractionScenario
            if seed is not None:self._initial_state_rng=np.random.default_rng(int(seed)+0x5A17)
            self.current_initial_state=self.initial_state_samples[int(self._initial_state_rng.integers(len(self.initial_state_samples)))]
            self.interaction_scenario=SRAIRLInteractionScenario.from_ngsim_initial_state(self.current_initial_state).validate(self.lane_num)
            self.world.set_interaction_scenario(self.interaction_scenario)
        else:self.current_initial_state=None
        # MetaDrive's explicit reset(seed=...) argument is a scenario index,
        # while the requested RNG seed is configured as start_seed at creation.
        stale=[v.id for v in getattr(self.world,'interaction_vehicles',{}).values()]
        if stale:self.world.engine.clear_objects(stale,force_destroy=True)
        self.world.interaction_vehicles={}
        _,world_info=self.world.reset()
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
        self.world.set_role_colors()
        roles=self._roles_live()
        obs=self.observer.reset(roles)
        self.distance_travelled_m=0.
        self._last_position=np.asarray(self.world.role_vehicle("elcv").position[:2],dtype=float)
        self.vehicles={"0":dict(zip(("lat","lon","v_long","v_lat"),roles["elcv"])),
                       "1":dict(zip(("lat","lon","v_long","v_lat"),roles["rfv"]))}
        self._lane_change_started=False;self._lane_change_completed=False
        self._lane_change_stable=0;self._lane_change_start_time=None;self._lane_change_completion_time=None
        self._rfv_initial_speed=float(np.linalg.norm(self.world.role_vehicle('rfv').velocity))
        self._fixed_rfv_speed=self._rfv_initial_speed
        self._rfv_min_accel=0.
        self._initial_gap=float(roles['rlv_target'][1]-roles['rfv'][1])
        if self._is_merge:
            ego=self.world.role_vehicle('elcv');rfv=self.world.role_vehicle('rfv')
            self._merge_initial_route=(ego.lane_index[:2],rfv.lane_index[:2])
            self._merge_start_x=float(ego.position[0])
            self._merge_end_x=self._merge_start_x+85.
            self._merge_complete_x=None
        self._last_info={"done":False,"time":0.,"distance_travelled_m":0.,"tracking_reward":0.,
                         "collision_proxy_any":False,"controlled_collision_proxy":False,
                         "metadrive_primary_collision":False,"metadrive_out_of_road":False,
                         "minimum_distance_m":None,"traffic_vehicle_count":max(0,len(self._vehicle_pool())-4)}
        if self.frame_callback:self.frame_callback(self,self._last_info)
        return {"elcv":obs.copy(),"rfv":obs.copy()},dict(scenario_id=self.scenario.scenario_id,
            dynamics_mode="metadrive_physics",scene_source="procedural",**world_info)

    def step(self,actions):
        if self._done:raise RuntimeError("call reset after the interaction episode ends")
        if set(actions)!={"elcv","rfv"}:raise ValueError("actions must contain exactly 'elcv' and 'rfv'")
        policy_actions={role:np.asarray(actions[role],dtype=float).copy() for role in ('elcv','rfv')}
        rfv_speed_before=float(np.linalg.norm(self.world.role_vehicle('rfv').velocity))
        ego=self.world.role_vehicle('elcv');target_lane=self._target_lane
        ego_source_longitudinal,_=self._source_lane.local_coordinates(ego.position[:2])
        ego_source_longitudinal-=self.world._interaction_s_offset
        if self.rfv_response=='fixed_speed':policy_actions['rfv']=np.asarray([0.,0.])
        elif self.rfv_response=='scripted_yield' and self._lane_change_started:policy_actions['rfv'][0]=-1.
        elif self.rfv_response=='scripted_gap_closing' and self._lane_change_started:policy_actions['rfv'][0]=.75
        controls={role:self._to_metadrive_action(self.world.role_vehicle(role),policy_actions[role]) for role in ("elcv","rfv")}
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
            if not self._lane_change_started and float(ego.position[0])>=self._merge_start_x+10.:
                self._lane_change_started=True;self._lane_change_start_time=self.time
            in_shared_target_lane=merge_node and ego.lane_index[2]==2
            self._lane_change_stable=self._lane_change_stable+1 if self._lane_change_started and in_shared_target_lane else 0
        else:
            lateral_from_source=abs(self._local_state(ego,self._source_lane)[0])
            if not self._lane_change_started and lateral_from_source>=.25 and self.interaction_scenario.interaction_start_x<=ego_source_longitudinal<=self.interaction_scenario.interaction_end_x:
                self._lane_change_started=True;self._lane_change_start_time=self.time
            _,ego_target_lateral=target_lane.local_coordinates(ego.position[:2])
            self._lane_change_stable=self._lane_change_stable+1 if self._lane_change_started and abs(float(ego_target_lateral))<=self.interaction_scenario.success_lateral_tolerance_m else 0
        if self._lane_change_stable>=self.interaction_scenario.success_stable_steps and not self._lane_change_completed:
            self._lane_change_completed=True;self._lane_change_completion_time=self.time
            if self._is_merge:self._merge_complete_x=float(ego.position[0])
        ego_zone_s=float(ego.position[0]) if self._is_merge else self._local_state(ego,self._source_lane)[1]-self.world._interaction_s_offset
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
                actor_state=self._local_state(actor,self._source_lane);other_state=self._local_state(other,self._source_lane)
                lateral_clearance=abs(other_state[0]-actor_state[0])
                if lateral_clearance>(actor.WIDTH+other.WIDTH)*.5+.75:continue
                gap=other_state[1]-actor_state[1];closing=actor_state[2]-other_state[2]
                if gap>0. and closing>1e-6:ttcs.append(max(0.,gap-(actor.LENGTH+other.LENGTH)*.5)/closing)
        zone_end=self._merge_end_x if self._is_merge else self.interaction_scenario.interaction_end_x
        zone_start=self._merge_start_x if self._is_merge else self.interaction_scenario.elcv_x
        zone_exited=ego_zone_s>=zone_end
        merge_lane_at_exit=bool(self.world.role_vehicle('elcv').lane_index[0] in ('1r0_0_','1r0_1_') and
                                self.world.role_vehicle('elcv').lane_index[2]==2)
        merge_success=bool(self._lane_change_completed and (not self._is_merge or merge_lane_at_exit) and
                            not collision and not out_of_road)
        done=self.steps>=self.max_steps or zone_exited or collision or out_of_road or bool(terminated.get("__all__",False))
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
        if self.world is not None:self.world.close()
