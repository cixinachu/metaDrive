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
                 seed=0, horizon=1000, view_mode="topdown", config=None):
        self.scenario_type=scenario_type
        self.scene_source="procedural"
        self.max_steps=int(horizon)
        states=np.asarray([
            [-.2,8.,0.,18.,0.],[-.1,8.,0.,18.,0.],
            [self.max_steps*.1,200.,0.,18.,0.],[(self.max_steps+1)*.1,202.,0.,18.,0.]
        ],dtype=float)
        tracks={str(i):{"id":str(i),"length":4.5,"width":1.8,"states":states.copy()} for i in range(4)}
        placeholder=SimpleNamespace(scenario_id=f"metadrive-{scenario_type}",elcv_id="0",rfv_id="1",
            rlv_current_id="2",rlv_target_id="3",tracks=tracks,start_time=0.,end_time=(self.max_steps+1)*.1)
        world=SRAIRLProceduralWorld(scenario_type,lane_num,traffic_density,view_mode,seed,horizon)
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
        manager=self.world.engine.traffic_manager
        vehicles.extend(v for v in getattr(manager,"traffic_vehicles",[]) if v not in vehicles)
        return vehicles

    def _roles_live(self):
        ego=self.world.role_vehicle("elcv")
        other=self.world.role_vehicle("rfv")
        lane=ego.lane
        ego_lane_index=tuple(ego.lane_index)
        ego_state=self._local_state(ego,lane)
        other_state=self._local_state(other,lane)
        lanes=self.world.current_map.road_network.graph.get(ego_lane_index[:2],[])
        lane_count=len(lanes)
        target_index=ego_lane_index[2]+1 if ego_lane_index[2]+1<lane_count else ego_lane_index[2]-1
        pool=self._vehicle_pool()
        leaders=[]
        for vehicle in pool:
            if vehicle is ego:continue
            index=tuple(vehicle.lane_index)
            if index[:2]!=ego_lane_index[:2]:continue
            state=self._local_state(vehicle,lane)
            if state[1]<=ego_state[1]+1.:continue
            if index[2]==ego_lane_index[2]:leaders.append((state[1]-ego_state[1],state))
        current=min(leaders,key=lambda item:item[0])[1] if leaders else ego_state+np.asarray([0.,120.,0.,0.])
        target_candidates=[]
        if 0<=target_index<lane_count:
            for vehicle in pool:
                if vehicle is ego:continue
                index=tuple(vehicle.lane_index)
                if index[:2]!=ego_lane_index[:2] or index[2]!=target_index:continue
                state=self._local_state(vehicle,lane)
                if state[1]>ego_state[1]+1.:target_candidates.append((state[1]-ego_state[1],state))
        target=min(target_candidates,key=lambda item:item[0])[1] if target_candidates else ego_state+np.asarray([0.,120.,0.,0.])
        return {"elcv":ego_state,"rfv":other_state,"rlv_current":current,"rlv_target":target}

    def reset(self,seed=None):
        self.time=0.;self.steps=0;self._done=False;self._metadrive_collision=False;self._metadrive_out_of_road=False
        # MetaDrive's explicit reset(seed=...) argument is a scenario index,
        # while the requested RNG seed is configured as start_seed at creation.
        _,world_info=self.world.reset()
        self.world.set_role_colors()
        roles=self._roles_live()
        obs=self.observer.reset(roles)
        self.distance_travelled_m=0.
        self._last_position=np.asarray(self.world.role_vehicle("elcv").position[:2],dtype=float)
        self.vehicles={"0":dict(zip(("lat","lon","v_long","v_lat"),roles["elcv"])),
                       "1":dict(zip(("lat","lon","v_long","v_lat"),roles["rfv"]))}
        self._last_info={"done":False,"time":0.,"distance_travelled_m":0.,"tracking_reward":0.,
                         "collision_proxy_any":False,"controlled_collision_proxy":False,
                         "metadrive_primary_collision":False,"metadrive_out_of_road":False,
                         "minimum_distance_m":None,"traffic_vehicle_count":len(self._vehicle_pool())-2}
        if self.frame_callback:self.frame_callback(self,self._last_info)
        return {"elcv":obs.copy(),"rfv":obs.copy()},dict(scenario_id=self.scenario.scenario_id,
            dynamics_mode="metadrive_physics",scene_source="procedural",**world_info)

    def step(self,actions):
        if self._done:raise RuntimeError("call reset after the interaction episode ends")
        if set(actions)!={"elcv","rfv"}:raise ValueError("actions must contain exactly 'elcv' and 'rfv'")
        controls={role:self._to_metadrive_action(self.world.role_vehicle(role),actions[role]) for role in ("elcv","rfv")}
        _,_,terminated,truncated,world_info=self.world.step(controls)
        self.steps+=1;self.time=self.steps*self.dt
        roles=self._roles_live();obs=self.observer.append(roles)
        position=np.asarray(self.world.role_vehicle("elcv").position[:2],dtype=float)
        self.distance_travelled_m+=float(np.linalg.norm(position-self._last_position));self._last_position=position
        tracking_reward=float(np.clip(roles["elcv"][2]/30.,-1.,1.))
        agents=[self.world.role_vehicle("elcv"),self.world.role_vehicle("rfv")]
        collision=any(bool(v.crash_vehicle or v.crash_object) for v in agents)
        out_of_road=any(bool(v.on_yellow_continuous_line or not v.on_lane) for v in agents)
        self._metadrive_collision|=collision;self._metadrive_out_of_road|=out_of_road
        pool=self._vehicle_pool();distances=[]
        for actor in agents:
            for other in pool:
                if other is actor:continue
                d=float(np.linalg.norm(np.asarray(actor.position[:2])-np.asarray(other.position[:2])))
                distances.append(d)
        done=self.steps>=self.max_steps or collision or out_of_road or bool(terminated.get("__all__",False))
        self._done=bool(done)
        info={"done":self._done,"time":self.time,"scenario_id":self.scenario.scenario_id,
              "tracking_reward":tracking_reward,"distance_travelled_m":self.distance_travelled_m,
              "collision_proxy_any":collision,"controlled_collision_proxy":bool(
                  np.linalg.norm(np.asarray(agents[0].position[:2])-np.asarray(agents[1].position[:2]))<
                  (agents[0].LENGTH+agents[1].LENGTH)*.5),
              "metadrive_primary_collision":collision,"metadrive_out_of_road":out_of_road,
              "minimum_distance_m":min(distances) if distances else None,
              "traffic_vehicle_count":max(0,len(pool)-2),"scene_source":"procedural",
              "scenario_type":self.scenario_type,"terminated":terminated,"truncated":truncated}
        self._last_info=info
        if self.frame_callback:self.frame_callback(self,info)
        return {"elcv":obs.copy(),"rfv":obs.copy()},info

    def close(self):
        if self.world is not None:self.world.close()
