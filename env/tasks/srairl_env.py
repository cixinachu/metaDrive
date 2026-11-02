"""Two-actor SR-AIRL environment with upstream reference and MetaDrive physics modes.

The reference mode reproduces the upstream kinematic update. The physics mode
maps each actor's longitudinal/lateral acceleration action to MetaDrive vehicle
controls and reads the resulting physical states back into the joint policy
observation. The action mapping is an approximation and requires calibration
before results can be treated as an exact reproduction of upstream dynamics.
"""
import numpy as np
from env.adapters.srairl_observation import SRAIRLObservationAdapter
from env.adapters.srairl_action import SRAIRLActionAdapter
from rl.srairl.config import SRAIRLConfig


class SRAIRLEnv:
    dt = 0.1
    dynamics_modes = {"legacy_kinematic", "metadrive_physics"}

    def __init__(self, scenario, dynamics_mode="legacy_kinematic", history_len=None, config=None, metadrive_world=None):
        if dynamics_mode not in self.dynamics_modes:
            raise ValueError(f"unsupported dynamics_mode: {dynamics_mode}")
        if dynamics_mode == "metadrive_physics" and (metadrive_world is None or not getattr(metadrive_world,"is_multi_agent",False)):
            raise ValueError("metadrive_physics requires a two-agent MetaDrive world")
        for role in ("rlv_current_id", "rlv_target_id"):
            if getattr(scenario, role) is None:
                raise ValueError(f"scenario must explicitly assign {role}")
        self.scenario = scenario
        self.world = metadrive_world
        self.dynamics_mode=dynamics_mode
        self._meta_rfv = None
        self._metadrive_collision = False
        self._metadrive_out_of_road = False
        self.config = config or SRAIRLConfig.from_mapping()
        self.action_adapter = SRAIRLActionAdapter.from_yaml()
        history_len = self.config.history_len if history_len is None else history_len
        self.observer = SRAIRLObservationAdapter(history_len,
            pos_lat_scale=self.config.pos_lat_scale, pos_long_scale=self.config.pos_long_scale,
            vel_scale=self.config.vel_scale, vel_offset=self.config.vel_offset)
        self.time = 0.0
        self.frame_callback = None
        self.vehicles = {}
        self.steps = 0
        self._done = False
        self._metadrive_collision = False
        self._metadrive_out_of_road = False
        role_ids = (scenario.elcv_id, scenario.rfv_id, scenario.rlv_current_id, scenario.rlv_target_id)
        self.start_time = max(float(scenario.start_time), *(float(scenario.tracks[k]["states"][0][0]) for k in role_ids))
        self.end_time = min(float(scenario.tracks[k]["states"][-1][0]) for k in role_ids)
        if scenario.end_time is not None:self.end_time=min(self.end_time,float(scenario.end_time))
        self.episode_end_time = self.start_time + np.floor((self.end_time - self.start_time) / self.dt + 1e-8) * self.dt
        if self.episode_end_time - self.start_time < self.dt:
            raise ValueError("the four role tracks need at least one common 0.1-second transition")

    @staticmethod
    def _sample(track, t):
        from datasets.srairl.kinematics import state
        return state(track,t,dt=.1,require_preroll=True)

    def _roles(self, t):
        mapping = {"elcv": self.scenario.elcv_id, "rfv": self.scenario.rfv_id,
                   "rlv_current": self.scenario.rlv_current_id, "rlv_target": self.scenario.rlv_target_id}
        roles = {}
        for role, ident in mapping.items():
            if role in ("elcv", "rfv") and ident in self.vehicles:
                v = self.vehicles[ident]
                roles[role] = np.asarray([v["lat"], v["lon"], v["v_long"], v["v_lat"]])
                continue
            raw = self._sample(self.scenario.tracks[ident], t)
            if raw is None:
                raise RuntimeError(f"role {role} ({ident}) has no recorded state at t={t:.3f}")
            x, y, vx, vy = raw
            roles[role] = np.asarray([y, x, vx, vy], dtype=np.float64)
        return roles

    def reset(self):
        self.time = self.start_time
        self.steps = 0
        self._done = False
        self.vehicles = {}
        for ident in (self.scenario.elcv_id, self.scenario.rfv_id):
            raw = self._sample(self.scenario.tracks[ident], self.time)
            if raw is None:
                raise ValueError(f"controlled track {ident} lacks an initial state")
            x, y, vx, vy = raw
            self.vehicles[ident] = {"lat": float(y), "lon": float(x), "v_long": float(vx), "v_lat": float(vy), "a_long": 0., "a_lat": 0.}
        obs = self.observer.reset(self._roles(self.time))
        if self.world is not None:
            traffic_manager=getattr(getattr(self.world,"engine",None),"traffic_manager",None)
            if traffic_manager is not None and hasattr(traffic_manager,"set_controlled_vehicles"):
                traffic_manager.set_controlled_vehicles([])
            self.world.reset()
            if self.dynamics_mode=="metadrive_physics":
                self.world.place_actors_at(self.time)
                self._meta_rfv=self.world.role_vehicle("rfv")
                self.world.set_role_colors()
                for role,ident in (("elcv",self.scenario.elcv_id),("rfv",self.scenario.rfv_id)):
                    self.vehicles[ident]=self._read_world_state(self.world.role_vehicle(role))
            else:
                from env.recorded_traffic import ReplayVehicle
                self._sync_world_vehicle(self.world.agent,self.vehicles[self.scenario.elcv_id])
                track=self.scenario.tracks[self.scenario.rfv_id]
                x,y,vx,vy=self._sample(track,self.time)
                heading=float(np.arctan2(vy,vx)) if np.hypot(vx,vy)>.1 else 0.
                self._meta_rfv=self.world.engine.spawn_object(ReplayVehicle,vehicle_config=dict(
                    length=track.get("length",4.5),width=track.get("width",1.8),height=1.5,
                    spawn_position_heading=((float(x),float(y)),heading),navigation_module=None,
                    enable_reverse=True,show_navi_mark=False,show_dest_mark=False))
                self._meta_rfv.body.setKinematic(True)
                self._sync_world_vehicle(self._meta_rfv,self.vehicles[self.scenario.rfv_id])
                self._meta_rfv.after_step()
            traffic_manager=self.world.engine.traffic_manager
            if hasattr(traffic_manager,'set_controlled_vehicles'):
                traffic_manager.set_controlled_vehicles([self.world.role_vehicle("elcv"),self.world.role_vehicle("rfv")]
                    if self.dynamics_mode=="metadrive_physics" else [self.world.agent,self._meta_rfv])
            if self.dynamics_mode!="metadrive_physics":
                self._set_world_color(self.world.agent,(.12,.42,.95))
                self._set_world_color(self._meta_rfv,(.98,.50,.10))
            else:
                obs=self.observer.reset(self._roles(self.time))
            replay=self.world.engine.traffic_manager
            for ident,vehicle in getattr(replay,"active",{}).items():
                if str(ident)==str(self.scenario.rlv_current_id):color=(.14,.68,.34)
                elif str(ident)==str(self.scenario.rlv_target_id):color=(.62,.30,.78)
                else:color=(.55,.57,.60)
                self._set_world_color(vehicle,color)
            self.world.engine.taskMgr.step()
        return {"elcv": obs.copy(), "rfv": obs.copy()}, {"scenario_id": self.scenario.scenario_id, "time": self.time, "dynamics_mode": self.dynamics_mode}

    @staticmethod
    def _read_world_state(vehicle):
        pos=np.asarray(vehicle.position,dtype=float);vel=np.asarray(vehicle.velocity,dtype=float)
        return {"lat":float(pos[1]),"lon":float(pos[0]),"v_long":float(vel[0]),"v_lat":float(vel[1]),"a_long":0.,"a_lat":0.}

    def _to_metadrive_action(self,vehicle,acceleration):
        return self.action_adapter.to_metadrive(vehicle,acceleration,self.dt)

    @staticmethod
    def _sync_world_vehicle(vehicle,state):
        heading=float(np.arctan2(state["v_lat"],state["v_long"])) if np.hypot(state["v_long"],state["v_lat"])>.1 else 0.
        vehicle.set_position((state["lon"],state["lat"]))
        vehicle.set_heading_theta(heading)
        vehicle.set_velocity((state["v_long"],state["v_lat"]))
        vehicle.recorded_velocity=np.asarray([state["v_long"],state["v_lat"]],dtype=np.float64)

    @staticmethod
    def _set_world_color(vehicle,color):
        if getattr(vehicle,"origin",None) is not None:
            vehicle.origin.setColor(float(color[0]),float(color[1]),float(color[2]),1.,10)

    def step(self, actions):
        if self._done:
            raise RuntimeError("call reset after the interaction episode ends")
        if set(actions) != {"elcv", "rfv"}:
            raise ValueError("actions must contain exactly 'elcv' and 'rfv'")
        normalized_actions={}
        for role, ident in (("elcv", self.scenario.elcv_id), ("rfv", self.scenario.rfv_id)):
            action = np.asarray(actions[role], dtype=np.float64)
            if action.shape != (2,) or not np.isfinite(action).all():
                raise ValueError(f"{role} action must be finite [a_long, a_lat]")
            normalized_actions[role]=action
            if self.dynamics_mode=="legacy_kinematic":
                state = self.vehicles[ident]
                state["v_long"] += float(action[0]) * self.dt
                state["v_lat"] += float(action[1]) * self.dt
                state["lat"] += state["v_lat"] * self.dt
                state["lon"] += state["v_long"] * self.dt
                state["a_long"], state["a_lat"] = map(float, action)
        self.steps += 1
        self.time = float(self.start_time + self.steps * self.dt)
        if self.world is not None:
            traffic_manager=self.world.engine.traffic_manager
            if hasattr(traffic_manager,'set_controlled_vehicles'):
                traffic_manager.set_controlled_vehicles([self.world.role_vehicle("elcv"),self.world.role_vehicle("rfv")]
                    if self.dynamics_mode=="metadrive_physics" else [self.world.agent,self._meta_rfv])
            if self.dynamics_mode=="metadrive_physics":
                md_actions={role:self._to_metadrive_action(self.world.role_vehicle(role),normalized_actions[role])
                            for role in ("elcv","rfv")}
                _,_,terminated,truncated,world_info=self.world.step(md_actions)
                for role,ident in (("elcv",self.scenario.elcv_id),("rfv",self.scenario.rfv_id)):
                    vehicle=self.world.role_vehicle(role)
                    self.vehicles[ident]=self._read_world_state(vehicle)
                    self.vehicles[ident]["a_long"],self.vehicles[ident]["a_lat"]=map(float,normalized_actions[role])
                    row=world_info.get(self.world.role_agent_ids[role],{}) if isinstance(world_info,dict) else {}
                    self._metadrive_collision=self._metadrive_collision or bool(
                        row.get('crash_vehicle') or row.get('crash_object') or vehicle.crash_vehicle or vehicle.crash_object)
                    self._metadrive_out_of_road=self._metadrive_out_of_road or bool(row.get('out_of_road',False))
            else:
                # Legacy reference mode advances the MetaDrive scene and then
                # synchronizes the kinematic actors for visualization only.
                for _ in range(int(self.world.config['decision_repeat'])):
                    traffic_manager.step();self.world.engine.step_physics_world()
                    self.world.engine.episode_step += 1
                    for name,manager in self.world.engine.managers.items():
                        if name not in ('agent_manager','record_manager'):manager.after_step()
                    self.world.agent.after_step();self._meta_rfv.after_step();self.world.engine.taskMgr.step()
                self._metadrive_collision=self._metadrive_collision or bool(
                    getattr(self.world.agent,'crash_vehicle',False) or getattr(self.world.agent,'crash_object',False) or
                    getattr(self._meta_rfv,'crash_vehicle',False) or getattr(self._meta_rfv,'crash_object',False))
                self._sync_world_vehicle(self.world.agent,self.vehicles[self.scenario.elcv_id])
                self._sync_world_vehicle(self._meta_rfv,self.vehicles[self.scenario.rfv_id])
                self.world.engine.taskMgr.step()
        roles = self._roles(self.time)
        obs = self.observer.append(roles)
        done = self.time >= self.episode_end_time - 1e-6
        self._done = bool(done)
        # Match upstream NGSIMEnv's negative normalized position-RMSE reward.
        errors = []
        actor_rewards={}
        position_errors = {}
        for ident in (self.scenario.elcv_id, self.scenario.rfv_id):
            recorded = self._sample(self.scenario.tracks[ident], self.time)
            current = self.vehicles[ident]
            label = np.asarray([recorded[0], recorded[1]]) / np.asarray([self.config.pos_long_scale, self.config.pos_lat_scale])
            pred = np.asarray([current["lon"], current["lat"]]) / np.asarray([self.config.pos_long_scale, self.config.pos_lat_scale])
            error = float(np.sqrt(np.mean((pred - label) ** 2)))
            errors.append(error)
            actor_rewards[ident]=-error
            position_errors[ident] = float(np.linalg.norm(
                np.asarray([current["lon"], current["lat"]]) - np.asarray([recorded[0], recorded[1]])))
        controlled = [self.vehicles[k] for k in (self.scenario.elcv_id, self.scenario.rfv_id)]
        separation = float(np.hypot(controlled[0]["lon"]-controlled[1]["lon"],
                                    controlled[0]["lat"]-controlled[1]["lat"]))
        dx=controlled[1]["lon"]-controlled[0]["lon"]
        dv=controlled[1]["v_long"]-controlled[0]["v_long"]
        ttc=float(-dx/dv) if dx*dv<0 else None
        tracking_reward = -sum(errors)
        lengths=[float(self.scenario.tracks[k].get("length",4.5)) for k in (self.scenario.elcv_id,self.scenario.rfv_id)]
        widths=[float(self.scenario.tracks[k].get("width",1.8)) for k in (self.scenario.elcv_id,self.scenario.rfv_id)]
        longitudinal_clearance=max(abs(controlled[0]["lon"]-controlled[1]["lon"])-sum(lengths)/2,0.)
        lateral_clearance=max(abs(controlled[0]["lat"]-controlled[1]["lat"])-sum(widths)/2,0.)
        clearance=float(np.hypot(longitudinal_clearance,lateral_clearance))
        collision_proxy=(abs(controlled[0]["lon"]-controlled[1]["lon"]) < sum(lengths)/2 and
                         abs(controlled[0]["lat"]-controlled[1]["lat"]) < sum(widths)/2)
        actors=[]
        for ident in (self.scenario.elcv_id,self.scenario.rfv_id):
            state=self.vehicles[ident];track=self.scenario.tracks[ident]
            actors.append((ident,state,float(track.get('length',4.5)),float(track.get('width',1.8))))
        for ident,track in self.scenario.tracks.items():
            if ident in (self.scenario.elcv_id,self.scenario.rfv_id):continue
            raw=self._sample(track,self.time)
            if raw is None:continue
            x,y,vx,vy=raw
            if self.world is not None:
                replay_vehicle=getattr(self.world.engine.traffic_manager,'active',{}).get(ident)
                if replay_vehicle is not None:
                    pos=np.asarray(replay_vehicle.position,dtype=float);vel=np.asarray(replay_vehicle.velocity,dtype=float)
                    x,y=float(pos[0]),float(pos[1]);vx,vy=float(vel[0]),float(vel[1])
            actors.append((ident,{'lon':float(x),'lat':float(y),'v_long':float(vx),'v_lat':float(vy)},
                           float(track.get('length',4.5)),float(track.get('width',1.8))))
        interaction_clearance=float('inf');interaction_ttc=float('inf');any_collision_proxy=False
        for i,(id_a,a,len_a,wid_a) in enumerate(actors):
            if id_a not in (self.scenario.elcv_id,self.scenario.rfv_id):continue
            for id_b,b,len_b,wid_b in actors[i+1:]:
                dx=b['lon']-a['lon'];dy=b['lat']-a['lat']
                gap_x=max(abs(dx)-(len_a+len_b)/2,0.);gap_y=max(abs(dy)-(wid_a+wid_b)/2,0.)
                interaction_clearance=min(interaction_clearance,float(np.hypot(gap_x,gap_y)))
                any_collision_proxy |= abs(dx)<(len_a+len_b)/2 and abs(dy)<(wid_a+wid_b)/2
                dv=b['v_long']-a['v_long']
                if abs(dy)<=(wid_a+wid_b)/2+1.0 and dx*dv<0:
                    interaction_ttc=min(interaction_ttc,float(-dx/dv))
        info = {"done": self._done, "time": self.time, "scenario_id": self.scenario.scenario_id,
                "dynamics_mode": self.dynamics_mode, "tracking_reward": tracking_reward,
                "tracking_reward_elcv":actor_rewards[self.scenario.elcv_id],
                "tracking_reward_rfv":actor_rewards[self.scenario.rfv_id],
                "route_completion":float(np.clip((self.time-self.start_time)/max(self.episode_end_time-self.start_time,1e-8),0.,1.)),
                "position_errors_m": position_errors, "controlled_separation_m": separation,
                "controlled_min_clearance_m": clearance,"controlled_ttc_s": ttc,
                "controlled_collision_proxy": bool(collision_proxy),
                "min_interaction_clearance_m":None if not np.isfinite(interaction_clearance) else interaction_clearance,
                "min_interaction_ttc_s":None if not np.isfinite(interaction_ttc) else interaction_ttc,
                "collision_proxy_any":bool(any_collision_proxy),
                "metadrive_primary_collision":bool(self._metadrive_collision),
                "metadrive_out_of_road":bool(self._metadrive_out_of_road)}
        if self.frame_callback is not None:
            self.frame_callback(self, info)
        return {"elcv": obs.copy(), "rfv": obs.copy()}, info

    def close(self):
        if self.world is not None:
            self.world.close()
            self.world=None
