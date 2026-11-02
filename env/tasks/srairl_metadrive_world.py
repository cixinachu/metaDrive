"""Two-agent MetaDrive world for SR-AIRL physical closed-loop experiments."""
import math

import numpy as np
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.obs.observation_base import DummyObservation
from metadrive.component.pgblock.first_block import FirstPGBlock


class SRAIRLMetaDriveWorld(MultiAgentMetaDrive):
    """MetaDrive controls both SR-AIRL actors with its vehicle dynamics."""

    def __init__(self, scenario, clip, view_mode="topdown", reactive_background=True):
        self.interaction_scenario=scenario
        self.dataset_clip=clip
        self.reactive_background=bool(reactive_background)
        role_tracks={"elcv":scenario.elcv_id,"rfv":scenario.rfv_id}
        agent_configs={}
        for role,ident in role_tracks.items():
            track=scenario.tracks[ident]
            state=np.asarray(track["states"],dtype=float)
            t=max(float(scenario.start_time),float(state[0,0]))
            raw=np.asarray([np.interp(t,state[:,0],state[:,i]) for i in range(1,5)])
            x,y,vx,vy=raw
            heading=math.atan2(vy,vx) if math.hypot(vx,vy)>.1 else 0.
            agent_id="agent0" if role=="elcv" else "agent1"
            agent_configs[agent_id]={
                "vehicle_model":"varying_dynamics_bounding_box",
                "length":float(track.get("length",4.5)),"width":float(track.get("width",1.8)),"height":1.5,
                "spawn_longitude":30. if role=="elcv" else 15.,
                "spawn_lane_index":(FirstPGBlock.NODE_1,FirstPGBlock.NODE_2,0 if role=="elcv" else min(1,scenario.lane_num-1)),
                "enable_reverse":False,"show_navi_mark":False,"show_dest_mark":False,
                "random_color":False,"use_special_color":True,
            }
        duration=min(float(scenario.tracks[k]["states"][-1][0]) for k in
                     (scenario.elcv_id,scenario.rfv_id,scenario.rlv_current_id,scenario.rlv_target_id))
        if scenario.end_time is not None:duration=min(duration,float(scenario.end_time))
        road_clip={**clip,"duration":duration}
        render_3d=view_mode!="topdown"
        config={
            "is_multi_agent":True,"num_agents":2,"agent_configs":agent_configs,
            "random_agent_model":False,"allow_respawn":False,"delay_done":0,
            "traffic_density":0.,"random_traffic":False,"map":"",
            "map_config":{"lane_num":int(scenario.lane_num),"lane_width":float(scenario.lane_width),
                          "exit_length":max(float(scenario.road_length),80.)},
            "horizon":max(10,int(math.ceil((duration-scenario.start_time)/.1))+2),
            "physics_world_step_size":.02,"decision_repeat":5,
            "use_render":False,"image_observation":render_3d,"show_interface":False,
            "agent_observation":DummyObservation,
            "crash_vehicle_done":False,"crash_object_done":False,"out_of_road_done":False,
            "log_level":50,"force_destroy":True,
        }
        if render_3d:
            config.update(window_size=(640,480),show_logo=False,show_fps=False,
                          sensors={"main_camera":("MainCamera",640,480)})
        self._controlled_ids={str(scenario.elcv_id),str(scenario.rfv_id)}
        self._recorded_clip=road_clip
        self._view_mode=view_mode
        self.view_mode=view_mode
        self.role_agent_ids={"elcv":"agent0","rfv":"agent1"}
        super().__init__(config)

    def _post_process_config(self,config):
        dimensions={k:config["map_config"][k] for k in ("lane_num","lane_width","exit_length")}
        for k in dimensions:config["map_config"][k]=self.default_config_copy["map_config"][k]
        config=super()._post_process_config(config)
        config["map_config"].update(dimensions)
        return config

    def setup_engine(self):
        super().setup_engine()
        from env.recorded_map import RecordedMapManager
        from env.recorded_traffic import InteractionTrafficManager
        self.engine.update_manager("map_manager",RecordedMapManager())
        self.engine.update_manager("traffic_manager",InteractionTrafficManager(
            self._recorded_clip,self._controlled_ids,reactive=self.reactive_background))

    def step(self,actions):
        manager=self.engine.traffic_manager
        manager.set_controlled_vehicles([self.agents[self.role_agent_ids["elcv"]],self.agents[self.role_agent_ids["rfv"]]])
        return super().step({self.role_agent_ids[role]:action for role,action in actions.items()})

    def role_vehicle(self,role):
        return self.agents[self.role_agent_ids[role]]

    def place_actors_at(self,t):
        for role,ident in (("elcv",self.interaction_scenario.elcv_id),("rfv",self.interaction_scenario.rfv_id)):
            from datasets.srairl.kinematics import state
            raw=state(self.interaction_scenario.tracks[ident],t,dt=.1,require_preroll=True)
            x,y,vx,vy=raw
            vehicle=self.role_vehicle(role)
            vehicle.set_position((float(x),float(y)))
            vehicle.set_heading_theta(math.atan2(vy,vx) if math.hypot(vx,vy)>.1 else 0.)
            vehicle.set_velocity((float(vx),float(vy)))
            vehicle.after_step()
        self.engine.taskMgr.step()

    def set_role_colors(self):
        colors={"elcv":(.12,.42,.95),"rfv":(.98,.50,.10)}
        for role,color in colors.items():
            vehicle=self.agents.get(self.role_agent_ids[role])
            if vehicle is not None and getattr(vehicle,"origin",None) is not None:
                vehicle.origin.setColor(*color,1.,10)
        manager=self.engine.traffic_manager
        for ident,vehicle in manager.active.items():
            color=(.14,.68,.34) if str(ident)==str(self.interaction_scenario.rlv_current_id) else \
                  (.62,.30,.78) if str(ident)==str(self.interaction_scenario.rlv_target_id) else (.55,.57,.60)
            if getattr(vehicle,"origin",None) is not None:vehicle.origin.setColor(*color,1.,10)

    def camera_frame(self,overview=False):
        camera=self.engine.main_camera
        if self.view_mode=="bird3d" or overview:
            from panda3d.core import NodePath
            x0,x1,y0,y1=self.current_map.road_network.get_bounding_box()
            cx,cy=(x0+x1)/2,(y0+y1)/2;extent=max(x1-x0,y1-y0,40.)
            pose=NodePath("srairl-overview-pose");pose.setPos(cx,-cy-extent*.65,extent*.85);pose.lookAt(cx,-cy,0)
            pixels=camera.perceive(to_float=False,new_parent_node=self.engine.render,
                                   position=tuple(pose.getPos()),hpr=tuple(pose.getHpr()))
        elif self.view_mode=="first3d":
            agent=self.role_vehicle("elcv")
            pixels=camera.perceive(to_float=False,new_parent_node=agent.origin,
                                   position=(0.,.8,1.5),hpr=(0.,0.,0.))
        else:
            self.engine.taskMgr.step()
            pixels=camera.perceive(to_float=False)
        return pixels[...,::-1]


class SRAIRLProceduralWorld(MultiAgentMetaDrive):
    """Two SR-AIRL policy vehicles plus MetaDrive traffic on a generated map.

    ELCV and RFV are the two upstream policy roles. Additional vehicles are
    simulator traffic, not additional SR-AIRL policies.
    """

    def __init__(self, scenario_type="straight", lane_num=3, traffic_density=.15,
                 view_mode="topdown", seed=0, horizon=1000, traffic_seed=None):
        from metadrive.component.pgblock.first_block import FirstPGBlock
        map_code={"straight":"S", "curve":"C", "merge":"r"}.get(scenario_type)
        if map_code is None:
            raise ValueError(f"unsupported SR-AIRL procedural map: {scenario_type}")
        if lane_num < 2:
            raise ValueError("SR-AIRL paired interaction requires at least two lanes")
        self.view_mode=view_mode
        self.scenario_type=scenario_type
        self.traffic_seed=int(seed if traffic_seed is None else traffic_seed)
        self.policy_roles={"elcv":"agent0", "rfv":"agent1"}
        lane_prefix=(FirstPGBlock.NODE_1,FirstPGBlock.NODE_2)
        agent_configs={
            "agent0":dict(spawn_lane_index=(*lane_prefix,0),spawn_longitude=30.,
                           use_special_color=True,show_navi_mark=False,show_dest_mark=False),
            "agent1":dict(spawn_lane_index=(*lane_prefix,1),spawn_longitude=15.,
                           use_special_color=True,show_navi_mark=False,show_dest_mark=False),
        }
        render_3d=view_mode!="topdown"
        config={
            "is_multi_agent":True,"num_agents":2,"agent_configs":agent_configs,
            "random_agent_model":False,"allow_respawn":False,"delay_done":0,
            "crash_done":False,"out_of_road_done":False,
            "traffic_density":float(traffic_density),"random_traffic":False,
            "map":map_code,"map_config":{"lane_num":int(lane_num),"lane_width":3.5,"exit_length":500.},
            # The wrapper terminates its own rollout one step before MetaDrive
            # truncation so both controlled agents remain available for the
            # final observation and collision accounting.
            "start_seed":int(seed),"num_scenarios":100000,"horizon":int(horizon)+1,
            "physics_world_step_size":.02,"decision_repeat":5,
            "use_render":False,"image_observation":render_3d,"show_interface":False,
            "crash_vehicle_done":False,"crash_object_done":False,"out_of_road_done":False,
            "log_level":50,"force_destroy":True,
        }
        if render_3d:
            config.update(window_size=(640,480),show_logo=False,show_fps=False,
                          sensors={"main_camera":("MainCamera",640,480)})
        super().__init__(config)

    def setup_engine(self):
        super().setup_engine()
        from metadrive.manager.traffic_manager import PGTrafficManager

        traffic_seed=self.traffic_seed
        class SeededPGTrafficManager(PGTrafficManager):
            def seed(self, random_seed):
                return super().seed((int(random_seed)+traffic_seed) % (2**32))

        self.engine.update_manager("traffic_manager",SeededPGTrafficManager())

    def _post_process_config(self,config):
        dimensions={k:config["map_config"][k] for k in ("lane_num","lane_width","exit_length")}
        for k in dimensions:config["map_config"][k]=self.default_config_copy["map_config"][k]
        config=super()._post_process_config(config)
        config["map_config"].update(dimensions)
        return config

    def role_vehicle(self,role):
        return self.agents[self.policy_roles[role]]

    def done_function(self,vehicle_id):
        # Keep both policy actors present for the full coordinated episode.
        # The wrapper records collision/off-road events and ends the rollout
        # after the MetaDrive step, so one actor cannot disappear mid-pair.
        _,info=super().done_function(vehicle_id)
        return False,info

    def set_role_colors(self):
        for role,color in (("elcv",(.12,.42,.95)),("rfv",(.98,.50,.10))):
            vehicle=self.role_vehicle(role)
            if getattr(vehicle,"origin",None) is not None:vehicle.origin.setColor(*color,1.,10)

    def camera_frame(self,overview=False):
        camera=self.engine.main_camera
        if self.view_mode=="bird3d" or overview:
            from panda3d.core import NodePath
            x0,x1,y0,y1=self.current_map.road_network.get_bounding_box()
            cx,cy=(x0+x1)/2,(y0+y1)/2;extent=max(x1-x0,y1-y0,40.)
            pose=NodePath("srairl-procedural-overview");pose.setPos(cx,-cy-extent*.65,extent*.85);pose.lookAt(cx,-cy,0)
            pixels=camera.perceive(to_float=False,new_parent_node=self.engine.render,
                                   position=tuple(pose.getPos()),hpr=tuple(pose.getHpr()))
        elif self.view_mode=="first3d":
            vehicle=self.role_vehicle("elcv")
            pixels=camera.perceive(to_float=False,new_parent_node=vehicle.origin,
                                   position=(0.,.8,1.5),hpr=(0.,0.,0.))
        else:
            self.engine.taskMgr.step();pixels=camera.perceive(to_float=False)
        return pixels[...,::-1]

    def step(self,actions):
        return super().step({self.policy_roles[role]:action for role,action in actions.items()})
