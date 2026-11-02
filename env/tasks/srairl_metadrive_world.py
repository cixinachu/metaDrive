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
                "max_speed_km_h":130,
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
                 view_mode="topdown", seed=0, horizon=1000, traffic_seed=None,
                 interaction_scenario=None):
        from env.tasks.srairl_interaction import SRAIRLInteractionScenario
        from metadrive.component.pgblock.first_block import FirstPGBlock
        map_code={"straight":"S", "curve":"C", "merge":"r"}.get(scenario_type)
        if map_code is None:
            raise ValueError(f"SR-AIRL interaction map is not validated: {scenario_type}")
        if lane_num < 2:
            raise ValueError("SR-AIRL paired interaction requires at least two lanes")
        if scenario_type == "merge" and lane_num < 3:
            raise ValueError("MetaDrive ramp merge requires three mainline lanes so the ramp joins a distinct auxiliary lane")
        self.interaction_scenario=(interaction_scenario or SRAIRLInteractionScenario()).validate(lane_num)
        self.view_mode=view_mode
        self.scenario_type=scenario_type
        self.traffic_seed=int(seed if traffic_seed is None else traffic_seed)
        self.policy_roles={"elcv":"agent0", "rfv":"agent1"}
        lane_prefix=(FirstPGBlock.NODE_1,FirstPGBlock.NODE_2)
        if scenario_type == "merge":
            # On MetaDrive's real ramp map, the ramp enters an auxiliary lane
            # which then connects to the mainline. ELCV follows that route;
            # RFV is initialized on a mainline lane with the same downstream
            # lane in its navigation route.
            agent_configs={
                "agent0":dict(spawn_lane_index=("1r1_2_","1r1_3_",0),spawn_longitude=4.,
                           spawn_velocity=(20.,0.),spawn_velocity_car_frame=True,
                           max_speed_km_h=130,use_special_color=True,show_navi_mark=False,show_dest_mark=False),
                "agent1":dict(spawn_lane_index=(">>>","1r0_0_",2),spawn_longitude=25.,
                           spawn_velocity=(18.,0.),spawn_velocity_car_frame=True,
                           max_speed_km_h=130,use_special_color=True,show_navi_mark=False,show_dest_mark=False),
            }
        else:
            agent_configs={
            "agent0":dict(spawn_lane_index=(*lane_prefix,self.interaction_scenario.source_lane),spawn_longitude=self.interaction_scenario.elcv_x,
                           spawn_velocity=(self.interaction_scenario.elcv_speed,0.),spawn_velocity_car_frame=True,
                           max_speed_km_h=130,
                           use_special_color=True,show_navi_mark=False,show_dest_mark=False),
            "agent1":dict(spawn_lane_index=(*lane_prefix,self.interaction_scenario.target_lane),spawn_longitude=self.interaction_scenario.rfv_x,
                           spawn_velocity=(self.interaction_scenario.rfv_speed,0.),spawn_velocity_car_frame=True,
                           max_speed_km_h=130,
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
        agent_manager=self.engine.agent_manager
        reset_agents=agent_manager.reset
        def reset_with_interaction_spawn():
            self._apply_interaction_scenario_to_engine()
            return reset_agents()
        agent_manager.reset=reset_with_interaction_spawn

    def _post_process_config(self,config):
        dimensions={k:config["map_config"][k] for k in ("lane_num","lane_width","exit_length")}
        for k in dimensions:config["map_config"][k]=self.default_config_copy["map_config"][k]
        config=super()._post_process_config(config)
        config["map_config"].update(dimensions)
        return config

    def role_vehicle(self,role):
        if role in self.policy_roles:return self.agents[self.policy_roles[role]]
        return self.interaction_vehicles[role]

    def set_interaction_scenario(self,scenario):
        """Update the next procedural reset's four-role initial geometry."""
        if self.scenario_type=='merge':raise ValueError('NGSIM lane-change samples cannot initialize a ramp-merge map')
        self.interaction_scenario=scenario.validate(int(self.config['map_config']['lane_num']))
        from metadrive.component.pgblock.first_block import FirstPGBlock
        lane_prefix=(FirstPGBlock.NODE_1,FirstPGBlock.NODE_2)
        updates={'agent0':dict(spawn_lane_index=(*lane_prefix,scenario.source_lane),spawn_longitude=scenario.elcv_x,
                               spawn_velocity=(scenario.elcv_speed,0.),spawn_velocity_car_frame=True),
                 'agent1':dict(spawn_lane_index=(*lane_prefix,scenario.target_lane),spawn_longitude=scenario.rfv_x,
                               spawn_velocity=(scenario.rfv_speed,0.),spawn_velocity_car_frame=True)}
        for agent_id,values in updates.items():self.config['agent_configs'][agent_id].update(values)
        self._active_spawn_updates=updates
        self._apply_interaction_scenario_to_engine()

    def _apply_interaction_scenario_to_engine(self):
        updates=getattr(self,'_active_spawn_updates',None)
        if updates and self.engine is not None:
            for agent_id,values in updates.items():self.engine.global_config['agent_configs'][agent_id].update(values)

    def _reset_global_seed(self,force_seed=None):
        super()._reset_global_seed(force_seed)
        self._apply_interaction_scenario_to_engine()

    def spawn_interaction_roles(self):
        """Spawn kinematic source/target lane leaders for the explicit gap task."""
        from env.recorded_traffic import ReplayVehicle
        from metadrive.component.vehicle.vehicle_type import vehicle_class_to_type,vehicle_type
        vehicle_class_to_type[ReplayVehicle]='dataset_replay';vehicle_type['dataset_replay']=ReplayVehicle
        if self.scenario_type == "merge":
            return self._spawn_merge_roles(ReplayVehicle,vehicle_class_to_type,vehicle_type)
        a,b=self.role_vehicle('elcv').lane_index[:2]
        lanes=self.current_map.road_network.graph[a][b]
        spec=self.interaction_scenario;self.interaction_vehicles={}
        self._interaction_elapsed=0.
        ego_lane=lanes[spec.source_lane]
        ego_s,_=ego_lane.local_coordinates(self.role_vehicle('elcv').position[:2])
        self._interaction_s_offset=float(ego_s)-spec.elcv_x
        for role,lane_index,x,speed in (
            ('rlv_current',spec.source_lane,spec.rlv_current_x,spec.rlv_current_speed),
            ('rlv_target',spec.target_lane,spec.rlv_target_x,spec.rlv_target_speed)):
            lane=lanes[lane_index];position=lane.position(float(x+self._interaction_s_offset),0.)
            vehicle=self.engine.spawn_object(ReplayVehicle,vehicle_config=dict(length=4.5,width=1.8,height=1.5,
                spawn_position_heading=((float(position[0]),float(position[1])),float(lane.heading_theta_at(float(x+self._interaction_s_offset)))),
                navigation_module=None,enable_reverse=False,show_navi_mark=False,show_dest_mark=False))
            vehicle.body.setKinematic(True);heading=float(lane.heading_theta_at(float(x+self._interaction_s_offset)))
            vehicle.recorded_velocity=np.asarray([speed*np.cos(heading),speed*np.sin(heading)])
            vehicle.set_velocity(vehicle.recorded_velocity);self.interaction_vehicles[role]=vehicle
        return self.interaction_vehicles

    def _spawn_merge_roles(self,ReplayVehicle,vehicle_class_to_type,vehicle_type):
        """Place explicit ramp and mainline leaders on MetaDrive's connected ramp graph."""
        vehicle_class_to_type[ReplayVehicle]='dataset_replay';vehicle_type['dataset_replay']=ReplayVehicle
        graph=self.current_map.road_network.graph
        self.interaction_vehicles={};self._interaction_elapsed=0.;self._interaction_paths={}
        ramp=[graph['1r1_3_']['1r0_0_'][0],graph['1r0_0_']['1r0_1_'][3]]
        mainline=[graph['>>>']['1r0_0_'][2],graph['1r0_0_']['1r0_1_'][2],graph['1r0_1_']['1r0_2_'][2]]
        # The current-lane leader stays on the ramp/auxiliary branch. The
        # target-lane leader and RFV share the mainline lane reached by ELCV.
        placements={
            'rlv_current':(ramp,1.,18.),
            'rlv_target':(mainline,60.,20.),
        }
        for role,(path,s0,speed) in placements.items():
            lane,local=self._path_position(path,s0)
            position=lane.position(local,0.);heading=float(lane.heading_theta_at(local))
            vehicle=self.engine.spawn_object(ReplayVehicle,vehicle_config=dict(length=4.5,width=1.8,height=1.5,
                spawn_position_heading=((float(position[0]),float(position[1])),heading),navigation_module=None,
                enable_reverse=False,show_navi_mark=False,show_dest_mark=False))
            vehicle.body.setKinematic(True)
            vehicle.recorded_velocity=np.asarray([speed*np.cos(heading),speed*np.sin(heading)])
            vehicle.set_velocity(vehicle.recorded_velocity)
            self.interaction_vehicles[role]=vehicle
            self._interaction_paths[role]=(path,float(s0),float(speed))
        self._interaction_paths={role:value for role,value in self._interaction_paths.items()}
        # Validate the actual map graph and reset locations, not a drawn proxy.
        ego=self.role_vehicle('elcv');rfv=self.role_vehicle('rfv')
        if not ego.lane_index[0].startswith('1r1_') or rfv.lane_index[:2] != ('>>>','1r0_0_'):
            raise RuntimeError('MetaDrive ramp-merge actors did not initialize on their specified routes')
        if not graph['1r1_3_']['1r0_0_'] or not graph['>>>']['1r0_0_']:
            raise RuntimeError('MetaDrive ramp and mainline routes do not share the merge node')
        if ego.navigation.final_lane is not rfv.navigation.final_lane:
            raise RuntimeError('MetaDrive ramp and mainline routes do not share a downstream target lane')
        return self.interaction_vehicles

    @staticmethod
    def _path_position(path,distance):
        remaining=max(0.,float(distance))
        for lane in path:
            if remaining <= lane.length:return lane,float(remaining)
            remaining-=lane.length
        lane=path[-1]
        return lane,float(lane.length)

    def advance_interaction_roles(self,dt):
        spec=self.interaction_scenario
        self._interaction_elapsed+=float(dt)
        if self.scenario_type == "merge":
            for role,(path,s0,speed) in self._interaction_paths.items():
                lane,local=self._path_position(path,s0+speed*self._interaction_elapsed)
                position=lane.position(local,0.);heading=float(lane.heading_theta_at(local))
                vehicle=self.interaction_vehicles[role]
                vehicle.set_position((float(position[0]),float(position[1])))
                vehicle.set_heading_theta(heading)
                vehicle.recorded_velocity=np.asarray([speed*np.cos(heading),speed*np.sin(heading)])
                vehicle.set_velocity(vehicle.recorded_velocity);vehicle.after_step()
            return
        a,b=self.role_vehicle('elcv').lane_index[:2]
        lanes=self.current_map.road_network.graph[a][b]
        for role,lane_index,x0,speed in (
            ('rlv_current',spec.source_lane,spec.rlv_current_x,spec.rlv_current_speed),
            ('rlv_target',spec.target_lane,spec.rlv_target_x,spec.rlv_target_speed)):
            x=x0+speed*self._interaction_elapsed;lane=lanes[lane_index];position=lane.position(x+self._interaction_s_offset,0.)
            vehicle=self.interaction_vehicles[role]
            vehicle.set_position((float(position[0]),float(position[1])))
            heading=float(lane.heading_theta_at(x+self._interaction_s_offset));vehicle.set_heading_theta(heading)
            vehicle.recorded_velocity=np.asarray([speed*np.cos(heading),speed*np.sin(heading)])
            vehicle.set_velocity(vehicle.recorded_velocity);vehicle.after_step()

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
        for role,color in (('rlv_current',(.14,.68,.34)),('rlv_target',(.62,.30,.78))):
            vehicle=self.interaction_vehicles.get(role)
            if vehicle is not None and getattr(vehicle,'origin',None) is not None:
                vehicle.origin.setColor(*color,1.,10)

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
        result=super().step({self.policy_roles[role]:action for role,action in actions.items()})
        if getattr(self,'interaction_vehicles',None):
            dt=float(self.config['decision_repeat'])*float(self.config['physics_world_step_size'])
            self.advance_interaction_roles(dt)
        return result
