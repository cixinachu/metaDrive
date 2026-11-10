"""Two-agent MetaDrive world for SR-AIRL physical closed-loop experiments."""
import math

import numpy as np
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.obs.observation_base import DummyObservation
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.ramp import InRampOnStraight
from metadrive.component.pg_space import BlockParameterSpace, Parameter, ParameterSpace, ConstantSpace
from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation


class SRAIRLSpawnNavigation(NodeNetworkNavigation):
    """Recover a Bullet localization miss only on a verified configured lane."""

    def reset(self, vehicle, dest=None, random_seed=None):
        from metadrive.utils.pg.utils import ray_localization
        from metadrive.component.navigation_module.base_navigation import BaseNavigation

        if ray_localization(vehicle.heading, vehicle.spawn_place, self.engine, use_heading_filter=False):
            return super().reset(vehicle, dest, random_seed)
        index = vehicle.config['spawn_lane_index']
        if index is None:
            return super().reset(vehicle, dest, random_seed)
        lane = self.map.road_network.get_lane(index)
        s, lateral = lane.local_coordinates(vehicle.spawn_place)
        center = np.asarray(lane.position(s, lateral), dtype=float)[:2]
        position = np.asarray(vehicle.spawn_place, dtype=float)[:2]
        if not (0. <= s <= lane.length and abs(lateral) <= lane.width_at(s) / 2.
                and np.linalg.norm(center - position) < .05):
            raise ValueError(f'Background traffic spawn is outside configured lane {index}: {position}')
        BaseNavigation.reset(self, lane)
        destination = vehicle.config['destination'] if dest is None else dest
        seed = self.engine.global_random_seed if random_seed is None else random_seed
        destination = self.auto_assign_task(self.map, lane.index, destination, seed)
        self.set_route(lane.index, destination)


from env.tasks.srairl_interaction import ROAD_GEOMETRY_CONFIG

CURVE_CENTER_RADIUS_M = ROAD_GEOMETRY_CONFIG["curve_center_radius_m"]
RAMP_CENTER_RADIUS_M = ROAD_GEOMETRY_CONFIG["ramp_center_radius_m"]


def activate_occupied_traffic_blocks(manager):
    """Trigger native traffic when actors spawn inside a block past its entry road."""
    occupied={tuple(v.lane_index[:2]) for v in manager.engine.agent_manager.active_agents.values()
              if v.lane_index is not None}
    for pending in list(manager.block_triggered_vehicles):
        for block in manager.current_map.blocks[1:]:
            if block.pre_block_socket.positive_road!=pending.trigger_road:continue
            graph=block.block_network.graph
            if any(end in graph.get(start,{}) for start,end in occupied):
                existing={id(vehicle) for vehicle in manager._traffic_vehicles}
                manager._traffic_vehicles.extend(vehicle for vehicle in manager.get_objects(pending.vehicles).values()
                                                 if id(vehicle) not in existing)
                manager.block_triggered_vehicles.remove(pending)
            break


class SRAIRLHighwayCurve(Curve):
    """MetaDrive Curve block with a highway-scale, fixed design radius."""
    PARAMETER_SPACE = ParameterSpace({
        **BlockParameterSpace.CURVE,
        Parameter.radius: ConstantSpace(CURVE_CENTER_RADIUS_M),
    })


class SRAIRLHighwayRamp(InRampOnStraight):
    """MetaDrive merge block with a long, gradual native ramp transition."""
    RADIUS = RAMP_CENTER_RADIUS_M
    PARAMETER_SPACE = ParameterSpace({
        **BlockParameterSpace.RAMP_PARAMETER,
        Parameter.length: ConstantSpace(ROAD_GEOMETRY_CONFIG['ramp_acceleration_length_m']),
        Parameter.extension_length: ConstantSpace(ROAD_GEOMETRY_CONFIG['ramp_extension_length_m']),
    })


class SRAIRLBlockDistConfig(PGBlockDistConfig):
    @classmethod
    def get_block(cls, block_id, version="v2"):
        if block_id == SRAIRLHighwayCurve.ID:
            return SRAIRLHighwayCurve
        if block_id == SRAIRLHighwayRamp.ID:
            return SRAIRLHighwayRamp
        return super().get_block(block_id, version)


class _LaneSlice:
    """A lane interval preserving MetaDrive's lane coordinate interface."""
    def __init__(self, lane, start_s, end_s):
        self.lane=lane;self.start_s=float(start_s);self.length=max(0.,float(end_s)-self.start_s)
        self.index=lane.index;self.width=getattr(lane,"width",3.5)

    def width_at(self,s):
        return self.lane.width_at(self.start_s+float(s))

    def position(self,s,lateral=0.):
        return self.lane.position(self.start_s+float(s),lateral)

    def heading_theta_at(self,s):
        return self.lane.heading_theta_at(self.start_s+float(s))

    def local_coordinates(self,position):
        s,lateral=self.lane.local_coordinates(position)
        return float(s)-self.start_s,float(lateral)


class _LaneTransition:
    """Smooth centerline through MetaDrive's native auxiliary-lane taper."""
    def __init__(self, source, source_s, target, target_s, length=24., samples=501):
        self.source=source;self.source_s=float(source_s);self.target=target
        self.target_s=float(target_s);self.width=3.5
        self.index=("merge-transition","merge-transition",2)
        p0=np.asarray(source.position(self.source_s,0.),dtype=float)
        p1=np.asarray(target.position(self.target_s,0.),dtype=float)
        h0=float(source.heading_theta_at(self.source_s));h1=float(target.heading_theta_at(self.target_s))
        self.length=float(target_s)-float(source_s)
        chord=max(float(np.linalg.norm(p1-p0)),self.length)
        t=np.linspace(0.,1.,int(samples))
        c1=p0+np.asarray([np.cos(h0),np.sin(h0)])*(chord/3.)
        c2=p1-np.asarray([np.cos(h1),np.sin(h1)])*(chord/3.)
        points=((1-t)**3)[:,None]*p0+3*((1-t)**2*t)[:,None]*c1+3*((1-t)*t**2)[:,None]*c2+(t**3)[:,None]*p1
        self._arc=np.concatenate(([0.],np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))))
        self._arc_length=float(self._arc[-1]);self.length=self._arc_length;self._points=points
        self._headings=np.unwrap(np.arctan2(np.gradient(points[:,1]),np.gradient(points[:,0])))

    def _sample(self,s):
        distance=float(np.clip(s,0.,self.length))*self._arc_length/max(self.length,1e-8)
        x=np.interp(distance,self._arc,self._points[:,0]);y=np.interp(distance,self._arc,self._points[:,1])
        heading=float(np.interp(distance,self._arc,self._headings))
        return np.asarray([x,y]),heading

    def width_at(self,s):
        return self.width

    def position(self,s,lateral=0.):
        point,heading=self._sample(s)
        right=np.asarray([np.sin(heading),-np.cos(heading)])
        return point+float(lateral)*right

    def heading_theta_at(self,s):
        return self._sample(s)[1]

    def local_coordinates(self,position):
        point=np.asarray(position,dtype=float)[:2]
        distances=np.linalg.norm(self._points-point[None,:],axis=1)
        index=int(np.argmin(distances));center=self._points[index]
        heading=float(self._headings[index]);right=np.asarray([np.sin(heading),-np.cos(heading)])
        logical_s=float(self._arc[index])*self.length/max(self._arc_length,1e-8)
        return logical_s,float((point-center)@right)


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
        # The stock single-block presets end shortly after the interaction
        # zone. Use MetaDrive's native block sequences to leave enough real
        # road for the required post-manoeuvre trajectory horizon.
        # Keep enough generated road after the interaction zone for the full
        # 20-frame warm-up plus a 100-step future demonstration window.
        # Two large-radius native curve blocks provide a sustained bend without
        # turning the map into repeated tight hairpins. Merge keeps MetaDrive's
        # actual on-ramp block, with its radius enlarged to the designed value.
        map_code={"straight":"S"*12, "curve":"CC"+"S"*10, "merge":"r"+"S"*12}.get(scenario_type)
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
                           spawn_velocity=(self.interaction_scenario.elcv_speed,0.),spawn_velocity_car_frame=True,
                           max_speed_km_h=130,use_special_color=True,show_navi_mark=False,show_dest_mark=False),
                # Place the target-lane follower far enough ahead to reach the
                # merge point before the ramp actor; its logical Frenet x is
                # re-centered against the interaction template below.
                "agent1":dict(spawn_lane_index=(">>>","1r0_0_",2),spawn_longitude=60.,
                           spawn_velocity=(self.interaction_scenario.rfv_speed,0.),spawn_velocity_car_frame=True,
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
            "map":map_code,"map_config":{"lane_num":int(lane_num),"lane_width":3.5,"exit_length":50.},
            "block_dist_config":SRAIRLBlockDistConfig,
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

        self.engine.global_config['traffic_vehicle_config']['navigation_module'] = SRAIRLSpawnNavigation
        traffic_seed=self.traffic_seed
        class SeededPGTrafficManager(PGTrafficManager):
            def seed(self, random_seed):
                return super().seed((int(random_seed)+traffic_seed) % (2**32))

            def before_step(self):
                activate_occupied_traffic_blocks(self)
                return super().before_step()

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
        self.interaction_scenario=scenario.validate(int(self.config['map_config']['lane_num']))
        from metadrive.component.pgblock.first_block import FirstPGBlock
        lane_prefix=(FirstPGBlock.NODE_1,FirstPGBlock.NODE_2)
        updates={'agent0':dict(spawn_lane_index=(*lane_prefix,scenario.source_lane),spawn_longitude=scenario.elcv_x,
                               spawn_velocity=(scenario.elcv_speed,0.),spawn_velocity_car_frame=True),
                 'agent1':dict(spawn_lane_index=(*lane_prefix,scenario.target_lane),spawn_longitude=scenario.rfv_x,
                               spawn_velocity=(scenario.rfv_speed,0.),spawn_velocity_car_frame=True)}
        if self.scenario_type=='merge':
            updates['agent0'].update(spawn_lane_index=('1r1_2_','1r1_3_',0),spawn_longitude=4.+scenario.elcv_x-30.)
            updates['agent1'].update(spawn_lane_index=('>>>','1r0_0_',2),spawn_longitude=60.+scenario.rfv_x-15.)
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
        self._interaction_lane_paths={}
        self._interaction_lane_paths[spec.source_lane]=self._lane_path(ego_lane)
        self._interaction_lane_paths[spec.target_lane]=self._lane_path(lanes[spec.target_lane])
        for role,lane_index,x,speed in (
            ('rlv_current',spec.source_lane,spec.rlv_current_x,spec.rlv_current_speed),
            ('rlv_target',spec.target_lane,spec.rlv_target_x,spec.rlv_target_speed)):
            path=self._interaction_lane_paths[lane_index]
            lane,local=self._path_position(path,float(x+self._interaction_s_offset));position=lane.position(local,0.)
            vehicle=self.engine.spawn_object(ReplayVehicle,vehicle_config=dict(length=4.5,width=1.8,height=1.5,
                spawn_position_heading=((float(position[0]),float(position[1])),float(lane.heading_theta_at(local))),
                navigation_module=None,enable_reverse=False,show_navi_mark=False,show_dest_mark=False))
            vehicle.body.setKinematic(True);heading=float(lane.heading_theta_at(float(x+self._interaction_s_offset)))
            vehicle.recorded_velocity=np.asarray([speed*np.cos(heading),speed*np.sin(heading)])
            vehicle.set_velocity(vehicle.recorded_velocity);self.interaction_vehicles[role]=vehicle
        return self.interaction_vehicles

    def _lane_path(self,start_lane,max_edges=64):
        """Follow the geometrically connected lane through the generated graph."""
        graph=self.current_map.road_network.graph;path=[start_lane];visited={tuple(start_lane.index)}
        for _ in range(max_edges-1):
            current=path[-1];node=current.index[1];outgoing=graph.get(node,{})
            end=np.asarray(current.position(float(current.length),0.),dtype=float)
            h0=float(current.heading_theta_at(max(0.,float(current.length)-.1)))
            choices=[]
            for lanes in outgoing.values():
                for lane in lanes:
                    if tuple(lane.index) in visited:continue
                    start=np.asarray(lane.position(0.,0.),dtype=float)
                    h1=float(lane.heading_theta_at(min(.1,float(lane.length))))
                    alignment=float(np.cos(h1-h0));gap=float(np.linalg.norm(start-end))
                    if alignment>.25:
                        choices.append((gap+10.*(1.-alignment),gap,lane))
            if not choices:break
            _,gap,lane=min(choices,key=lambda item:item[0])
            if gap>2.0:break
            path.append(lane);visited.add(tuple(lane.index))
        return path

    @staticmethod
    def _compose_merge_reference_path(ramp_path,mainline_path,transition_length=ROAD_GEOMETRY_CONFIG["merge_transition_length_m"]):
        """Join the taper-ending ramp lane to its adjacent mainline lane smoothly."""
        if len(ramp_path)<1 or len(mainline_path)<2:
            raise RuntimeError("MetaDrive merge map is missing its ramp or mainline lanes")
        source=ramp_path[-1];target=mainline_path[1]
        if source.index[1]!=target.index[1]:
            raise RuntimeError("MetaDrive ramp taper and mainline do not share the expected merge node")
        length=min(float(transition_length),float(source.length),float(target.length))
        source_start=float(source.length)-length
        target_start=source_start
        composed=list(ramp_path[:-1])
        if source_start>1e-6:composed.append(_LaneSlice(source,0.,source_start))
        composed.append(_LaneTransition(source,source_start,target,target_start+length,length))
        target_end=target_start+length
        if target_end<float(target.length)-1e-6:
            composed.append(_LaneSlice(target,target_end,float(target.length)))
        composed.extend(mainline_path[2:])
        return composed

    def _spawn_merge_roles(self,ReplayVehicle,vehicle_class_to_type,vehicle_type):
        """Place explicit ramp and mainline leaders on MetaDrive's connected ramp graph."""
        vehicle_class_to_type[ReplayVehicle]='dataset_replay';vehicle_type['dataset_replay']=ReplayVehicle
        graph=self.current_map.road_network.graph
        self.interaction_vehicles={};self._interaction_elapsed=0.;self._interaction_paths={}
        ego=self.role_vehicle('elcv')
        ramp_native=self._lane_path(ego.lane)
        mainline=self._lane_path(graph['>>>']['1r0_0_'][2])
        ramp=self._compose_merge_reference_path(ramp_native,mainline)
        self._merge_ramp_reference_path=ramp
        transition_index=next(i for i,lane in enumerate(ramp) if isinstance(lane,_LaneTransition))
        self._merge_transition_end_s=sum(float(lane.length) for lane in ramp[:transition_index+1])
        ego_path_s=self._path_distance(ramp,ego.position[:2])
        current_leader_s=ego_path_s+max(12.,self.interaction_scenario.rlv_current_x-
                                             self.interaction_scenario.elcv_x)
        rfv=self.role_vehicle('rfv')
        # Align the real target-lane gap with the ramp actor's physical
        # projection. Fixed mainline spawn s=60 ceases to describe that gap
        # when the highway ramp radius/length changes.
        ego_mainline_s=self._path_distance(mainline,ego.position[:2])
        rfv_path_s=ego_mainline_s-(self.interaction_scenario.elcv_x-self.interaction_scenario.rfv_x)
        rfv_lane,rfv_local=self._path_position(mainline,rfv_path_s)
        rfv_heading=float(rfv_lane.heading_theta_at(rfv_local))
        rfv.set_position(rfv_lane.position(rfv_local,0.))
        rfv.set_heading_theta(rfv_heading)
        rfv.set_velocity(np.asarray([np.cos(rfv_heading),np.sin(rfv_heading)])*self.interaction_scenario.rfv_speed)
        mainline_offset=rfv_path_s-float(self.interaction_scenario.rfv_x)
        # Keep the current-lane leader on the same continuous ramp-to-mainline
        # route as ELCV; the target-lane leader and RFV share the mainline.
        placements={
            'rlv_current':(ramp,current_leader_s,self.interaction_scenario.rlv_current_speed),
            'rlv_target':(mainline,self.interaction_scenario.rlv_target_x+mainline_offset,
                          self.interaction_scenario.rlv_target_speed),
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

    @staticmethod
    def _path_distance(path,position):
        offset=0.
        best=(float('inf'),0.)
        for lane in path:
            local,_=lane.local_coordinates(position)
            clipped=float(np.clip(local,0.,lane.length))
            center=np.asarray(lane.position(clipped,0.),dtype=float)
            distance=float(np.linalg.norm(np.asarray(position,dtype=float)[:2]-center))
            if distance<best[0]:best=(distance,offset+clipped)
            offset+=float(lane.length)
        return float(best[1])

    def advance_interaction_roles(self,dt):
        spec=self.interaction_scenario
        self._interaction_elapsed+=float(dt)
        if self.scenario_type == "merge":
            for role,(path,s0,speed) in self._interaction_paths.items():
                elapsed=self._interaction_elapsed
                current_speed=float(speed)
                if role=="rlv_current":
                    # The ramp leader follows the native taper, then accelerates
                    # smoothly to mainline speed after entering the mainline.
                    merge_time=max(0.,(self._merge_transition_end_s-float(s0))/float(speed))
                    after=max(0.,elapsed-merge_time);accel=1.0
                    accel_duration=max(0.,(float(spec.rlv_target_speed)-float(speed))/accel)
                    accelerated=min(after,accel_duration)
                    distance=(float(speed)*min(elapsed,merge_time)+float(speed)*accelerated+
                              .5*accel*accelerated*accelerated+
                              float(spec.rlv_target_speed)*max(0.,after-accel_duration))
                    current_speed=min(float(spec.rlv_target_speed),float(speed)+accel*after)
                else:
                    distance=float(speed)*elapsed
                lane,local=self._path_position(path,float(s0)+distance)
                position=lane.position(local,0.);heading=float(lane.heading_theta_at(local))
                vehicle=self.interaction_vehicles[role]
                vehicle.set_position((float(position[0]),float(position[1])))
                vehicle.set_heading_theta(heading)
                vehicle.recorded_velocity=np.asarray([current_speed*np.cos(heading),current_speed*np.sin(heading)])
                vehicle.set_velocity(vehicle.recorded_velocity);vehicle.after_step()
            return
        for role,lane_index,x0,speed in (
            ('rlv_current',spec.source_lane,spec.rlv_current_x,spec.rlv_current_speed),
            ('rlv_target',spec.target_lane,spec.rlv_target_x,spec.rlv_target_speed)):
            x=x0+speed*self._interaction_elapsed;path=self._interaction_lane_paths[lane_index]
            lane,local=self._path_position(path,x+self._interaction_s_offset);position=lane.position(local,0.)
            vehicle=self.interaction_vehicles[role]
            vehicle.set_position((float(position[0]),float(position[1])))
            heading=float(lane.heading_theta_at(local));vehicle.set_heading_theta(heading)
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
