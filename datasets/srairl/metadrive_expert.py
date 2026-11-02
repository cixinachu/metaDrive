"""Collect explicit two-actor rule-based demonstrations in MetaDrive."""
import numpy as np

from datasets.srairl.expert_builder import ExpertTransition


class MetaDriveExpertController:
    """Simple, inspectable driving teacher for the ELCV/RFV pair.

    ELCV tracks the target gap between RFV and RLV_target while RFV follows its
    current lane leader. This is a scripted task teacher, not a human expert.
    """
    def __init__(self, env, desired_speed=22., time_headway=1.4, standstill_gap=7.):
        self.env=env;self.desired_speed=float(desired_speed)
        self.time_headway=float(time_headway);self.standstill_gap=float(standstill_gap)
        spec=env.interaction_scenario
        self.target_lanes={'elcv':int(spec.target_lane),'rfv':int(spec.target_lane)}

    def action(self,role):
        vehicle=self.env.world.role_vehicle(role);lane=vehicle.lane
        x,y=map(float,vehicle.position[:2]);vx,vy=map(float,vehicle.velocity[:2])
        s,lateral=lane.local_coordinates((x,y));heading=float(lane.heading_theta_at(s))
        tangent=np.asarray([np.cos(heading),np.sin(heading)]);normal=np.asarray([-tangent[1],tangent[0]])
        vlong=float(np.asarray([vx,vy])@tangent);vlat=float(np.asarray([vx,vy])@normal)
        leader_gap=float('inf');leader_speed=self.desired_speed
        for other in self.env._vehicle_pool():
            if other is vehicle:continue
            try:other_index=tuple(other.lane_index)
            except AttributeError:
                roles=self.env.world.interaction_vehicles
                if other is roles.get('rlv_current'):other_lane=self.env.interaction_scenario.source_lane
                elif other is roles.get('rlv_target'):other_lane=self.env.interaction_scenario.target_lane
                else:continue
                other_index=(*tuple(vehicle.lane_index[:2]),int(other_lane))
            if other_index!=tuple(vehicle.lane_index):continue
            os,_=lane.local_coordinates(other.position[:2]);gap=float(os-s)-float((vehicle.LENGTH+other.LENGTH)*.5)
            if 0.<gap<leader_gap:
                ov=np.asarray(other.velocity[:2]);leader_gap=gap;leader_speed=float(ov@tangent)
        desired_gap=self.standstill_gap+self.time_headway*max(vlong,0.)
        if np.isfinite(leader_gap):
            a_long=float(np.clip(.55*(leader_speed-vlong)+.12*(leader_gap-desired_gap),-4.,2.))
        else:a_long=float(np.clip(.6*(self.desired_speed-vlong),-3.,2.))
        target_lane_index=self.target_lanes[role]
        lane_a,lane_b=vehicle.lane_index[:2]
        target_lane=self.env.world.current_map.road_network.graph[lane_a][lane_b][target_lane_index]
        target_s,target_lateral=target_lane.local_coordinates((x,y))
        target_heading=float(target_lane.heading_theta_at(target_s))
        target_normal=np.asarray([-np.sin(target_heading),np.cos(target_heading)])
        target_vlat=float(np.asarray([vx,vy])@target_normal)
        a_lat=float(np.clip(1.0*float(target_lateral)-1.5*target_vlat,-2.5,2.5))
        return np.asarray([a_long,a_lat],dtype=np.float32)


def collect_metadrive_expert(env, episodes, seed, callback=None):
    """Run the explicit teacher and return paired 16D SR-AIRL transitions."""
    transitions=[];episode_rows=[]
    for episode in range(int(episodes)):
        if callback and not callback(dict(state='running',step=episode,total=int(episodes),episodes=episode)):
            break
        observations,_=env.reset(seed=int(seed)+episode)
        controller=MetaDriveExpertController(env)
        done=False;step=0;collisions=0;offroad=0;distance=0.;started=False;completed=False
        initial_gap=float(env._initial_gap);final_gap=initial_gap;time_to_lane_change=None
        initial_rfv_speed=float(env._rfv_initial_speed);final_rfv_speed=initial_rfv_speed;rfv_min_accel=0.
        while not done:
            state=observations['elcv'].copy()
            actions={role:controller.action(role) for role in ('elcv','rfv')}
            next_observations,info=env.step(actions)
            next_state=next_observations['elcv'].copy();done=bool(info['done'])
            for role in ('elcv','rfv'):
                transitions.append(ExpertTransition(state.copy(),actions[role].copy(),next_state.copy(),done,
                    role,f"metadrive-{env.scenario_type}-seed{int(seed)+episode}",step))
            step+=1;collisions+=int(bool(info['metadrive_primary_collision']));offroad+=int(bool(info['metadrive_out_of_road']))
            started|=bool(info['lane_change_started']);completed|=bool(info['lane_change_completed'])
            final_gap=float(info['target_gap_m']);time_to_lane_change=info.get('time_to_lane_change')
            final_rfv_speed=float(info['rfv_speed_mps']);rfv_min_accel=min(rfv_min_accel,float(info['rfv_min_accel_mps2']))
            distance=float(info['distance_travelled_m']);observations=next_observations
            if callback and not callback(dict(state='running',step=episode,total=int(episodes),episodes=episode,
                    episode_step=step,global_steps=len(transitions)//2,traffic_vehicle_count=info['traffic_vehicle_count'])):break
        episode_rows.append(dict(episode=episode+1,seed=int(seed)+episode,steps=step,
            distance_travelled_m=distance,collision=bool(collisions),out_of_road=bool(offroad),
            lane_change_started=started,lane_change_completed=completed,
            merge_success=bool(completed and not collisions and not offroad),time_to_lane_change_s=time_to_lane_change,
            initial_target_gap_m=initial_gap,final_target_gap_m=final_gap,target_gap_change_m=final_gap-initial_gap,
            delta_v_rfv_mps=final_rfv_speed-initial_rfv_speed,rfv_min_accel_mps2=rfv_min_accel))
        if callback and not callback(dict(state='running',step=episode+1,total=int(episodes),episodes=episode+1,
                global_steps=len(transitions)//2,episode_steps=step,collision=bool(collisions),out_of_road=bool(offroad))):break
    if not transitions:raise RuntimeError('expert collection stopped before producing transitions')
    return transitions,episode_rows
