"""Collect explicit two-actor rule-based demonstrations in MetaDrive."""
import numpy as np

from datasets.srairl.expert_builder import ExpertTransition


class MetaDriveExpertController:
    """Simple, inspectable driving teacher for the ELCV/RFV pair.

    Each actor follows its own nearest-lane leader with a desired headway and
    gently returns toward its assigned spawn lane. This is a scripted teacher,
    not an optimal or human expert policy.
    """
    def __init__(self, env, desired_speed=22., time_headway=1.4, standstill_gap=7.):
        self.env=env;self.desired_speed=float(desired_speed)
        self.time_headway=float(time_headway);self.standstill_gap=float(standstill_gap)
        self.target_lanes={role:int(env.world.role_vehicle(role).lane_index[2]) for role in ('elcv','rfv')}

    def action(self,role):
        vehicle=self.env.world.role_vehicle(role);lane=vehicle.lane
        x,y=map(float,vehicle.position[:2]);vx,vy=map(float,vehicle.velocity[:2])
        s,lateral=lane.local_coordinates((x,y));heading=float(lane.heading_theta_at(s))
        tangent=np.asarray([np.cos(heading),np.sin(heading)]);normal=np.asarray([-tangent[1],tangent[0]])
        vlong=float(np.asarray([vx,vy])@tangent);vlat=float(np.asarray([vx,vy])@normal)
        leader_gap=float('inf');leader_speed=self.desired_speed
        for other in self.env._vehicle_pool():
            if other is vehicle or tuple(other.lane_index)!=tuple(vehicle.lane_index):continue
            os,_=lane.local_coordinates(other.position[:2]);gap=float(os-s)-float((vehicle.LENGTH+other.LENGTH)*.5)
            if 0.<gap<leader_gap:
                ov=np.asarray(other.velocity[:2]);leader_gap=gap;leader_speed=float(ov@tangent)
        desired_gap=self.standstill_gap+self.time_headway*max(vlong,0.)
        if np.isfinite(leader_gap):
            a_long=float(np.clip(.55*(leader_speed-vlong)+.12*(leader_gap-desired_gap),-4.,2.))
        else:a_long=float(np.clip(.6*(self.desired_speed-vlong),-3.,2.))
        target_lane=self.target_lanes[role]
        lane_index=tuple(vehicle.lane_index);lanes=self.env.world.current_map.road_network.graph.get(lane_index[:2],[])
        lane_center=(target_lane-lane_index[2])*float(self.env.world.config.get('lane_width',3.5))
        a_lat=float(np.clip(1.2*(lane_center-float(lateral))-1.5*vlat,-2.5,2.5))
        return np.asarray([a_long,a_lat],dtype=np.float32)


def collect_metadrive_expert(env, episodes, seed, callback=None):
    """Run the explicit teacher and return paired 16D SR-AIRL transitions."""
    transitions=[];episode_rows=[]
    for episode in range(int(episodes)):
        if callback and not callback(dict(state='running',step=episode,total=int(episodes),episodes=episode)):
            break
        observations,_=env.reset(seed=int(seed)+episode)
        controller=MetaDriveExpertController(env)
        done=False;step=0;collisions=0;offroad=0;distance=0.
        while not done:
            state=observations['elcv'].copy()
            actions={role:controller.action(role) for role in ('elcv','rfv')}
            next_observations,info=env.step(actions)
            next_state=next_observations['elcv'].copy();done=bool(info['done'])
            for role in ('elcv','rfv'):
                transitions.append(ExpertTransition(state.copy(),actions[role].copy(),next_state.copy(),done,
                    role,f"metadrive-{env.scenario_type}-seed{int(seed)+episode}",step))
            step+=1;collisions+=int(bool(info['metadrive_primary_collision']));offroad+=int(bool(info['metadrive_out_of_road']))
            distance=float(info['distance_travelled_m']);observations=next_observations
            if callback and not callback(dict(state='running',step=episode,total=int(episodes),episodes=episode,
                    episode_step=step,global_steps=len(transitions)//2,traffic_vehicle_count=info['traffic_vehicle_count'])):break
        episode_rows.append(dict(episode=episode+1,seed=int(seed)+episode,steps=step,
            distance_travelled_m=distance,collision=bool(collisions),out_of_road=bool(offroad)))
        if callback and not callback(dict(state='running',step=episode+1,total=int(episodes),episodes=episode+1,
                global_steps=len(transitions)//2,episode_steps=step,collision=bool(collisions),out_of_road=bool(offroad))):break
    if not transitions:raise RuntimeError('expert collection stopped before producing transitions')
    return transitions,episode_rows
