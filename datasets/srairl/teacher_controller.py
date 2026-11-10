"""Deterministic gap-aware teacher; all commands remain physical accelerations."""
from dataclasses import asdict,dataclass,replace
import math
import numpy as np

TEACHER_VERSION='gap-aware-quintic-v4'
SUPPORTED_TEACHER_VERSIONS=frozenset({'curvature-feedback-v3',TEACHER_VERSION})


@dataclass(frozen=True)
class TeacherConfig:
    time_headway: float=1.2
    standstill_gap_m: float=3.
    lane_change_duration_s: float=8.
    max_wait_s: float=12.
    gap_confirmation_s: float=.4
    minimum_clearance_m: float=3.
    clearance_time_s: float=.2
    comfortable_braking_mps2: float=1.
    front_reaction_s: float=1.
    rear_reaction_s: float=3.
    anticipation_s: float=2.
    max_acceleration_mps2: float=1.5
    max_command_jerk_mps3: float=2.
    merge_gap_request_enabled: bool=True

    def __post_init__(self):
        if not isinstance(self.merge_gap_request_enabled,bool):
            raise ValueError('Merge gap request switch must be boolean')
        if any(not math.isfinite(v) or v<=0 for key,v in asdict(self).items() if key!='merge_gap_request_enabled'):
            raise ValueError('Teacher parameters must be finite and positive')
        if self.max_acceleration_mps2>1.5 or self.comfortable_braking_mps2>self.max_acceleration_mps2:
            raise ValueError('Teacher acceleration must remain inside the calibrated command domain')
        if self.lane_change_duration_s<4.:
            raise ValueError('Teacher lane-change duration must be at least four seconds')


def quintic_progress(elapsed,duration):
    u=float(np.clip(elapsed/duration,0.,1.))
    q=10*u**3-15*u**4+6*u**5
    dq=(30*u**2-60*u**3+30*u**4)/duration
    ddq=(60*u-180*u**2+120*u**3)/(duration*duration)
    return q,dq,ddq


def braking_margin(clearance,follower_speed,leader_speed,delay,config):
    """Constant-speed leader, delayed comfortable follower braking assumption."""
    closing=max(0.,float(follower_speed)-float(leader_speed))
    reserve=config.minimum_clearance_m+config.clearance_time_s*max(0.,follower_speed)
    return float(clearance)-reserve-closing*delay-closing*closing/(2*config.comfortable_braking_mps2)


class MetaDriveExpertController:
    def __init__(self,env,desired_speed=None,time_headway=None,standstill_gap=None,config=None):
        self.env=env;self.config=config or TeacherConfig()
        if time_headway is not None:self.config=replace(self.config,time_headway=float(time_headway))
        if standstill_gap is not None:self.config=replace(self.config,standstill_gap_m=float(standstill_gap))
        self.time_headway=self.config.time_headway;self.standstill_gap=self.config.standstill_gap_m
        spec=env.interaction_scenario
        self.desired_speed=(dict(desired_speed) if isinstance(desired_speed,dict) else
            {role:float(desired_speed) for role in ('elcv','rfv')} if desired_speed is not None else
            {'elcv':float(spec.elcv_speed),'rfv':float(spec.rfv_speed)})
        if set(self.desired_speed)!={'elcv','rfv'} or any(not math.isfinite(v) or not 0<v<=25 for v in self.desired_speed.values()):
            raise ValueError('Teacher desired speeds must be finite and in (0,25] m/s')
        self.target_lanes={'elcv':spec.target_lane,'rfv':spec.target_lane}
        self.phase='waiting';self.refusal_reason=None;self.events=[]
        self._phase_start=float(getattr(env,'time',0.));self._confirmed=0.
        self._last_tick=None;self._actions={};self._last_action={role:np.zeros(2) for role in ('elcv','rfv')}
        self.gap_diagnostics={};self.emergency_braking_count=0;self.command_saturation_count=0
        self.merge_gap_requested=False
        self._progress_anchor=None

    def configuration(self):return asdict(self.config)

    def _set_phase(self,phase,reason=None):
        if phase==self.phase:return
        self.phase=phase;self._phase_start=float(self.env.time)
        self.events.append(dict(type='teacher_phase',phase=phase,time_s=float(self.env.time),reason=reason))
        if phase=='refused':self.refusal_reason=reason

    def set_warmup(self,enabled):
        if not enabled and self.phase=='warmup' and self._last_tick is not None:
            offset=float(self._last_tick[1])+float(self.env.dt)-self._phase_start
            for event in self.events:event['time_s']-=offset
        self._set_phase('warmup' if enabled else 'waiting')
        self._confirmed=0.;self._last_tick=None;self._actions={}
        self._progress_anchor=None
        # Time is reset to t=0 after warm-up; state decisions must use that clock.
        self._phase_start=float(self.env.time)

    def _path(self,role,target=False):
        if self.env._is_merge:
            return self.env.world._merge_ramp_reference_path if role=='elcv' and not target else self.env._merge_mainline_path
        return self.env._target_lane_path if target or role=='rfv' else self.env._source_lane_path

    @staticmethod
    def _nearest_lane(path,position):
        candidates=[]
        for lane in path:
            s,_=lane.local_coordinates(position);s=float(np.clip(s,0.,lane.length))
            candidates.append((float(np.linalg.norm(np.asarray(position)[:2]-lane.position(s,0.))),lane,s))
        _,lane,s=min(candidates,key=lambda item:item[0]);return lane,s

    def _target_lane(self,vehicle,role):
        # Kept for callers inspecting the physical teacher route.
        target=role=='elcv' and not self.env._is_merge and self.phase in ('stabilizing','completed')
        return self._nearest_lane(self._path(role,target),vehicle.position[:2])[0]

    def _gap_status(self):
        ego=self.env.world.role_vehicle('elcv');path=self._path('rfv',True)
        e=self.env._path_state(ego,path);front=None;rear=None
        for other in self.env._vehicle_pool():
            if other is ego:continue
            if np.linalg.norm(np.asarray(other.position[:2])-np.asarray(ego.position[:2]))>350.:continue
            state=self.env._path_state(other,path)
            if abs(state[0])>float(other.WIDTH)*.5+.5:continue
            difference=float(state[1]-e[1]);clearance=abs(difference)-(ego.LENGTH+other.LENGTH)*.5
            if difference>=0 and (front is None or difference<front[0]):front=(difference,clearance,float(state[2]))
            elif difference<0 and (rear is None or -difference<rear[0]):rear=(-difference,clearance,float(state[2]))
        cfg=self.config
        front_margin=math.inf if front is None else braking_margin(front[1],e[2],front[2],cfg.front_reaction_s,cfg)
        rear_margin=math.inf if rear is None else braking_margin(rear[1],rear[2],e[2],cfg.rear_reaction_s,cfg)
        roles=self.env._roles_live()
        slot_order=bool(roles['rfv'][1]<roles['elcv'][1]<roles['rlv_target'][1])
        self.gap_diagnostics=dict(front_clearance_m=None if front is None else front[1],
            rear_clearance_m=None if rear is None else rear[1],
            front_braking_margin_m=None if not math.isfinite(front_margin) else front_margin,
            rear_braking_margin_m=None if not math.isfinite(rear_margin) else rear_margin,
            slot_order_valid=slot_order,safe=bool(slot_order and front_margin>=0 and rear_margin>=0))
        return self.gap_diagnostics['safe']

    def _merge_remaining(self):
        path=self.env.world._merge_ramp_reference_path
        transition=next((i for i,lane in enumerate(path) if hasattr(lane,'source_s') and hasattr(lane,'target_s')),None)
        if transition is None:raise RuntimeError('Merge teacher requires the validated native transition path')
        start=sum(float(lane.length) for lane in path[:transition])
        ego=self.env.world.role_vehicle('elcv')
        return start-self.env.world._path_distance(path,ego.position[:2])

    def _update(self):
        tick=(int(self.env.steps),float(self.env.time))
        if tick==self._last_tick:return
        self._last_tick=tick;self._actions={}
        if self.phase=='warmup' and not self.env._is_merge:return
        safe=self._gap_status()
        if self.phase=='executing':
            position=float(self.env._path_state(self.env.world.role_vehicle('elcv'),self._path('elcv'))[1])
            if self._progress_anchor is None or position-self._progress_anchor[0]>=1.:
                self._progress_anchor=(position,float(self.env.time))
            elif self.env.time-self._progress_anchor[1]>=self.config.max_wait_s:
                self._set_phase('refused','route_blocked')
            if (self.env._is_merge and self._merge_remaining()>0
                    and not self.gap_diagnostics['slot_order_valid']):
                self._set_phase('refused','target_slot_lost')
        if (self.env._is_merge and self.config.merge_gap_request_enabled and self.phase in ('warmup','waiting')
                and self.gap_diagnostics['slot_order_valid']
                and self.gap_diagnostics['rear_braking_margin_m'] is not None
                and self.gap_diagnostics['rear_braking_margin_m']<0
                and not self.merge_gap_requested):
            # Explicit cooperative merge intent, not a claim of calibrated human behavior.
            # Only RFV's bounded following response is requested; entry still waits for a real safe gap.
            self.merge_gap_requested=True
            self.events.append(dict(type='merge_gap_request',time_s=float(self.env.time),reason='rear_gap_insufficient'))
        if self.phase=='waiting':
            if not self.gap_diagnostics['slot_order_valid']:
                self._set_phase('refused','target_slot_lost')
            else:
                self._confirmed=self._confirmed+float(self.env.dt) if safe else 0.
                if self._confirmed+1e-9>=self.config.gap_confirmation_s:self._set_phase('executing')
                elif self.env.time-self._phase_start>=self.config.max_wait_s:self._set_phase('refused','gap_wait_timeout')
        if not self.env._is_merge and self.phase=='executing':
            if self.env.time-self._phase_start>=self.config.lane_change_duration_s:self._set_phase('stabilizing')
        if self.phase in ('executing','stabilizing') and self.env._last_info.get('lane_change_completed'):
            self._set_phase('completed')

    def _reference(self,vehicle,role):
        source,s=self._nearest_lane(self._path(role),vehicle.position[:2])
        point=np.asarray(source.position(s,0.),dtype=float)
        heading=float(source.heading_theta_at(s));dq=ddq=displacement=0.
        if role=='elcv' and not self.env._is_merge:
            target,ts=self._nearest_lane(self._path(role,True),vehicle.position[:2])
            if self.phase=='executing':q,dq,ddq=quintic_progress(self.env.time-self._phase_start,self.config.lane_change_duration_s)
            else:q=1. if self.phase in ('stabilizing','completed') else 0.
            target_point=np.asarray(target.position(ts,0.),dtype=float)
            normal=np.asarray([-np.sin(heading),np.cos(heading)])
            displacement=float((target_point-point)@normal)
            point=(1-q)*point+q*target_point
            target_heading=float(target.heading_theta_at(ts))
            heading+=q*((target_heading-heading+np.pi)%(2*np.pi)-np.pi)
        normal=np.asarray([-np.sin(heading),np.cos(heading)])
        lateral_error=float((np.asarray(vehicle.position[:2])-point)@normal)
        velocity=np.asarray(vehicle.velocity[:2]);speed=max(.5,float(velocity@np.asarray([np.cos(heading),np.sin(heading)])))
        reference_heading=heading+math.atan2(dq*displacement,speed)
        heading_error=(float(vehicle.heading_theta)-reference_heading+np.pi)%(2*np.pi)-np.pi
        ds=min(1.,max(.25,s),max(.25,source.length-s));s0=max(0.,s-ds);s1=min(float(source.length),s+ds)
        delta=(float(source.heading_theta_at(s1))-float(source.heading_theta_at(s0))+np.pi)%(2*np.pi)-np.pi
        curvature=float(delta/max(s1-s0,1e-6))
        return lateral_error,heading_error,float(velocity@normal),dq*displacement,ddq*displacement,curvature,speed

    def tracking_error(self,vehicle,role):
        self._update();reference=self._reference(vehicle,role)
        return float(reference[0]),float(reference[1])

    def action(self,role):
        if role not in ('elcv','rfv'):raise ValueError('Unknown teacher role')
        self._update()
        if role in self._actions:return self._actions[role].copy()
        cfg=self.config;vehicle=self.env.world.role_vehicle(role);path=self._path(role)
        actor=self.env._path_state(vehicle,path);vlong=float(actor[2]);desired=self.desired_speed[role]
        if role=='elcv' and self.env._is_merge and self.phase in ('waiting','refused'):
            remaining=max(0.,self._merge_remaining()-5.)
            desired=min(desired,math.sqrt(2*cfg.max_acceleration_mps2*remaining))
        acceleration=.6*(desired-vlong);emergency=False
        if (role=='rfv' and self.merge_gap_requested and self.phase in ('warmup','waiting','executing','stabilizing')):
            roles=self.env._roles_live()
            gap=float(roles['elcv'][1]-roles['rfv'][1])-(vehicle.LENGTH+self.env.world.role_vehicle('elcv').LENGTH)*.5
            if gap>0:
                desired_gap=cfg.standstill_gap_m+cfg.time_headway*max(vlong,0.)
                acceleration=min(acceleration,.55*(float(roles['elcv'][2])-vlong)+.12*(gap-desired_gap))
        for other in self.env._vehicle_pool():
            if other is vehicle:continue
            if np.linalg.norm(np.asarray(other.position[:2])-np.asarray(vehicle.position[:2]))>350.:continue
            state=self.env._path_state(other,path);distance=float(state[1]-actor[1])
            if distance<=0:continue
            lateral=float(state[0]-actor[0]);future=lateral+float(state[3]-actor[3])*cfg.anticipation_s
            separation=0. if lateral*future<0 else min(abs(lateral),abs(future))
            half_width=(vehicle.WIDTH+other.WIDTH)*.5
            weight=float(np.clip((half_width+.5-separation)/.8,0.,1.))
            if weight<=0:continue
            gap=distance-(vehicle.LENGTH+other.LENGTH)*.5
            desired_gap=cfg.standstill_gap_m+cfg.time_headway*max(vlong,0.)
            follow=.55*(float(state[2])-vlong)+.12*(gap-desired_gap)
            closing=max(0.,vlong-float(state[2]))
            if closing>0:
                follow=min(follow,-closing*closing/(2*max(.1,gap-cfg.minimum_clearance_m)))
            acceleration=min(acceleration,(1-weight)*acceleration+weight*follow)
            if gap<cfg.minimum_clearance_m and abs(lateral)<half_width:
                acceleration=-cfg.max_acceleration_mps2;emergency=True
        error,heading_error,vlat,target_vlat,target_alat,curvature,speed=self._reference(vehicle,role)
        lateral=curvature*speed*speed+target_alat-.6*error-.8*heading_error-1.2*(vlat-target_vlat)
        command=np.asarray([acceleration,lateral],dtype=float)
        if np.any(np.abs(command)>cfg.max_acceleration_mps2):self.command_saturation_count+=1
        command=np.clip(command,-cfg.max_acceleration_mps2,cfg.max_acceleration_mps2)
        limit=cfg.max_command_jerk_mps3*float(self.env.dt)
        limited=np.clip(command,self._last_action[role]-limit,self._last_action[role]+limit)
        if emergency:
            limited[0]=command[0];self.emergency_braking_count+=1
        self._last_action[role]=limited.copy();self._actions[role]=limited.astype(np.float32)
        return self._actions[role].copy()
