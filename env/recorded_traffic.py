"""Timestamp-based background replay; ego remains controlled by the RL policy."""
import math
import numpy as np
from metadrive.manager.base_manager import BaseManager
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.vehicle.vehicle_type import VaryingDynamicsBoundingBoxVehicle, vehicle_type, vehicle_class_to_type


def sample(track, t):
    states=np.asarray(track['states'])
    if t<states[0,0]-1e-6 or t>states[-1,0]+1e-6: return None
    return np.array([np.interp(t,states[:,0],states[:,i]) for i in range(1,5)])


class ReplayVehicle(VaryingDynamicsBoundingBoxVehicle):
    def _add_visualization(self):
        previous = BaseVehicle.model_collection.get(self.path[0])
        super()._add_visualization()
        if previous is None:
            BaseVehicle.model_collection.pop(self.path[0], None)
        else:
            BaseVehicle.model_collection[self.path[0]] = previous
        if self.render:
            # The stock box asset appears black under the offscreen shader.
            # Flat per-vehicle colors keep exact-size replay geometry readable.
            self.origin.setShaderOff(10)
            self.origin.setLightOff(10)
            self.origin.setMaterialOff(10)
            self.origin.setTextureOff(10)
            self.origin.setColor(*(max(.25,float(c)) for c in self.panda_color[:3]),1.,10)

    @property
    def velocity(self):
        return getattr(self,'recorded_velocity',super().velocity)


class RecordedTrafficManager(BaseManager):
    def __init__(self, clip):
        super().__init__()
        vehicle_type['dataset_replay'] = ReplayVehicle
        vehicle_class_to_type[ReplayVehicle] = 'dataset_replay'
        self.clip=clip
        self.tracks=[t for t in clip['tracks'] if not t['ego']]
        self.active={}
        self.tick=0

    @property
    def traffic_vehicles(self):
        return list(self.active.values())

    def before_reset(self):
        super().before_reset();self.active={};self.tick=0

    def reset(self):
        self.place(0.)

    def place(self,t):
        for track in self.tracks:
            state=sample(track,t);key=track['id']
            if state is None:
                if key in self.active:
                    self.clear_objects([self.active.pop(key).id])
                continue
            x,y,vx,vy=state
            heading=math.atan2(vy,vx) if math.hypot(vx,vy)>.1 else 0.
            if key not in self.active:
                vehicle=self.spawn_object(ReplayVehicle,vehicle_config=dict(
                    length=track['length'],width=track['width'],height=1.5,
                    spawn_position_heading=((float(x),float(y)),heading),navigation_module=None,
                    enable_reverse=True,show_navi_mark=False,show_dest_mark=False))
                vehicle.body.setKinematic(True)
                self.active[key]=vehicle
            vehicle=self.active[key]
            vehicle.set_position((float(x),float(y)))
            vehicle.set_heading_theta(heading)
            vehicle.recorded_velocity=np.array([vx,vy])
            vehicle.set_velocity((float(vx),float(vy)))

    def step(self):
        self.tick+=1
        self.place(self.tick*self.engine.global_config['physics_world_step_size'])

    def after_step(self):
        self.place(self.tick*self.engine.global_config['physics_world_step_size'])
        for v in self.active.values():v.after_step()
        return {}
