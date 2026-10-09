"""Optional offscreen camera; the policy still receives vector lidar observations."""
from env.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.obs.state_obs import LidarStateObservation


class StudioEnv(SafeMetaDriveEnv):
    def __init__(self, research_config, view_mode='topdown'):
        self.view_mode = view_mode
        super().__init__(research_config=research_config)

    def _post_process_config(self, config):
        if self.view_mode != 'topdown':
            config.update(dict(image_observation=True, window_size=(960, 720),
                               show_interface=False, show_logo=False, show_fps=False))
            config['sensors']['main_camera'] = ('MainCamera', 960, 720)
            config['agent_observation'] = LidarStateObservation
        return super()._post_process_config(config)

    def camera_frame(self, overview=False):
        engine = self.engine
        camera = engine.main_camera
        if self.view_mode == 'bird3d' or overview:
            from panda3d.core import NodePath
            x0, x1, y0, y1 = self.current_map.road_network.get_bounding_box()
            center_x, center_y = (x0+x1)/2, (y0+y1)/2
            extent = max(x1-x0, y1-y0, 40)
            # Panda and MetaDrive have opposite y axes.
            pose = NodePath('overview-pose')
            pose.setPos(center_x, -center_y-extent*.65, extent*.85)
            pose.lookAt(center_x, -center_y, 0)
            pixels = camera.perceive(to_float=False, new_parent_node=engine.render,
                                    position=tuple(pose.getPos()), hpr=tuple(pose.getHpr()))
        elif self.view_mode == 'first3d':
            # Forward camera at windshield height, mounted to the ego vehicle.
            pixels = camera.perceive(to_float=False, new_parent_node=self.vehicle.origin,
                                    position=(0., .8, 1.5), hpr=(0., 0., 0.))
        else:
            engine.taskMgr.step()
            pixels = camera.perceive(to_float=False)
        # MainCamera's RAM buffer uses BGR channel order.
        return pixels[..., ::-1]
