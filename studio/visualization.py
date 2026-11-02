"""Optional offscreen camera; the policy still receives vector lidar observations."""
from env.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.obs.observation_base import DummyObservation


class StudioEnv(SafeMetaDriveEnv):
    def __init__(self, research_config, view_mode='topdown', dataset_clip=None, controlled_track_ids=None, reactive_background=False, preview=False):
        self.view_mode = view_mode
        self.reactive_background=bool(reactive_background)
        self.preview=bool(preview)
        if self.preview and self.view_mode != 'topdown':
            # MetaDrive's headless image-observation workaround initializes a
            # throwaway graphics engine before the real preview engine. The
            # preview worker is isolated and captures one frame, so avoid that
            # duplicate warmup; normal train/eval environments are untouched.
            from metadrive.engine import engine_utils
            engine_utils._fix_offscreen_rendering = lambda: None
        super().__init__(research_config=research_config,dataset_clip=dataset_clip,
                         controlled_track_ids=controlled_track_ids)

    def setup_engine(self):
        super().setup_engine()
        if self.dataset_clip and self.interaction_controlled_ids:
            from env.recorded_traffic import InteractionTrafficManager
            self.engine.update_manager("traffic_manager",InteractionTrafficManager(
                self.dataset_clip,self.interaction_controlled_ids,reactive=self.reactive_background))

    def _post_process_config(self, config):
        if self.view_mode != 'topdown':
            width,height=(480,360) if self.preview else (960,720)
            config.update(dict(image_observation=True, window_size=(width,height),
                               show_interface=False, show_logo=False, show_fps=False))
            config['sensors']['main_camera'] = ('MainCamera', width, height)
            config['agent_observation'] = DummyObservation if self.preview else LidarStateObservation
            if self.preview:
                # A static preview does not need MetaDrive's pedestrian and
                # roadside-object warmup pass; it renders the real road/traffic.
                config['preload_models'] = False
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
