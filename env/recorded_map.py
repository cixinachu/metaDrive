"""Straight, one-direction mainline in short blocks for stable lane localization."""
import math
from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.straight import Straight
from metadrive.manager.pg_map_manager import PGMapManager


class RecordedMap(PGMap):
    def _big_generate(self,parent_node_path,physics_world):
        # Long single convex lane bodies cause intermittent Bullet ray misses.
        # Build contiguous 50m pieces instead of a single 600–800m hull.
        segment_length=50.
        count=max(1,math.ceil(self._config['exit_length']/segment_length))
        first=FirstPGBlock(self.road_network,self._config['lane_width'],self._config['lane_num'],
                          parent_node_path,physics_world,length=segment_length,remove_negative_lanes=True)
        self.blocks=[first]
        for i in range(1,count):
            block=Straight(i,self.blocks[-1].get_socket(0),self.road_network,random_seed=i,
                           ignore_intersection_checking=True,remove_negative_lanes=True)
            block.construct_block(parent_node_path,physics_world,extra_config={'length':segment_length})
            self.blocks.append(block)


class RecordedMapManager(PGMapManager):
    def spawn_object(self,object_class,*args,**kwargs):
        return super().spawn_object(RecordedMap if object_class is PGMap else object_class,*args,**kwargs)
