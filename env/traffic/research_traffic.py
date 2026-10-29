"""Reactive procedurally generated background traffic shared by tasks."""
from metadrive.manager.traffic_manager import PGTrafficManager


class ResearchTrafficManager(PGTrafficManager):
    """PG traffic with an independent, reproducible traffic seed stream."""
    def __init__(self, seed_manager):
        self.seed_manager = seed_manager
        super().__init__()

    def seed(self, random_seed):
        super().seed(self.seed_manager.traffic(int(random_seed)))

    def reset(self):
        # Also covers the first engine reset (manager may be created after engine.seed).
        self.seed(self.engine.global_random_seed)
        return super().reset()
