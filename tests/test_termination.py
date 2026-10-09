import unittest
from env.safe_metadrive_env import termination_flags
from env.episode_statistics import EpisodeStatistics


class TerminationTests(unittest.TestCase):
    def test_flags(self):
        c = {"terminate_on_collision": False, "terminate_on_out_of_road": True}
        self.assertEqual(termination_flags({"crash": True}, c, False), (False, False))
        self.assertEqual(termination_flags({}, c, True), (False, True))
        self.assertEqual(termination_flags({"out_of_road": True}, c, False), (True, False))
        self.assertEqual(termination_flags({"arrive_dest": True}, c, False), (True, False))
        c = {"terminate_on_collision": True, "terminate_on_out_of_road": False}
        self.assertEqual(termination_flags({"crash": True}, c, False), (True, False))
        self.assertEqual(termination_flags({"out_of_road": True}, c, False), (False, False))

    def test_statistics(self):
        stats = EpisodeStatistics(.1, .5)
        for collision, cost in ((False,.2), (True,1), (True,1), (False,0), (True,1)):
            info = dict(cost=cost, binary_cost=int(collision), continuous_cost=cost, collision=collision, out_of_road=False,
                        min_ttc=2, min_distance=3, min_predicted_distance=1)
            stats.update(info, 1, 10)
        result = stats.result(info)
        self.assertAlmostEqual(result["risk_exposure"], .32)
        self.assertEqual(result["collision_events"], 2)
        self.assertEqual(result["time_to_first_unsafe_event"], .2)
        self.assertEqual(result["episode_length"], 5)
