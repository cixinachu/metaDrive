"""A ray miss must not prevent valid traffic spawns or admit off-road ones."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from env.tasks.srairl_metadrive_world import SRAIRLSpawnNavigation


@pytest.mark.parametrize('lateral,valid', [(0., True), (2., False)])
def test_configured_lane_fallback_checks_geometry(lateral, valid):
    index = ('a', 'b', 0)
    lane = SimpleNamespace(index=index, length=100.,
        local_coordinates=lambda p: (p[0], p[1]),
        position=lambda s, d: (s, d), width_at=lambda s: 3.5)
    navigation = SimpleNamespace(engine=SimpleNamespace(global_random_seed=42),
        map=SimpleNamespace(road_network=SimpleNamespace(get_lane=lambda i: lane)),
        auto_assign_task=Mock(return_value='destination'), set_route=Mock())
    vehicle = SimpleNamespace(heading=(1., 0.), spawn_place=(20., lateral),
        config={'spawn_lane_index': index, 'destination': None})
    with patch('metadrive.utils.pg.utils.ray_localization', return_value=[]), patch(
            'metadrive.component.navigation_module.base_navigation.BaseNavigation.reset') as reset:
        if valid:
            SRAIRLSpawnNavigation.reset(navigation, vehicle)
            reset.assert_called_once_with(navigation, lane)
            navigation.set_route.assert_called_once_with(index, 'destination')
        else:
            with pytest.raises(ValueError, match='outside configured lane'):
                SRAIRLSpawnNavigation.reset(navigation, vehicle)
            reset.assert_not_called()
            navigation.set_route.assert_not_called()
