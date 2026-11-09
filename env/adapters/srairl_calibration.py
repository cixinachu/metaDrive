"""Acceptance gates derived from measured action and physical trajectory errors."""
import math

from env.tasks.srairl_interaction import ROAD_GEOMETRY_CONFIG


def trajectory_calibration_passed(report):
    if not isinstance(report,dict) or report.get('road_geometry')!=ROAD_GEOMETRY_CONFIG:return False
    paths=report.get('paths',[])
    if {p.get('scenario') for p in paths}!={'straight','curve','merge'}:return False
    for path in paths:
        curvature=path.get('max_curvature_per_m',float('inf'))
        if not math.isfinite(curvature) or curvature<0 or not path.get('continuous'):return False
        if 25**2*curvature>ROAD_GEOMETRY_CONFIG['lateral_acceleration_limit_mps2']-ROAD_GEOMETRY_CONFIG['tracking_reserve_mps2']:return False
    rows=report.get('closed_loop_teacher_episodes',[])
    for scene in ('straight','curve','merge'):
        for speed in (15.,20.,25.):
            selected=[r for r in rows if r.get('scenario')==scene and r.get('requested_ego_speed_mps')==speed]
            if not selected:return False
            for row in selected:
                if not row.get('accepted') or row.get('collision') or row.get('out_of_road'):return False
                for key,limit in [('steering_saturation_fraction',.01),('lateral_command_saturation_fraction',.2),
                                  ('mean_abs_lane_lateral_error_m',.5),('max_abs_heading_error_deg',10.)]:
                    value=row.get(key,float('inf'))
                    if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=limit:return False
    return True


def action_calibration_passed(summary, *, require_closed_loop=False):
    if not isinstance(summary,dict) or summary.get('equivalence_status')!='passed':return False
    for axis in ('longitudinal','lateral'):
        for split in ('calibration_grid','held_out'):
            row=summary.get('metrics',{}).get(axis,{}).get(split,{})
            try:
                mae=float(row['mae_mps2']);peak=float(row['max_error_mps2'])
                if not (math.isfinite(mae) and math.isfinite(peak) and 0<=mae<.15 and mae<=peak<.30
                        and row['wrong_sign_count']==0 and row.get('passed') is True):return False
            except (KeyError,TypeError,ValueError):return False
    return not require_closed_loop or trajectory_calibration_passed(summary.get('closed_loop_report'))


def calibration_summary_path():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    config=json.loads((root/'configs/srairl/metadrive_physics.yaml').read_text())
    return (root/config['action_adapter']['calibration_file']).parent/'summary.json'
