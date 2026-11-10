"""Measure curvature and steady lateral-acceleration demand on SR-AIRL maps."""
import argparse
import json
from pathlib import Path
import sys
from dataclasses import replace
import math
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

class DiagnosticProgress:
    """Flush progress to stderr, including during slow imports and map resets."""
    def __init__(self, interval=5.):
        self.interval=interval;self.started=time.monotonic();self.last_print=self.started
        self.message="准备启动";self.lock=threading.Lock();self.stopped=threading.Event()

    def update(self,message,force=False):
        with self.lock:
            self.message=message
            now=time.monotonic()
            if force or now-self.last_print>=self.interval:
                print(f"[{now-self.started:7.1f}s] {message}",file=sys.stderr,flush=True)
                self.last_print=now

    def _heartbeat(self):
        while not self.stopped.wait(min(1.,self.interval)):
            with self.lock:
                now=time.monotonic()
                if now-self.last_print>=self.interval:
                    print(f"[{now-self.started:7.1f}s] 等待中：{self.message}",file=sys.stderr,flush=True)
                    self.last_print=now

    def __enter__(self):
        self.thread=threading.Thread(target=self._heartbeat,daemon=True)
        self.thread.start();return self

    def __exit__(self,*exc):
        self.stopped.set();self.thread.join(timeout=1.)


def collection_progress(progress,scenario,speed,total):
    def callback(state):
        if 'episode_step' in state:
            progress.update(f"{scenario} {speed:g} m/s | 回合 {state['episodes']+1}/{total} | 仿真步 {state['episode_step']}")
        elif 'episode_steps' in state:
            progress.update(f"{scenario} {speed:g} m/s | 回合 {state['episodes']}/{total} 完成 | "
                f"步数 {state['episode_steps']} | 碰撞={state['collision']} 越界={state['out_of_road']}",force=True)
        else:
            progress.update(f"{scenario} {speed:g} m/s | 回合 {state['episodes']+1}/{total} | 初始化地图及车辆、采集历史帧",force=True)
        return True
    return callback


def lane_curvature(lane, samples=400):
    """Return sampled absolute curvature along one generated MetaDrive lane."""
    length = float(lane.length)
    s = np.linspace(0.0, length, max(3, int(samples)))
    heading = np.unwrap(np.asarray([lane.heading_theta_at(float(x)) for x in s]))
    curvature = np.abs(np.gradient(heading, s, edge_order=2))
    return s, curvature


def path_geometry(path, samples_per_lane=400, details=False):
    lanes = []
    all_k = []
    for lane in path:
        s, k = lane_curvature(lane, samples_per_lane)
        all_k.append(k)
        max_index = int(np.argmax(k))
        lanes.append({
            "lane_index": [str(part) for part in lane.index],
            "length_m": float(lane.length),
            "max_curvature_per_m": float(k[max_index]),
            "min_radius_m": float(1.0 / k[max_index]) if k[max_index] > 1e-9 else None,
            "max_curvature_s_m": float(s[max_index]),
        })
    joins=[]
    for first,second in zip(path,path[1:]):
        h0=first.heading_theta_at(first.length);h1=second.heading_theta_at(0.)
        joins.append({'position_gap_m':float(np.linalg.norm(np.asarray(first.position(first.length,0.))-np.asarray(second.position(0.,0.)))),
                      'heading_jump_rad':float(abs((h1-h0+np.pi)%(2*np.pi)-np.pi))})
    values = np.concatenate(all_k) if all_k else np.asarray([0.0])
    max_k = float(np.max(values))
    return {
        "lane_count": len(path),
        "joins": joins,
        "continuous": all(j["position_gap_m"]<0.05 and j["heading_jump_rad"]<0.01 for j in joins),
        "length_m": float(sum(float(lane.length) for lane in path)),
        "max_curvature_per_m": max_k,
        "min_radius_m": float(1.0 / max_k) if max_k > 1e-9 else None,
        "highest_curvature_lanes": sorted(lanes, key=lambda row: row["max_curvature_per_m"],
                                           reverse=True)[:5],
        **({"lanes": lanes} if details else {}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", choices=("straight", "curve", "merge"), default=("straight", "curve", "merge"))
    parser.add_argument("--lane-num", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-lane", type=int, default=400)
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--closed-loop", action="store_true",
                        help="also collect a MetaDrive teacher episode at each requested speed")
    parser.add_argument("--closed-loop-episodes", type=int, default=1)
    parser.add_argument("--speeds", nargs="+", type=float, default=(15, 20, 25))
    parser.add_argument("--progress-interval",type=float,default=5.,
                        help="stderr progress/heartbeat interval in seconds (default: 5)")
    args = parser.parse_args()
    if not math.isfinite(args.progress_interval) or args.progress_interval<=0:
        parser.error('--progress-interval must be finite and positive')
    if args.closed_loop_episodes<1:
        parser.error('--closed-loop-episodes must be positive')
    if any(not math.isfinite(speed) or not 0<speed<=25 for speed in args.speeds):
        parser.error('--speeds must be finite and in (0,25] m/s')
    with DiagnosticProgress(args.progress_interval) as progress:
        count=len(args.scenarios)*len(args.speeds)*args.closed_loop_episodes if args.closed_loop else 0
        progress.update(f"开始诊断：场景={','.join(args.scenarios)}；闭环物理回合共 {count} 个。进度写入 stderr，最终 JSON 写入 stdout。",force=True)
        try:
            run_diagnostics(args,progress)
        except KeyboardInterrupt:
            progress.update('用户中断，诊断未完成；不会输出完整结果 JSON。',force=True)
            raise


def run_diagnostics(args,progress):
    progress.update('加载 MetaDrive 及物理引擎',force=True)
    from env.tasks.srairl_metadrive_world import SRAIRLProceduralWorld
    reports = []
    for scenario in args.scenarios:
        progress.update(f"几何检查 {scenario}：初始化原生道路",force=True)
        world = SRAIRLProceduralWorld(
            scenario_type=scenario, lane_num=args.lane_num, traffic_density=0.0,
            seed=args.seed, horizon=500,
        )
        try:
            world.reset(seed=args.seed)
            if scenario == "merge":world.spawn_interaction_roles()
            ego = world.role_vehicle("elcv")
            ramp_or_curve_path = world._lane_path(ego.lane)
            if scenario == "curve":
                first_road=world.current_map.road_network.graph[">>"][">>>"]
                paths={f"curve_route_lane_{index}":world._lane_path(lane)
                       for index,lane in enumerate(first_road)}
            elif scenario == "straight":
                paths={"straight_route":ramp_or_curve_path}
            else:
                paths={"merge_ramp_to_mainline_route":world._merge_ramp_reference_path}
                graph = world.current_map.road_network.graph
                paths["merge_mainline_lane_2"] = world._lane_path(graph[">>>"]["1r0_0_"][2])
            for name, path in paths.items():
                geometry = path_geometry(path, args.samples_per_lane, details=args.details)
                geometry["scenario"] = scenario
                geometry["path"] = name
                geometry["steady_lateral_acceleration_mps2"] = {
                    str(speed): float(speed * speed * geometry["max_curvature_per_m"])
                    for speed in args.speeds
                }
                reports.append(geometry)
                progress.update(f"几何检查 {scenario}/{name} 完成：曲率={geometry['max_curvature_per_m']:.6g} 1/m，连接连续={geometry['continuous']}",force=True)
        finally:
            world.close()
    closed_loop = []
    if args.closed_loop:
        from env.tasks.srairl_interaction import SRAIRLInteractionScenario
        from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv
        from datasets.srairl.metadrive_expert import collect_metadrive_expert
        groups=len(args.scenarios)*len(args.speeds)
        for scene_index,scenario in enumerate(args.scenarios):
            for speed_index,speed in enumerate(args.speeds):
                group=scene_index*len(args.speeds)+speed_index+1
                progress.update(f"闭环组 {group}/{groups}：{scenario} {speed:g} m/s，{args.closed_loop_episodes} 回合",force=True)
                base=SRAIRLInteractionScenario()
                interaction=replace(base,elcv_speed=speed,rfv_speed=speed,
                    rlv_current_speed=speed,rlv_target_speed=speed)
                env=SRAIRLProceduralEnv(scenario,args.lane_num,0.,args.seed,700,
                    interaction_scenario=interaction)
                try:
                    _,rows,episodes=collect_metadrive_expert(env,args.closed_loop_episodes,args.seed,
                        callback=collection_progress(progress,scenario,speed,args.closed_loop_episodes))
                    for row,episode in zip(rows,episodes):
                        desired=np.asarray(episode["desired_acceleration"],dtype=float)
                        actual=np.asarray(episode["actual_acceleration"],dtype=float)
                        mapped=np.asarray(episode["metadrive_actions"],dtype=float)
                        tracking=np.asarray(episode["tracking_errors"],dtype=float)
                        lateral_actual=np.abs(actual[:,:,1]) if actual.size else np.empty((0,))
                        smoothed_peak=None
                        if lateral_actual.ndim==2 and len(lateral_actual)>=5:
                            smoothed=np.stack([np.convolve(lateral_actual[:,role],np.ones(5)/5.,mode="valid")
                                               for role in range(lateral_actual.shape[1])],axis=1)
                            smoothed_peak=float(np.max(smoothed))
                        closed_loop.append({
                            "scenario":scenario,"requested_ego_speed_mps":float(speed),
                            "episode":row["episode"],"seed":row["seed"],"accepted":row["accepted"],
                            "lane_change_success":row["merge_success"],"collision":row["collision"],
                            "out_of_road":row["out_of_road"],"steps":row["steps"],
                            "lateral_command_saturation_fraction":float(np.mean(np.abs(desired[:,:,1])>=1.499)) if desired.size else None,
                            "steering_saturation_fraction":float(np.mean(np.abs(mapped[:,:,0])>=0.99)) if mapped.size else None,
                            "max_abs_normalized_steering":float(np.max(np.abs(mapped[:,:,0]))) if mapped.size else None,
                            "max_abs_measured_lateral_acceleration_mps2":float(np.max(np.abs(actual[:,:,1]))) if actual.size else None,
                            "max_5_step_mean_abs_lateral_acceleration_mps2":smoothed_peak,
                            "mean_abs_lateral_acceleration_error_mps2":float(np.mean(
                                np.abs(actual[:,:,1]-desired[:,:,1]))) if desired.size else None,
                            "mean_abs_lane_lateral_error_m":float(np.mean(np.abs(tracking[:,:,0]))) if tracking.size else None,
                            "max_abs_lane_lateral_error_m":float(np.max(np.abs(tracking[:,:,0]))) if tracking.size else None,
                            "mean_abs_heading_error_deg":float(np.degrees(np.mean(np.abs(tracking[:,:,1])))) if tracking.size else None,
                            "max_abs_heading_error_deg":float(np.degrees(np.max(np.abs(tracking[:,:,1])))) if tracking.size else None,
                            "validation_errors":episode["validation"]["errors"],
                        })
                    accepted=sum(bool(row['accepted']) for row in rows)
                    progress.update(f"闭环组 {group}/{groups} 完成：{scenario} {speed:g} m/s | 有效 {accepted}/{len(rows)}",force=True)
                finally:
                    env.close()
    progress.update(f"诊断结束：{len(reports)} 条路径、{len(closed_loop)} 个闭环回合；正在输出 JSON",force=True)
    print(json.dumps({
        "simulator": "MetaDrive",
        "road_geometry": __import__("env.tasks.srairl_interaction",fromlist=["ROAD_GEOMETRY_CONFIG"]).ROAD_GEOMETRY_CONFIG,
        "seed": args.seed,
        "lane_num": args.lane_num,
        "samples_per_lane": args.samples_per_lane,
        "speeds_mps": args.speeds,
        "paths": reports,
        "closed_loop_teacher_episodes": closed_loop,
    }, indent=2),flush=True)


if __name__ == "__main__":
    main()
