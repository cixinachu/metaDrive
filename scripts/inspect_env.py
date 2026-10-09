import json
from evaluate_env import arguments, resolve_config, SafeMetaDriveEnv

if __name__ == "__main__":
    args = arguments()
    env = SafeMetaDriveEnv(research_config=resolve_config(args))
    try:
        obs, info = env.reset()
        print("observation_space:", env.observation_space)
        print("action_space:", env.action_space)
        print("observation valid:", env.observation_space.contains(obs))
        print("physics Hz:", 1 / env.config["physics_world_step_size"])
        print("decision Hz:", 1 / env.dt)
        print("background manager:", type(env.engine.traffic_manager).__name__)
        print(json.dumps(info, default=str, indent=2))
        print(env.step([0, 0.1])[1:])
    finally:
        env.close()
