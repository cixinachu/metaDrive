# Environment Audit for MetaDrive Safety RL Research

## Full-rollout validation finding (2026-09-22)

The installed engine's default object pooling is not sufficient for full seeded
rollout reproducibility. A fixed SAC checkpoint on merge seed 0 produced episode
lengths 139/193/188 with different collision outcomes under pooled resets.
With the existing MetaDrive configuration `force_destroy=True`, three repeated
rollouts had identical length 139 and zero maximum position/cost difference;
recreating the engine also reproduced them exactly. See
`outputs/rollout_reproducibility.json`.

The research configuration now requires `force_destroy=True`, and tests include
a complete repeated merge episode. The earlier short-trajectory reproducibility
checks were insufficient. Final acceptance runs have completed under
`outputs/frozen/`; pre-fix training and baseline results are development evidence,
not the final frozen baseline. All 23 tests, both 100-episode baselines, two
10k smoke runs, the corrected 100k learning run, and the 9-case evaluation matrix
passed their interface checks. See design section 14 for the v1 freeze decision.

## 2026-09-21 source verification addendum (before implementation)

The active shell Python is base Conda and cannot import MetaDrive. Use the
Python interpreter from the `llm_drive` Conda environment defined by `environment.yml`.
The source examined is the installed `metadrive` package in that environment.
Earlier runtime claims below are historical evidence, to be rechecked by the new inspector.
There is no AGENTS.md in this workspace. No SAC algorithm changes are authorized.

- `MetaDriveEnv.reward_function` **replaces** progress/speed reward on a crash;
  merely setting penalties to zero still removes task reward on unsafe steps.
  CMDP mode must preserve `step_reward` and apply only destination reward.
- `done_function` always terminates building crashes and optionally terminates
  vehicle/object/human crashes. Set explicit research termination semantics.
  Set `truncate_as_terminate=False` and keep horizon truncation separate.
- `PGTrafficManager._create_vehicles_once` uses `IDMPolicy`. Density is a
  fraction of spawn capacity (`floor(total_lane_length / VEHICLE_GAP) * density`,
  rounded down), not vehicles per metre. Triggered vehicles become active by road.
- `dist_to_left_side/right_side` measure **route corridor** lateral distances;
  they are not nearest physical road edges in junctions. Do not label them as
  physical margins. Investigate actual boundary sensing before freezing.
- Default block distribution V2 includes Straight, Curve, StdInterSection,
  StdTInterSection, Roundabout, InRampOnStraight and OutRampOnStraight.
  Resolve diagnostic IDs from these classes, not guessed strings.
- `random_traffic=True` skips ordinary traffic-manager reseeding and does not
  mean randomized IDM aggressiveness. Keep false; expose unsupported behavior
  randomization explicitly instead of silently aliasing these concepts.
- Existing training callback evaluates on training seeds, and references
  `scripts/diagnose_policy.py`, which is absent in this checkout.
- PyYAML is absent. Configuration may use JSON syntax (a YAML subset) with
  stdlib parsing, avoiding an unneeded installation.

Required next: configuration, dedicated research environment, pure risk math,
explicit statistics/seeds, separate diagnostic/evaluation interface, tests,
random/rule baselines, then unchanged SB3 SAC smoke validation.

## Scope

This document records the current environment state before any algorithm changes. The goal is to audit what the project already does, what is already available from MetaDrive, and what must be changed to support a CMDP-style single-agent driving environment with explicit safety cost and clean train/test separation.

No RL algorithm structure is modified in this phase.

---

## 1. Current MetaDrive version

Current project environment:

- Conda environment: `llm_drive`
- Installed simulator package: `metadrive-simulator==0.4.3`
- Runtime inspection from the project environment confirms the active package is MetaDrive 0.4.3.

Evidence from the current environment:

- `environment.yml` declares `metadrive-simulator`
- `MetaDriveEnv.default_config()` in the installed package includes:
  - `start_seed: 0`
  - `traffic_density: 0.1`
  - `num_scenarios: 1`
  - `map: 3`
  - `random_traffic: False`
  - `decision_repeat: 5`
  - `physics_world_step_size: 0.02`
  - `traffic_mode: trigger`

This is the source of truth for the current code path.

---

## 2. Current SAC environment class

The project currently wraps the simulator through a thin factory:

- `env/driving_env.py`
- `make_metadrive_env(**overrides)`

Current behavior:

- `default_metadrive_config()` returns a dictionary with:
  - `start_seed`
  - `use_render`
  - `num_scenarios`
  - `traffic_density`
  - `horizon`
  - `image_observation: False`
  - `log_level: 50`
- `make_metadrive_env()` calls `MetaDriveEnv(config)`.
- The training path in `rl/sac_train.py` builds a `DummyVecEnv` with `Monitor(env)` around one `MetaDriveEnv` instance.

So the current SAC baseline is using the default MetaDrive environment class directly, not a custom custom-env wrapper that implements a strict CMDP interface.

---

## 3. Observation space

Current runtime observation is confirmed as:

- `np.ndarray`
- shape: `(259,)`
- dtype: `float32`
- `env.observation_space`: `Box(0.0, 1.0, (259,), float32)`

This is a low-dimensional vector observation that the project already assumes is valid for the SAC baseline. This is consistent with the project description and supports the current training workflow.

Important note:

- The current observation is not explicitly organized into a typed CMDP state with separate ego state, navigation, road, nearby traffic, and safety fields.
- It is simply the simulator-provided vector observation used by the baseline policy.

---

## 4. Action space

Current runtime action space is:

- `env.action_space`: `Box(-1.0, 1.0, (2,), float32)`
- Action dimension: 2
- Interpretation:
  - steering
  - throttle / brake (continuous control in a single action vector)

This matches the intended single-agent continuous driving control interface and does not need algorithmic change at this stage.

---

## 5. Reward function

The current reward is the default MetaDrive reward, not an explicit CMDP reward decomposition.

Project evidence:

- `MetaDriveEnv.default_config()` contains `driving_reward: 1.0`
- Penalties include:
  - `crash_vehicle_penalty: 5.0`
  - `crash_object_penalty: 5.0`
  - `out_of_road_penalty: 5.0`
  - `crash_sidewalk_penalty: 0.0`

This means the current environment includes crash and out-of-road penalties as part of the built-in driving reward.

Current project behavior:

- `step()` returns `reward`
- `info['step_reward']` also exists
- `reward` is not currently decomposed into `task reward` and `safety cost`

This is exactly the issue to fix for a future CMDP interface: reward and cost are not separated.

---

## 6. Cost existence

Yes, there is already a `cost` field in the environment info, but it is not yet a principled safety cost interface.

Confirmed runtime info keys include:

- `info['cost']`
- `info['crash']`
- `info['out_of_road']`
- `info['arrive_dest']`
- `info['route_completion']`

However:

- `cost` is currently ambient MetaDrive output, not a documented CMDP safety signal
- There is no explicit `binary_cost` / `continuous_cost` interface
- There is no `vehicle_risk`, `road_risk`, or `critical_vehicle_id`
- There is no standardization for episode-level risk statistics

Therefore, the project already exposes a built-in cost-like field, but it is not yet a clean continuous safety cost suitable for Safe RL research.

---

## 7. Termination / truncation conditions

Current behavior is the default MetaDrive logic, not a custom environment policy.

The current runtime step returns:

- `terminated`
- `truncated`
- `done = terminated or truncated`

The project does not yet define a custom split such as:

- `termination: terminate_on_collision: false`
- `termination: terminate_on_out_of_road: true`

Current safety semantics are therefore not yet explicit or consistent with the planned CMDP design.

The current baseline project is effectively using the default simulator termination rules and horizon behavior without documenting them as a research-level policy choice.

---

## 8. Traffic density

The current default configuration sets:

- `traffic_density = 0.1`

The runtime config confirms:

- `traffic_density` is accepted by `MetaDriveEnv`
- `random_traffic` is `False` by default
- `traffic_mode` is `trigger`

Important caveat:

- The current project passes `traffic_density` as a plain scalar and uses MetaDrive’s built-in background traffic generation.
- There is not yet a separate explicit configuration object for `training traffic density` vs `evaluation traffic density` vs `stress traffic density`.

This is acceptable for a baseline, but insufficient for the safety-study environment needed later.

---

## 9. Map / scenario generation configuration

Current project uses a simple map factory:

- `env/driving_env.py`
- `parse_map(value)` accepts:
  - integer block count, e.g. `3`
  - block string, e.g. `S`, `SC`

Training config examples in `README.md` use:

- `--map S`
- `--map SC`
- `--map 3`

Current default wrapper config also includes:

- `num_scenarios: 1`
- `start_seed: seed`

At present:

- the project uses procedural generation for training and evaluation with a single map setup
- there is no explicit train/test split by scenario seed family
- there are no named fixed diagnostic scenarios like straight / merge / intersection / T-junction / roundabout in the codebase yet

This is not yet the research design required for safety evaluation.

---

## 10. Horizon

Current project config:

- `default_metadrive_config()` sets `horizon = 1000`
- training and evaluation command-line arguments pass `--horizon` and default to `1000`

Important note:

- `MetaDriveEnv.default_config()['horizon']` is `None` in the installed package
- the project wrapper explicitly sets horizon to `1000`, so the actual runtime environment is not using the raw default but the project wrapper default

Thus the effective horizon for the baseline experiments is: `1000` steps, controlled by project wrapper configuration.

---

## 11. Simulation frequency / decision frequency

Confirmed runtime values from the installed MetaDrive config:

- `physics_world_step_size = 0.02`
- `decision_repeat = 5`

Thus, the effective time scale is approximately:

- physics timestep: `0.02 s`
- control decision interval: `0.02 * 5 = 0.1 s`

This matters because all risk quantities that depend on time-to-collision, closing speed, and exposure are measured against a real time scale. The project currently does not explicitly log or standardize this in an environment interface.

---

## 12. Random seed usage

Current project uses a simple seed setup:

- wrapper default: `seed=0`
- `make_metadrive_env()` maps `seed` to `start_seed`
- training environment: `build_env(seed, traffic_density, horizon, map_spec, num_scenarios)`
- evaluation: `env.reset(seed=scenario_seed)`, where `scenario_seed = args.seed + episode % args.num_scenarios`

This means:

- the project uses a single `seed` value for the environment and training repeatability
- there is no explicit separation between:
  - RL training seed
  - environment scene seed
  - traffic randomness seed
  - evaluation seed
- no centralized seed manager exists

This is adequate for a baseline smoke test, but insufficient for a rigorous experimental protocol.

---

## 13. Current train/test scenario isolation

Current state: not isolated.

Evidence:

- training and evaluation both use the same environment factory and similar scenario configuration
- training uses `build_env(seed, traffic_density, horizon, map, num_scenarios)`
- evaluation reuses the same `seed` family and a small scenario count
- there is no designated training seed range and evaluation seed range
- there are no separate ID / OOD scenario folders or fixed diagnostic scenarios

Therefore, the project currently does not meet the requirement of strict train/test scenario separation needed for experimental validity.

---

## Current strengths

- The project has a working MetaDrive baseline wrapper.
- The SAC baseline is trainable in the current environment.
- The runtime observation and action interfaces are already continuous and compatible with the baseline.
- The environment already exposes a lightweight `info['cost']` field and crash/out-of-road flags.
- The simulator supports procedural maps, traffic density control, and continuous action control.

---

## Required design changes before the safety RL environment can be frozen

The current project must be extended to satisfy the research design.

### Mandatory changes

1. Create an explicit CMDP environment contract.
   - single-agent ego vehicle control
   - background traffic generated via MetaDrive traffic policy
   - no multi-agent RL
   - state decomposed into ego + navigation + road + traffic features

2. Separate reward from safety cost.
   - `reward_mode = "metadrive"` and `reward_mode = "cmdp"`
   - in `cmdp` mode, remove safety penalties from reward
   - keep the raw default MetaDrive reward available for sanity checks

3. Add explicit safety cost modes.
   - `cost_mode = "binary"`
   - `cost_mode = "continuous"`
   - all safety cost values must be in `[0, 1]`

4. Implement continuous vehicle interaction risk.
   - TTC-based risk
   - dynamic safe-distance risk
   - closest-approach risk with predicted future distance

5. Add road-boundary risk.
   - continuous risk based on edge margin, not only `in/out of road`

6. Standardize `info` fields.
   - `info["cost"]`
   - `info["binary_cost"]`
   - `info["vehicle_risk"]`
   - `info["road_risk"]`
   - `info["critical_vehicle_id"]`
   - `info["min_ttc"]`
   - `info["min_distance"]`
   - `info["min_predicted_distance"]`
   - `info["time_to_closest_approach"]`
   - `info["collision"]`
   - `info["out_of_road"]`

7. Implement episode-level safety statistics.
   - cumulative cost
   - cost integral over time
   - collision count
   - out-of-road events
   - min TTC / min distance / min predicted distance
   - episode duration and success / completion metrics

8. Define train/test split explicitly.
   - training scene seeds separated from evaluation scene seeds
   - fixed ID and OOD diagnostic scenarios
   - no overlap between train and test seeds

9. Add config-driven safety parameters.
   - `safety.ttc_threshold`
   - `safety.min_distance`
   - `safety.time_headway`
   - `safety.comfortable_deceleration`
   - `safety.prediction_horizon`
   - `safety.road_margin`
   - `safety.vehicle_risk_aggregation`

10. Add environment unit tests before any RL training beyond smoke tests.
   - reset / step stability
   - cost range checks
   - collision cost = 1
   - OOR cost = 1
   - reproducibility with fixed seed
   - train/test non-overlap

11. Add diagnostic scripts for cost and policy sanity.
   - random-policy diagnostics
   - rule-based baseline diagnostics
   - fixed scenario evaluation without training

---

## High-priority risk summary

The most important conclusion is this:

- The current project is a valid baseline MetaDrive + SAC environment, but it is not yet a valid safety-CMDP environment.
- The current built-in cost field is not enough to support later WCSAC / distributional safety critic work.
- The current train/test separation is not strict enough for research-grade evaluation.
- The current reward and safety logic are not distinguished in a principled way.

This is a documentation-level audit of the current environment, not a code change. The next step is to implement a dedicated environment wrapper and config system that makes these distinctions explicit while keeping the current SAC baseline compatible.

---

## Final audit verdict

The current environment is suitable only for:

- baseline smoke testing
- simple SAC training sanity checks
- compatibility checks with the current MetaDrive simulator

It is not yet suitable to freeze as a unified experimental environment for:

- SAC / SAC-Lagrangian / WCSAC / WCSAC-IQN
- safety-cost benchmarking
- robust ID/OOD evaluation
- reproducible CMDP environment research

The environment must be redesigned and standardized before it can be considered frozen for future research.
