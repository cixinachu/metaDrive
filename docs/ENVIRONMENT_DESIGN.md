# MetaDrive 单车安全驾驶实验环境

本文定义当前研究环境的可执行接口和已知限制。环境审计见
[ENVIRONMENT_AUDIT.md](../ENVIRONMENT_AUDIT.md)。验证数据保存在 `outputs/`；
最终训练验证状态另见本文末尾，不能把 smoke test 正常运行当作收敛证明。

## 1. Problem formulation

任务是 **Single-agent autonomous driving in a dynamic multi-vehicle traffic environment**。
一个 RL policy 只控制 ego；背景车辆由 MetaDrive 的 `IDMPolicy` 控制，包括其内置车道选择行为。
环境为 `SafeMetaDriveEnv(MetaDriveEnv)`，不是 multi-agent RL。研究接口按
\(\mathcal M=(\mathcal S,\mathcal A,P,r,c,\gamma)\) 组织；\(P\) 来自车辆动力学、道路与背景交通交互。

完整模拟器状态包含 ego、导航、道路和其他车辆；policy 使用原生有限范围向量观测，
不声称该观测包含完整 Markov 状态。安全指标可使用模拟器真实交通状态，属于环境提供的
ground-truth cost，未额外加入 policy observation。SAC 只优化 reward，环境不定义 \(\gamma_c\)。

## 2. Runtime、observation 和 action

已在本机 Conda `llm_drive` 实际运行确认：

| Item | Value |
|---|---|
| MetaDrive | `metadrive-simulator==0.4.3` |
| Physics backend | `panda3d==1.10.13`，由 MetaDrive 0.4.3 精确依赖锁定 |
| Stable Baselines3 | `2.3.2` |
| Gymnasium | `0.29.1` |
| observation_space | `Box(0.0, 1.0, (259,), float32)`（打印可能显示 `-0.0`） |
| observation 内容 | 原生 ego 状态、导航/道路信息、lidar 交通表示 |
| action_space | `Box(-1.0, 1.0, (2,), float32)` |
| action 顺序 | steering，throttle/brake；第二维正值加速、负值制动 |

不增加 camera、BEV、Transformer、GNN、LLM 或额外感知网络。
`reset()` 返回 `(obs, info)`，`step()` 返回 `(obs, reward, terminated, truncated, info)`。
episode 结束后再次 `step()` 会报错，必须先 reset。

## 3. Reward

参数在 `configs/env/base.yaml` 的 `reward` 段：`driving_reward=1.0`、
`speed_reward=0.1`、`success_reward=10.0`、`use_lateral_reward=false`，与核实的原生默认值一致。

`reward_mode="metadrive"` 保留当前版本原生 reward：先计算纵向进度和归一化速度收益；
到达终点时替换为成功奖励；越界/碰撞时按原生分支替换为相应惩罚。原生 vehicle/object/OOR
惩罚为 5，sidewalk 惩罚为 0。该模式只保留 **reward**，研究环境的终止和 cost 语义仍保持统一。

`reward_mode="cmdp"` 复用原生 `step_reward`：

\[
r_{task}=w_p\Delta\ell\,s_{road}+w_v\frac{v_{km/h}}{v_{max,km/h}}s_{road}.
\]

道路方向符号及参考 lane 选择与原生实现一致；到达终点时仍返回 `success_reward`。
碰撞/越界惩罚配置为零，且不进入原生替换分支，从而不会把当步进度奖励隐式清零。
TTC、边界裕度、安全距离均不进入 reward。已有 SAC 的网络、激活函数、优化器、replay buffer
和 reward-only objective 不变。当前工作台由 `studio/worker.py` 记录安全信息；
普通 SAC 仅优化 reward，三个安全变体通过各自算法模块使用独立 cost。训练统一使用研究环境。

## 4. Safety cost：单位与参数

位置、距离和车身长宽使用米；world-frame `vehicle.velocity` 使用 m/s；heading 使用弧度；
预测时间、TTC 使用秒。不要把 `speed_km_h` 带入安全距离公式。
世界坐标位置增量与速度向量的关系另有真实仿真积分测试，避免二维预测中混用轴向或单位。

以下值集中定义在 `configs/env/base.yaml`。除指定的 4 s TTC 阈值外，风险参数是第一版
**显式设计选择**，不是冒充 MetaDrive 默认值或已校准的真实道路标准。

| Parameter | Value | Rationale / interpretation |
|---|---:|---|
| `ttc_threshold` | 4.0 s | 用户要求的第一版 TTC 窗口 |
| `min_distance` | 2.0 m | 静态跟车净间距基项 |
| `time_headway` | 1.5 s | 速度相关跟车余量 |
| `comfortable_deceleration` | 3.0 m/s² | 相对速度制动余量的参考减速度；不是舒适性 cost |
| `prediction_horizon` | 4.0 s | 常速度最近接预测窗口 |
| `encounter_margin` | 2.0 m | 预测车身净距的软风险带 |
| `road_margin` | 1.0 m | 车身到有效边界的软风险带 |
| `road_sensor_range` | 20.0 m | 真实边界传感器射线长度 |
| `road_sensor_rays` | 120 | 环绕 ego 的均匀角度采样 |
| `same_direction_cosine` | cos(30°) | 跟车 TTC 的同向资格条件 |
| `critical_threshold` | 0.5 | critical step 的统计阈值 |
| `epsilon` | 1e-6 | 数值防护，尤其接近零的相对速度 |
| `vehicle_risk_aggregation` | `max` | 不按车辆数量线性叠加 |

这些参数仍需要在正式论文实验前进行敏感性分析；当前验证证明公式和接口工作，
不证明这些数值是最优安全标定。

### Binary mode

\[
c_{binary}=\mathbf 1[collision\lor out\_of\_road].
\]

`cost_mode="binary"` 时 `info["cost"]` 使用此值；连续分项仍计算并保留供诊断。
collision 使用原生 `crash` 的合并语义，包含 vehicle/object/building/human/sidewalk 接触。

### 跟车 TTC

仅对 ego 前方、方向差不超过 30°、车身横向投影重叠的车辆启用跟车分项。
`gap` 是双方车身沿 ego 朝向的投影净距，而非中心距。closing speed 是沿 ego 朝向的
\((v_{ego}-v_j)\) 投影。

若 gap > 0 且 closing speed > epsilon，则 TTC = gap / closing speed。
非接近状态 TTC 为 `None`；已经没有净距时 TTC 为 0。

\[
c_{TTC}^j=\begin{cases}
1,&TTC_j\le0,\\
(1-TTC_j/T_{safe})^2,&0<TTC_j<T_{safe},\\
0,&TTC_j\ge T_{safe}\text{ 或无有效 TTC}.
\end{cases}
\]

因此，不会用一个巨大哨兵数表示“没有 TTC”。很大的**有限** TTC 若实际算出则保留，
不应与无意义的魔法数字混淆。

### 动态安全距离

对相同的跟车候选车辆：

\[
d_{safe}=d_0+T_h\max(v_{ego}^{\parallel},0)
+\frac{\max(v_{ego}^{\parallel}-v_j^{\parallel},0)^2}{2b_{safe}},
\qquad c_d^j=\left[\max\left(0,1-\frac{gap_j}{\max(d_{safe},\epsilon)}\right)\right]^2.
\]

该分项覆盖“速度相近但跟车很近”的情况。速度只用于安全余量，没有速度效率、舒适性或时间惩罚。

### 二维最近接风险

所有已激活背景车辆都进入二维评估，不限于同向车辆：

\[
r=p_j-p_{ego},\quad v^{rel}=v_j-v_{ego},\quad
\tau^*=clip\left(-\frac{r^Tv^{rel}}{\|v^{rel}\|^2+\epsilon},0,H_{risk}\right).
\]

中心最近接位置为 \(r+\tau^*v^{rel}\)。风险使用该时刻双方**保持当前朝向的矩形车身**
之间的净距 \(d_{rect}^*\)，避免用大圆盘造成正常相邻车道的虚假重叠。
矩形相交时净距为 0；否则计算顶点到对方边段的最短欧氏距离。

\[
c_{encounter}^j=
\left[\max\left(0,1-\frac{d_{rect}^*}{d_{encounter}}\right)\right]^2
\left[\max\left(0,1-\frac{\tau^*}{H_{risk}}\right)\right]^2.
\]

这是常速度、固定朝向的预测近似，**不是**弯道真实碰撞时间求解器，
也不是在整个时间窗内优化矩形净距的精确最小值。`min_predicted_distance` 对应上述
中心最近接时刻的矩形净距，不能解释为完全真实的未来最小间距。
横穿、cut-in、旋转不变性已有受控测试；资格门控为分段规则，不保证关于完整状态全局可微。

\[
c_j=\max(c_{TTC}^j,c_d^j,c_{encounter}^j),\qquad
c_{vehicle}=\max_j c_j.
\]

无背景车时 vehicle risk 为 0。`softmax` 名称保留，但调用会显式报错，第一版仅启用 max。

### 道路边界风险

不使用 `dist_to_left_side/right_side` 作为物理边界：源码确认这些是导航走廊投影，
路口处并不可靠。本实现调用原生 `SideDetector`，在 static physics world 中检测
`ContinuousLaneLine | Sidewalk`，即当前研究所定义的有效连续线/路缘边界。

传感器返回归一化射线距离，乘以量程还原米。命中点变换到 ego 车身坐标系后，
计算命中点到 ego 矩形车身的净距，取所有有效命中点中的最小值：

\[
d_{sample}=\min_k\sqrt{\max(|x_k|-L/2,0)^2+\max(|y_k|-W/2,0)^2}.
\]

\[
c_{road}=\left[\max\left(0,1-\frac{\max(d_{sample},0)}{d_{edge}^{safe}}\right)\right]^2.
\]

无命中时 `road_clearance=None`、road risk=0；实际 out-of-road 强制 road risk=1。
该指标是采样边界点的车身净距，**不是精确的最近边界距离**：有限角度和遮挡可能高估裕度。
它明确避开导航走廊假几何，但仍保留传感器近似这一限制。

直线贴近边界的测量已验证单调且与已知几何相符（考虑碰撞体线宽，误差允许 0.2 m）。
五类道路 seed=0、每类最多 300 个 IDM 决策、每 10 步采样，对比 120 和 720 条射线：
距离差 p95=0.01062 m、最大 0.19084 m；road risk 差 p95=0.00344、最大 0.00589。
数据见 `outputs/boundary_sampling.json`。这不是覆盖所有地图、所有姿态的精度保证。

### 最终 cost

\[
c_{continuous}=\max(c_{vehicle},c_{road},c_{binary})\in[0,1].
\]

collision 或 out-of-road 一律 cost=1；其他情况下按 cost mode 选 binary/continuous。
对非有限车辆状态、道路测量及最终风险显式报错，不静默将 NaN 转成安全状态。

## 5. Step info

| Field | Meaning |
|---|---|
| `cost` | 当前 cost mode 的原始单步信号 |
| `binary_cost`, `continuous_cost` | 两套信号同时保存 |
| `vehicle_risk`, `road_risk` | 两类连续风险分项 |
| `min_ttc` | 所有有效跟车 TTC 的最小值 |
| `min_distance` | 当前车身矩形净距的跨车最小值 |
| `min_predicted_distance` | 最近接时刻矩形净距的跨车最小值 |
| `time_to_closest_approach` | 最危险车辆对应的预测时间 |
| `collision`, `out_of_road` | 模拟器原始事件语义 |
| `critical_vehicle_id` | 当前最危险车辆的运行期 ID |
| `critical_vehicle_distance/ttc/risk` | 与该 ID 一致的诊断量 |
| `road_clearance` | 上述采样净距，未命中为 None |
| `active_traffic_count` | 真正激活的背景车数量 |
| `metadrive_cost` | 原生 cost，仅供核查，不是研究 cost |
| `scene_seed`, `rl_seed`, `environment_seed` | 场景、RL、场景采样 RNG 的种子 |
| `traffic_seed`, `traffic_randomness_seed`, `evaluation_seed` | 交通基种子、派生交通 RNG seed、评估策略 RNG seed |
| `dt` | 每次策略决策对应的仿真秒数 |

没有有效量时使用 `None`，CSV 中为空。跨车最小 TTC、最小距离可能来自不同车辆，
不要把它们强行拼成同一对车的状态；单车对应关系应读取 critical 字段。
ID 用于一次运行内追踪，不保证跨进程对象名称一致。

## 6. Episode statistics

终止或截断的最后一步提供 `info["episode_safety"]`，包括：

\[
C_{undiscounted}=\sum_t c_t,\qquad C_{exposure}=\sum_t c_t\Delta t.
\]

| Field | Definition |
|---|---|
| `episode_reward` | reward 总和 |
| `episode_risk_sum`, `risk_exposure`, `mean_step_risk` | 风险总和、风险加权时间、平均单步风险 |
| `binary_unsafe_steps` | collision 或 out-of-road 的决策步数 |
| `collision_events` | 合并 collision 标志从 false 到 true 的次数 |
| `collision`, `out_of_road` | episode 内是否曾发生，而非仅看最后一步 |
| `min_ttc`, `ttc_p05` | 有效逐步 min TTC 的最小值和 5% 分位；缺失值不参与 |
| `min_distance`, `min_predicted_distance` | 整个 episode 的最小对应距离 |
| `time_to_first_unsafe_event` | 首个 binary unsafe 的观测时刻 |
| `safety_critical_steps`, `time_to_first_critical_event` | continuous_cost >= 0.5 的计数与首次时刻 |
| `episode_length`, `travel_time` | 决策数、决策数乘 dt |
| `success`, `route_completion` | 原生到达目标和路线完成度 |
| `mean_speed` | 逐决策 world velocity 范数的均值，m/s |
| `terminated`, `truncated` | 独立记录 episode 结束类型 |

首个事件发生在第 k 次 step 的观测结果中，时间记为 k*dt。collision_events 是合并接触标志
的上升沿计数，不能解释为物理冲击次数或同时撞到的车辆数。统计不折扣，未加入未来算法假设。

## 7. Termination / truncation

| Event | Default result |
|---|---|
| 到达目标 | terminated=true |
| 车辆/其他 collision | 不立即结束；`terminate_on_collision=true` 时结束 |
| out-of-road | terminated=true；可配置继续 |
| horizon | truncated=true，不标记为独立 failure |

两个标志可能在同一步同时为 true（例如恰好在 horizon 到达目标）。
out-of-road 采用原生判定：不在可用车道上，或接触有效连续黄/白线/sidewalk；
并非只有车身中心完全离开沥青才算越界。`out_of_route_done=false`、broken line 不触发越界。
默认 `terminate_on_collision=false`、`terminate_on_out_of_road=true`。

## 8. Scenario generation 和 train/test split

训练默认 `map=3`、3 个车道、lane width=3.5 m、exit_length=50 m，采用原生 V2 procedural
block distribution：Curve 0.30、Straight 0.10、InRampOnStraight 0.10、OutRampOnStraight 0.10、
StdInterSection 0.15、StdTInterSection 0.15、Roundabout 0.10。每张地图不要求同时包含全部类型，
这些道路类型在训练地图族中有正采样概率。

训练保留范围由 `configs/env/train.yaml` 定义：1000–1999；默认 ID test 为 0–199。
配置校验禁止测试集合与保留训练区间重叠；显式 reset(seed=...) 也必须属于本环境配置范围。
大于等于 2000 的额外测试 seeds 可通过 CLI 指定，不代表已经做过分布偏移有效性验证。

| Diagnostic config | Actual block class / code |
|---|---|
| `straight.yaml` | Straight / `S` |
| `merge.yaml` | InRampOnStraight / `r` |
| `intersection.yaml` | StdInterSection / `X` |
| `t_junction.yaml` | StdTInterSection / `T` |
| `roundabout.yaml` | Roundabout / `O` |

代码已经与安装版本类定义核对。最终诊断配置默认固定 seed=0、num_scenarios=1，
形成五个可重复的固定测试场景。需要同拓扑的更多变体时，CLI 可显式覆盖 seed 范围。
这不是训练五个 policy；同一个 checkpoint 在所有诊断环境中复用。
直线配置提供可出现跟车的动态道路，不是强制注入某条预定轨迹的 car-following 测试；
cut-in 等几何风险另由受控单元测试覆盖。

## 9. Traffic

`ResearchTrafficManager` 继承原生 `PGTrafficManager`，只分离 traffic RNG；交通车辆仍由
原生 `IDMPolicy` 生成和控制。源码与动态运行测试都确认了这一点。
默认 trigger 模式：相关道路触发后车辆才进入 active traffic 列表，未激活对象不作为动态风险源。

trigger 模式 density 的含义是生成容量比例：先按 lane 总长 / `VEHICLE_GAP` 估计候选容量，
再乘 density 并向下取整。它不是 vehicles/m。主训练 0.10；评估独立指定 0.05、0.10、0.20、0.30。
切换结构不会自动改变 density。

`traffic.randomize_behavior=false`。MetaDrive `random_traffic=true` 会跳过普通 reseed，
并不等价于 IDM 激进程度随机化，因此当前保持 false；请求 behavior randomization 会明确报错。
允许显式 `--traffic-config FILE` 覆盖交通 mode/vehicle_config；未实现 aggressive / heterogeneous IDM 参数模式。

## 10. Randomness 和复现边界

默认 rl_seed=0、environment_seed=42、traffic_seed=43、evaluation_seed=44。
环境 SeedManager 的独立 NumPy Generator 选择训练 scene seed；traffic RNG seed 由
`SeedSequence([traffic_seed, scene_seed])` 派生并显式应用于交通 manager。
MetaDrive 自身其他引擎随机性仍受 scene seed 控制，不能声称仅一个 traffic seed 控制所有引擎随机源。

SB3 的全局 RNG、网络初始化和探索 action_space 由 rl_seed 控制。研究版 VecEnv 不把 rl_seed
再传给地图 reset，防止 RL seed=0 意外进入测试地图。评估随机策略在脚本中单独用
evaluation_seed 和 scene_seed 重置 action_space，不侵入训练探索 RNG。

已验证固定场景 initial observation 重复、存在动态 IDM 交通时的轨迹和 cost 重复。
完整 SAC merge 轨迹进一步暴露了原生对象池复用问题：同 seed 的三次运行长度为 139/193/188，
碰撞标签也不同。将 `force_destroy=true` 后，三次轨迹长度均为 139，位置和 cost 的最大差均为 0；
每次新建引擎也得到相同结果。详见 `outputs/rollout_reproducibility.json`。
因此最终研究配置强制销毁旧车辆/物体，不允许恢复对象池复用，并加入完整 merge episode 回归测试。
未宣称跨硬件、CUDA/CPU 或所有接触状态下逐 bit 确定。

每次评估保存完整配置、版本、源码哈希、checkpoint 哈希、逐 episode seeds 和 episode 顺序。
训练保存参数、环境配置、版本和源码哈希；日志目录与模型目录同名。
评估可用 `--device cpu/cuda/auto` 固定推理设备，并记录实际设备；不同设备之间不要求逐 bit 相同。
对象池修复后的两次独立 CUDA CMDP 训练，在前 10k 步内的 12 个完整 episode 中，
scene seed、episode 长度、碰撞/越界标签、reward、cost sum 和 exposure 完全一致；
最大 reward/cost 差为 0。证据见 `outputs/frozen/training_prefix_reproducibility.json`。

## 11. Simulation time

physics timestep=0.02 s（50 Hz），decision_repeat=5，environment step 与 policy decision
均为 0.1 s（10 Hz）。horizon=1000 对应最多 100 s 仿真时间，不是程序 wall-clock 时间。
cost 在每个决策结束状态采样；风险暴露是离散积分，不声称捕捉每个物理子步的连续风险峰值。

## 12. Evaluation interface

```bash
python scripts/evaluate_env.py --policy sac --model models/RUN/sac_metadrive_baseline.zip \
  --scenario intersection --traffic-density 0.20 --seed-start 0 --num-scenarios 200
python scripts/evaluate_matrix.py --model models/RUN/sac_metadrive_baseline.zip --episodes 2
```

矩阵覆盖 unseen procedural ID、low/high/stress density 和五类固定结构。
每个 run 保存 `config.json`、`episodes.jsonl`、`episodes.csv`、`summary.json`；
`--steps-csv` 另存 `steps.csv`。episode JSONL 增量写入，完整结果目录不覆盖既有结果。
所有任务/安全指标均出现在 episode 表中；summary 提供主要均值/率，不能用小样本率冒充论文结论。

## 13. 历史开发记录

对象池修复前的临时模型、日志与基线结果已清理；正式结果以第 14 节及 `outputs/frozen/` 为准。
清理后的算法训练由 Studio 工作台调用 `rl/` 模块；旧 `rl/sac_train.py` 入口已移除。
历史冻结清单用于追溯当时版本，不是清理后的文件完整性清单。

## 14. 对象池修复后的验收（最终版本，已完成）

本节对应强制 `force_destroy=true` 的最终代码配置，以 `outputs/frozen/` 下的数据为准。
第 13 节的早期结果仅保留开发历史，不作为这一版本的正式基线。

| Baseline | Episodes | Success | Collision any | OOR any | Mean cost sum | Mean exposure |
|---|---:|---:|---:|---:|---:|---:|
| Random | 100 | 0% | 0% | 13% | 64.83513 | 6.48351 s |
| IDM | 100 | 67% | 26% | 32% | 25.15966 | 2.51597 s |

两组逐步 CSV 与 episode 汇总的 cost sum、risk exposure、collision-any 全部一致。
cost 均在 [0,1]；所有实际碰撞/越界步 cost=1。最终随机/IDM 平均单步风险分别为
0.068519/0.058911。不能脱离驾驶进度，把低速停留的随机策略视为更安全的驾驶策略。

- 完整环境测试：23 项全部通过，包含完整 merge episode 复现。
- CMDP reward：10k smoke test 通过，12 个完整 episode，loss 有限。
- MetaDrive reward：10k smoke test 通过，12 个完整 episode，loss 有限。
- 最终版本 100k 学习检查：通过，336 个完整 episode、285 个不同训练场景，loss 全部有限。
- 最终 checkpoint 的 ID/OOD 矩阵：9 组共 18 个 episode 全部完成。

100k CMDP 训练使用 CUDA，场景样本范围为 1007–1996，共 93 个 episode 成功。
前 20 / 后 20 个 episode 平均 reward 为 12.95336 / 338.69659，成功率为 0% / 55%，
路线完成度为 0.04376 / 0.81876，风险暴露为 5.22524 / 11.82821 s。
可学习性检查通过，但不能据此宣称统计收敛或策略安全。
最终模型：`models/sac_metadrive_20260922_162550_315328/sac_metadrive_baseline.zip`。
学习曲线：`outputs/frozen/learning_sanity/learning_curves.png`。

| Evaluation | Episodes | Success | Collision any | OOR any | Mean cost sum | Mean exposure |
|---|---:|---:|---:|---:|---:|---:|
| ID density 0.10 | 2 | 100% | 100% | 0% | 154.04176 | 15.40418 s |
| Low density 0.05 | 2 | 0% | 100% | 100% | 18.35727 | 1.83573 s |
| High density 0.20 | 2 | 0% | 50% | 100% | 114.75443 | 11.47544 s |
| Stress density 0.30 | 2 | 0% | 100% | 100% | 223.57300 | 22.35730 s |
| Straight | 2 | 100% | 0% | 0% | 8.00265 | 0.80026 s |
| Intersection | 2 | 0% | 0% | 0% | 1.83641 | 0.18364 s |
| Merge | 2 | 0% | 100% | 0% | 113.86648 | 11.38665 s |
| T-junction | 2 | 100% | 0% | 0% | 1.54175 | 0.15417 s |
| Roundabout | 2 | 100% | 0% | 0% | 3.71714 | 0.37171 s |

固定结构的两次运行是相同 seed=0 的重复检查，五类全部 episode 指标完全一致，
不能算作独立泛化样本。Intersection、merge 均到达 horizon=1000，未触发成功。
ID 两次均成功且均碰撞，这符合 collision 不终止的定义，也说明 success 必须与 safety 分开报告。
这些小样本只验收评估接口；论文结论仍需多训练 seed、足量独立测试场景和置信区间。
复现证据：`outputs/frozen/fixed_scenario_reproducibility.json`。

### 最终模块验收

| Module | Status | File | Test | Remaining Issue |
|---|---|---|---|---|
| 源码审计与环境定义 | 完成 | `ENVIRONMENT_AUDIT.md`, `docs/ENVIRONMENT_DESIGN.md` | 安装版 API、单位和默认配置核查 | 版本升级须重新审计 |
| Gymnasium 接口与终止 | 通过 | `env/safe_metadrive_env.py` | reset/step、空间、dt、terminated/truncated | 无阻塞项 |
| Reward 与 cost 分离 | 通过 | `env/safe_metadrive_env.py`, `env/safety_cost.py` | 两种 reward、binary/continuous、事件 cost=1 | 参数是设计选择，非校准概率 |
| TTC / 动态距离 / CPA | 通过 | `env/risk_utils.py` | 单调性、跟车、横穿、cut-in、矩形几何 | CPA 使用恒速预测 |
| 道路边界风险 | 通过 | `env/safe_metadrive_env.py` | 五类地图、120/720 射线对照 | 有限角度采样近似 |
| Episode 统计与日志 | 通过 | `env/episode_statistics.py`, `scripts/analyze_safety_runs.py` | 200 个基线 episode 逐步/汇总一致 | 无阻塞项 |
| 场景 / seed / 背景交通 | 通过 | `env/research_config.py`, `configs/` | train/test 无交集、完整轨迹、固定场景重复 | 固定 force_destroy；不保证跨设备逐 bit 一致 |
| Random / IDM 基线 | 通过 | `scripts/evaluate_env.py`, `outputs/frozen/` | 各 100 episode | IDM 不是安全上界 |
| SAC 接口与学习 | 通过 | `rl/sac.py`, `studio/worker.py` | 两种 reward 各 10k + CMDP 100k；cost 不参与优化 | 当前策略不安全，尚未验证收敛 |
| ID / OOD / stress | 通过 | `scripts/evaluate_matrix.py` | 9 组共 18 episode | 正式研究需扩大独立样本量 |

用户文档的 21 项环境验收条件均已完成；统一机器可读报告为
`outputs/frozen/validation_summary.json`，代码/配置 SHA256 与依赖版本为
`docs/ENVIRONMENT_FREEZE_MANIFEST.json`。

### 冻结结论

**可以冻结为 v1 统一实验环境，用于后续 SAC / SAC-Lagrangian / WCSAC / WCSAC-IQN。**
当前无环境阶段阻塞项。冻结对象是本报告的环境定义、配置、版本、指标与接口，
不是当前 reward-only SAC 的安全性或算法效果。后续算法应保持本环境 reward/cost/seed 协议一致，
通过独立算法模块使用 cost；本阶段没有实现或修改 Lagrangian、safety critic、WCSAC 或 IQN。
MetaDrive 0.4.3、Panda3D 1.10.13、`force_destroy=true` 及清单中的代码配置作为复现实验基准。
风险近似与小样本限制已明确记录；更改这些定义或依赖后应产生新版本并重新验证。

