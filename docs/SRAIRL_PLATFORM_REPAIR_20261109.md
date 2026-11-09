# SR-AIRL 平台修正与实际验收记录（2026-11-09）

**状态：partial，尚未完成用户要求的完整正式实验验收。** 实际模型在三个场景均于候选可行性阶段失败，没有成功执行完整模型闭环。以下明确区分真实教师诊断、训练接口验证和正式实验。

## 基线与修改范围

- 本地 HEAD：`7d6a2d48f96742901f4caea100aaef9f0595019d`。未 push、未提交、未删除模型，既有正式数据未覆盖。
- 起始已有修改 `archive/manifest.json` 与 `rl/srairl/upstream/Step2_Planner/utils/training.py`，本轮保留。
- Safe RL 算法主链未修改，完整测试套件包含其原有测试。
- 上游奖励方法：`be71aade7acff6455fffc8f1143953ac72bdbca6`，通过 GitHub 实际读取；测试 oracle 的来源见 `tests/fixtures/srairl_upstream/PROVENANCE.txt`。
- 本地奖励网络原有的 11 帧、8 通道三层因果 TCN、预测交互势函数与上游数值一致；新增同参数/同输入数值测试及势函数更新测试。未机械覆盖已有正确实现。

## 关键修正

1. 场景默认速度统一读取 `SCENARIO_SPEED_CONFIG`；Studio API 返回默认值和道路参数，默认不发送速度覆盖。显式自定义后按场景保存，批量场景互不覆盖。
2. 高速原生道路参数集中到 `ROAD_GEOMETRY_CONFIG`：Curve 650 m、Ramp 700 m、加速段 200 m、渐变段 160 m。诊断检查完整参考路径及连接节点，过渡路径采用真实弧长。
3. 教师跟踪真正连续的匝道—主线路径；RFV 按 ELCV 的主线投影初始化，避免道路加长后物理间隙失真。专家历史采用真实 20 帧控制运行，未来至少 100 帧；测量时间以历史末帧 t=0 对齐。
4. 修复 MetaDrive 长车道射线定位偶发漏检：射线未定位时，核验整车四角是否被真实原生车道覆盖；不把虚拟过渡参考线当路面，保留碰撞和实线检测。漏检单独记录。
5. 三场景统一左法向横向坐标，Merge 四车使用同一个主线参考系；训练窗口直接读取记录的参考坐标，与在线 AIRL 归一化一致。候选和 Tracker 通过完整路径转换到真实世界，并增加曲率前馈。
6. 默认关闭标签平滑及 equilibrium 正则，保留显式配置用于消融；PPO AIRL logit 与 Selector 结构化奖励保持分离。反事实标签仍由独立间隙/TTC 筛选。
7. 禁止重复场景/episode seed、混合道路或坐标版本；同 episode 窗口不跨划分。旧数据、旧坐标模型被当前训练和 Bundle 门禁拒绝。
8. 标定门禁重新核算数值误差，并检查完整道路几何、目标速度回合、跟踪误差和饱和，不能只改 passed 字段。
9. ModelBundle 从真实 checkpoint 重建兼容性；校验源数据、架构、时间设置、道路/参考系和 Guide→Reward、Predictor→Guide 内容哈希。
10. 修复跨场景训练的单例引擎清理、RLV 无导航模块访问、路径切片 width_at 接口；RLV 预测仅使用观察历史。不可行候选单独记录为 model_candidates_infeasible。增加控制、实测加速度、jerk、饱和及失败原因记录和 Studio 展示。

## 道路与教师实测

- MetaDrive 原生道路网络及车辆物理保留。诊断 seed=0、三车道、背景密度=0，每个速度每场景仅一个回合；不能视为统计稳定性证明。
- 设计约束：横向需求 v²|κ|，命令域 ±1.5 m/s²，预留 0.5 m/s² 跟踪/换道裕量。实际瞬时加速度另外报告，不把命令限幅等同于实测硬上限。

| 场景 | 最大实际路径曲率 (1/m) | 25 m/s 稳态横向需求 (m/s²) | 路径连接 |
|---|---:|---:|---|
| straight | 0.000000000 | 0.000000 | 连续 |
| curve | 0.001538462 | 0.961538 | 连续 |
| merge | 0.001428571 | 0.892857 | 连续 |

| 场景 | 速度 m/s | 有效 | 碰撞/越界 | 平均横向误差 m | 横向命令饱和比例 | 最大实测横向加速度 m/s² |
|---|---:|---|---|---:|---:|---:|
| straight | 15 | True | False/False | 0.2621 | 0.0102 | 1.6171 |
| straight | 20 | True | False/False | 0.2594 | 0.0136 | 1.6672 |
| straight | 25 | True | False/False | 0.2592 | 0.0137 | 1.6967 |
| curve | 15 | True | False/False | 0.2770 | 0.0203 | 1.6251 |
| curve | 20 | True | False/False | 0.3011 | 0.0270 | 1.6817 |
| curve | 25 | True | False/False | 0.3525 | 0.0405 | 1.7034 |
| merge | 15 | True | False/False | 0.0043 | 0.0000 | 0.3350 |
| merge | 20 | True | False/False | 0.0134 | 0.0000 | 0.5973 |
| merge | 25 | True | False/False | 0.0187 | 0.0000 | 0.8353 |

完整几何、连接、航向误差、转向饱和等见 `outputs/srairl_revision_20261109/road_and_teacher_current.json`。

## 动作标定

| 轴 | 网格 MAE | 留出 MAE | 留出最大误差 | 留出符号错误 |
|---|---:|---:|---:|---:|
| longitudinal | 0.000898 | 0.000901 | 0.005722 | 0 |
| lateral | 0.001507 | 0.019132 | 0.078038 | 0 |

联合门禁结果：`True`。当前文件：`outputs/calibration/srairl_action_adapter_v3/summary.json`；v2 历史文件保留。
标定原始 CSV、表面 NPZ 与图像在同目录。当前专家归档的逐回合标定 revision 字段在采集时缺失；未伪造或重写归档，而在 `expert_calibration_provenance.json` 中用归档 SHA256 绑定本会话实际使用、未改变的标定文件。代码已修复未来采集的逐回合 revision 记录。

## 当前专家数据

| 场景 | 尝试回合 | 有效独立回合 | 20+100 窗口 | train/validation/test 窗口 |
|---|---:|---:|---:|---|
| straight | 3 | 3 | 142 | 47/48/47 |
| curve | 3 | 3 | 143 | 48/48/47 |
| merge | 9 | 4 | 487 | 247/121/119 |

共 10 个有效独立回合、772 个窗口；窗口数未冒充 episode 数。原始四车状态和失败回合保存在 `reference_v2_expert/`、`reference_v2_merge/`。初始间隙及相对速度受控变化，本轮背景交通密度固定为 0，尚未覆盖密度变化和更广随机样本。

Studio 可选归档：
- `datasets/srairl/metadrive/srairl_highway_v4_reference_v2_straight_20261109.npz`
- `datasets/srairl/metadrive/srairl_highway_v4_reference_v2_curve_20261109.npz`
- `datasets/srairl/metadrive/srairl_highway_v4_reference_v2_merge_20261109.npz`

## 实际训练与模型包

**下面均为真实数据、真实优化器更新与验证，不是 mock；但仅为短训练接口验证，未完成正式收敛训练。** 奖励 6 回合、PPO 每次 2 epochs；Goal/Guide/Predictor 各 1 epoch；DPPO 1 update。三场景使用同一组数据来源，种子 60000。

| 组件 | 当前 checkpoint / 目录 | 验证结果 |
|---|---|---|
| SR-AIRL Reward | `outputs/srairl_revision_20261109/current_pipeline_validation/reward/`（双策略、双判别器及 training_state.pt） | 6 回合、210 步，判别器与势函数实际更新；未做正式奖励恢复验收 |
| Goal Predictor | `current_pipeline_validation/planner/planner_goal/results/goal_model.pth` | x RMSE 23.195776 m；速度 RMSE 0.757162 m/s |
| DPPO Planner | `current_pipeline_validation/planner/planner_dppo/state_final.pt` | 1 个真实更新，训练损失见日志 |
| Reward Guide | `current_pipeline_validation/guide/results/best_model.pth` | 验证奖励 MSE 15.652847 |
| Predictor | `current_pipeline_validation/predictor/results/best_model.pth` | 验证 ADE 130.231177 m，明显不合格 |

表中简写路径均相对 `outputs/srairl_revision_20261109/`。训练日志、TensorBoard、验证 summary 和 checkpoint 全部保留。

ModelBundle 严格重载：`True`。实际重载记录 `current_pipeline_validation/strict_bundle_load.json`；Bundle `current_pipeline_validation/model_bundle.json`。它证明当前五类 checkpoint 能真实加载并通过当前兼容检查，不证明模型质量合格。各运行 runtime_metadata.json 保留依赖版本；未验证跨版本运行兼容矩阵。

## 三场景真实模型评估

独立 seed=71000，与专家 10000/11000 系列及训练 60000 分开；每场景候选数=2，真实 MetaDrive 收集 20 帧历史后实际调用 Planner/Predictor。无未来测试轨迹参与选择。

| 场景 | 结果 | 选中候选数 | 完整模型闭环 |
|---|---|---:|---|
| straight | model_candidates_infeasible | 0 | 未完成 |
| curve | model_candidates_infeasible | 0 | 未完成 |
| merge | model_candidates_infeasible | 0 | 未完成 |

失败原因是 Planner/Predictor 候选超出标定纵/横向加速度域，并存在预测 RFV 的道路不可行。候选被阻止执行，没有用回放、随机替代或扩大控制域绕过。
这些回合没有模型控制阶段的有效暴露，不能把 collision=false / out_of_road=false 解读为模型安全率；TTC、Tracker 误差、实际模型加速度/jerk 等没有有效执行样本，记录为空，不伪造 0。完整原因和推理耗时见 `current_pipeline_validation/final_evaluation/{straight,curve,merge}.json`。

## 自动化验证

- `105 passed, 2 warnings, 3 subtests passed in 151.35s (0:02:31)`。
- 最后一次补充验证：19 项 revision 测试通过，包含实际标定加载、三场景物理参考系往返和新增遥测。
- Python compileall、app.js / run.js Node 语法检查通过，详见 syntax_checks.json。
- 场景切换的前端逻辑通过 Node DOM 测试；未运行真实浏览器端到端点击/视频人工验收。
- 新增/完善：场景默认值及隔离覆盖、速度异常、上游数值一致和共享势梯度、真实道路参考系往返、初始车辆物理帧、数据坐标归一化、重复 seed、标定字段篡改拒绝、车辆有向外形碰撞、动作 jerk/饱和、Studio catalog。

## 未通过与剩余工作

1. **完整正式训练未完成。** 现有短训练 Predictor 的 ADE 约 130 m，三个场景无可执行候选。需要在当前版本数据上完成充分训练与验证，不能将现有验证 checkpoint 当作正式研究结果。
2. **三场景成功模型闭环未完成。** 当前均在模型候选门禁失败；双 Tracker 的长回合稳定性、有效碰撞率/越界率、TTC、jerk、控制饱和及完成时间的正式统计尚不可验收。
3. 专家数据仍是小样本、密度 0；Merge 9 次仅 4 次通过，其余真实失败已隔离。仍需扩大受控分布、密度和独立种子，验证教师稳定性及模型泛化。
4. 当前归档的标定 revision 以会话哈希审计旁文件补充，逐 episode 字段未重写；正式大规模采集应使用修复后的记录代码重新生成完整元数据。
5. 浏览器交互/视频人工验收、跨依赖版本兼容矩阵与正式训练验证表现尚未完成。

**evaluation 注册状态保持 partial，未改为 full。**

## 修改文件清单（不含两处起始用户修改）

- `configs/srairl/metadrive_physics.yaml`
- `datasets/srairl/expert_builder.py`
- `datasets/srairl/metadrive/srairl_highway_v4_reference_v2_curve_20261109.npz`
- `datasets/srairl/metadrive/srairl_highway_v4_reference_v2_merge_20261109.npz`
- `datasets/srairl/metadrive/srairl_highway_v4_reference_v2_straight_20261109.npz`
- `datasets/srairl/metadrive_expert.py`
- `datasets/srairl/native_windows.py`
- `docs/SRAIRL_PLATFORM_REPAIR_20261109.md`
- `env/adapters/srairl_action.py`
- `env/adapters/srairl_calibration.py`
- `env/tasks/srairl_interaction.py`
- `env/tasks/srairl_metadrive_world.py`
- `env/tasks/srairl_procedural_env.py`
- `env/tasks/srairl_trajectory_tracker.py`
- `rl/srairl/pipeline/evaluator.py`
- `rl/srairl/pipeline/executor.py`
- `rl/srairl/pipeline/model_bundle.py`
- `rl/srairl/pipeline/provenance.py`
- `rl/srairl/pipeline/role_forecasting.py`
- `rl/srairl/pipeline/selector.py`
- `rl/srairl/reward_config.py`
- `rl/srairl/trainer.py`
- `scripts/calibrate_srairl_action_adapter.py`
- `scripts/collect_srairl_expert.py`
- `scripts/diagnose_srairl_road_geometry.py`
- `scripts/train_srairl.py`
- `studio/config.py`
- `studio/server.py`
- `studio/static/app.js`
- `studio/static/index.html`
- `studio/static/run.js`
- `studio/worker.py`
- `tasks/interaction_evaluation_task.py`
- `tests/fixtures/srairl_upstream/PROVENANCE.txt`
- `tests/fixtures/srairl_upstream/airl.py`
- `tests/test_platform_architecture.py`
- `tests/test_srairl_candidate_pipeline.py`
- `tests/test_srairl_interaction_game.py`
- `tests/test_srairl_protocol_gates.py`
- `tests/test_srairl_revision.py`

最终机器可读汇总：`outputs/srairl_revision_20261109/acceptance_summary.json`。
