# MetaDrive Studio 使用与算法说明

在项目根目录运行：

```bash
conda activate llm_drive
python studio/server.py --port 8765
```

浏览器打开 http://localhost:8765 。后端仅监听本机。Python 标准库提供网页服务，
前端不依赖 CDN。运行依赖沿用 `environment.yml`（SB3、Torch、Pillow、TensorBoard、MetaDrive/pygame）。
Playwright 仅用于开发验收，不是工作台运行依赖。

## 页面操作

1. 在「训练实验」选择 SAC、SAC-Lagrangian、WCSAC 或 WCSAC-IQN。
2. 选择程序化道路、直道、汇入、十字路口、T 字路口或环岛。点击场景后自动启动真实场景预览。
   图标仅用于识别拓扑，中央画面来自 MetaDrive 原生 top-down renderer；预览展示全图，运行时跟随 ego。
   改密度或 seed 后点击「生成所选场景的真实预览」更新。一次实验选择一种地图生成类型。
3. 设置训练步数、场景范围、密度、horizon、设备等。展开高级选项设置优化器、网络宽度、安全预算。
4. 点击开始训练。预热前不出现 loss；完整回合结束后出现 episode 指标。侧面显示实际记录的全部标量，支持筛选和平滑。
5. 「停止并保存」在下一环境步骤结束后正常退出，保存最终 checkpoint。新地图初始化期间停止可能稍有等待。
6. 「模型评估」选择 checkpoint，自动识别算法和训练 reward/cost 设置。选择独立测试 seed 后开始评估。
7. 每条曲线旁的 CSV 下载保存原始数据（不受展示平滑影响）；支持全部曲线 CSV、逐回合 CSV、
   TensorBoard 日志 ZIP、完整实验 ZIP、最终模型和当前帧。历史记录可重新打开。

现有 `models/*/*.zip` 中 SAC checkpoint 会出现在选择器。算法统一由本地 `rl/algorithms.py` 注册和创建，四个独立模块为 `rl/sac.py`、`rl/sac_lagrangian.py`、`rl/wcsac.py`、`rl/wcsac_iqn.py`；
公共优化代码复用 `rl/constrained_sac.py`。新增算法说明见 `rl/README.md`。
新模型、训练日志与中间 checkpoint 保存在 `models/<train-run>/`；
评估结果、曲线和日志保存在 `results/<eval-run>/`。旧 `outputs/studio/` 模型仍可选择。评估不更新网络。

## 文件与记录

- `request.json`：所有 UI 参数及输入模型路径。
- `environment_config.json`：完整环境配置。
- `runtime_metadata.json`：依赖版本、源码哈希和输入模型哈希。
- `metrics.jsonl`：step、wall elapsed、实际标量；下载 CSV 不丢弃记录。
- `episodes.jsonl`：完整回合的任务与安全指标。中途停止的未完整回合不参与汇总。
- `tensorboard/events.out.tfevents.*`、`progress.csv`：可由真实 TensorBoard 读取。
- `model.zip`、`model.json`：最终权重、优化器状态、算法与配置；周期性 `checkpoint_*.zip`。
- `frame.jpg`：最新原生仿真俯视帧；`worker.log`：错误及运行输出。

也可运行 `tensorboard --logdir models`（训练）或 `tensorboard --logdir results`（评估） 查看原始 TensorBoard。
模型保存优化器状态，但此界面目前不提供继续训练按钮，不保存 replay buffer，不能把重新加载权重当作逐位恢复完整训练。

## 算法实现口径

普通 SAC 直接使用 SB3 2.3.2 的 SAC；旧训练脚本已清理，场景 seed 适配独立放在 `rl/vec_env.py`；环境风险与动力学定义保持不变。
新算法在 `rl/constrained_sac.py` 中继承 SB3 SAC，沿用其 actor、双 reward critic、自动熵调节，
新增包含 cost 的 replay buffer、safety target network 与独立优化器。
三个算法的 safety critic 都不向 cost Bellman target 加 entropy。

参考：Yang et al., [Safety-constrained reinforcement learning with a distributional safety critic](https://link.springer.com/article/10.1007/s10994-022-06187-8)，
尤其第 4 节公式 13–23。这里是项目内实现，非作者官方代码或已完成论文数值复现。

- SAC-Lagrangian：非负 mean cost critic，Bellman MSE；actor 加入 `lambda * Q_cost`。
- WCSAC（Gaussian）：独立均值、方差网络，使用论文式 14/15 的矩投影目标，
  最小化均值差平方与标准差差平方（单维 Gaussian W2）。为数值稳定方差下限 1e-6；
  真终止目标方差归零到该下限。
- WCSAC-IQN：余弦分位数 embedding，与状态动作特征相乘，输出 cost quantile；
  独立均匀采样当前/目标 quantile，成对 quantile Huber loss（kappa=1）。
  critic 学习完整分布，actor 在 `[1-alpha,1]` 采样估计最坏 alpha 比例的风险。
- 约束乘子：softplus 参数化，Adam 优化 `-lambda*(risk-budget)`，risk 超限时提高安全权重。
  log 参数限制在 [-20,100] 作为数值保护；这是明确的实现选择。
- target network：Polyak 更新；SB3 正常终止掩码与 time-limit bootstrap 保持一致。

**预算单位**：`cost_limit` 直接对应 replay 状态的折扣剩余 cost，
`C_t = sum_k gamma_cost^k cost_(t+k)`，不是整回合未折扣 sum、不是碰撞概率，也不是风险暴露秒数。
UI 默认预算 5、tail_fraction 0.1 只是起始参数，需要在独立验证集校准。
该 replay-state 约束不能宣称每个状态均满足约束或有实际安全保证。

## 评估与公平性

训练 seed 仍限制 1000–1999，评估禁止与其重叠。评估 episode 必须不大于场景数，按 seed 顺序各跑一次；
这避免重复同一 seed 伪装独立样本，但不保证地图拓扑互异。
所有算法共享环境 reward/cost；评估强制与训练时 reward/cost 模式一致。
交通密度与道路类型可以用于分布变化测试，所有最终有效参数都保存。
普通 SAC 的旧模型缺少 cost 折扣配置时，评估使用页面的 gamma_cost/tail_fraction，仅作为统计参数。

曲线 `episode/*` 为当前回合，`rolling20/*` 为最近 20 回合，`evaluation/*` 为已完成评估回合累计汇总。
不把训练回合成功率称为测试成功率。
评估额外计算实测初始状态有限回合 discounted_cost 的经验上尾 CVaR，按尾部质量做分数权重，
不是简单对离散数量取 ceil。它与训练 replay-state 预测风险并非同一个分布，不能直接对比判断约束满足。
`effective_tail_samples = episodes * tail_fraction` 可用于判断尾部样本是否明显不足。
本界面不会自动选择验证集，也不会自动执行多训练 seed 的统计对照。

## 验证

```bash
python -m unittest tests.test_studio_algorithms -v
# 工作台服务运行时：4 个算法各训练 160 步、重载并评估 2 回合，验证画面/曲线/下载
python scripts/validate_studio.py
```

短测试只验证优化更新、保存重载和前后端流程，不证明算法收敛或安全性能。
环境 v1 冻结清单保留历史基准；新增算法与工作台另行记录源码版本，不改写原冻结环境定义。

### 清理后的验证

保留环境最终验收数据 `outputs/frozen/`。旧 Studio 短训练模型、截图和临时验收产物已清理，
需要时可通过 `scripts/validate_studio.py`、`validate_studio_controls.py`、`validate_studio_browser.py` 重新生成。
集成测试默认访问本机 8765 端口。
正式验收模型及用户训练保存在 `models/`；删除清单见 `docs/CLEANUP_REPORT.json`。
历史环境冻结清单描述旧版路径，当前文件哈希见 `docs/STUDIO_IMPLEMENTATION_MANIFEST.json`。

### 三维画面

场景设置中的“显示视角”支持三维鸟瞰、三维跟车、二维俯视；新任务默认三维鸟瞰。
选择后生成预览，或启动训练/评估即可显示。视角在新任务启动时生效，历史任务仍展示当时保存的画面。
三维画面来自 MetaDrive 离屏相机，包含真实道路材质与车辆模型，需要可用的 Panda3D 图形上下文。
三维会增加运行开销，可选择二维俯视提高训练速度。鸟瞰覆盖整张道路地图，跟车视角位于自车后方。
`studio/visualization.py` 仅为网页开启渲染，相机图像不进入算法；策略继续使用 259 维 LidarStateObservation。
参考图片中的收费站、停车场、多智能体或施工障碍并未因此加入当前六种道路配置。
验证：同一 seed 与 30 个固定动作下，三种视角的观测、reward、cost 一致（绝对误差阈值 1e-6）。

第一人称（车内向前）将相机固定在自车上，随车辆转向，用于预览、训练和评估；不包含完整驾驶舱装饰。

### 多车道设置

场景面板可选择主路每方向 1–4 车道，默认 3；预览、训练和评估共用该参数。
匝道、路口连接段和程序化道路局部变化仍遵循 MetaDrive 道路模块规则。
车道数写入任务设置及环境配置，并随模型保存；选择评估模型时自动回填，可再修改以测试跨车道数泛化。
旧任务未记录此选项时按 3 车道兼容。相同交通密度下，改变车道数也可能改变背景车辆数量。
验证了六类场景 × 四种车道数的生成和单步执行（seed 1000），以及 23 项环境回归测试。

### 真实轨迹来源

网页现可选择程序化道路、NGSIM、HighD，并按训练/评估划分选择已导入片段；详情见 [DATASETS.md](DATASETS.md)。数据场景使用固定背景回放，单片段评估仅计一个场景。
