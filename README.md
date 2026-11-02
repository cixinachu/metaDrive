# MetaDrive 安全强化学习实验室

自动驾驶科研实验平台，分为 Safe RL 单车 CMDP 与 SR-AIRL 双策略车交互两条独立实验线。Safe RL 支持 SAC、SAC-Lagrangian、WCSAC 和 WCSAC-IQN；SR-AIRL 支持明确绑定角色的 NGSIM 专家片段及 MetaDrive 生成道路。

## 启动

```bash
conda env create -f environment.yml
conda activate llm_drive
python studio/server.py --port 6002
```

打开 http://localhost:6002 。Safe RL 训练/评估使用单车实验流程。SR-AIRL 的 NGSIM 默认流程采用上游运动学和固定背景轨迹回放；MetaDrive 物理及响应式背景属于独立迁移/敏感性实验。当前 SR-AIRL 训练检查点已保存优化器、步数和 RNG 状态，可通过 `SRAIRLTrainer.load_training_state()` 在相同配置与数据下恢复；Studio 尚未提供续训按钮。

## 目录

| 目录 | 内容 |
|---|---|
| `rl/` | 四个独立算法模块、统一注册表和公共实现 |
| `env/` | 研究环境、安全风险计算、统计和 seed 管理 |
| `configs/` | 环境与场景配置 |
| `studio/` | 本地网页、任务进程、实时曲线与下载 |
| `models/<run>/` | 训练模型、checkpoint；Studio 训练日志也保存在此 |
| `results/<run>/` | Studio 评估结果、曲线与日志 |
| `logs/` | 保留的正式环境验收训练日志 |
| `outputs/frozen/` | 最终环境验收记录 |
| `outputs/studio/` | 可删除的场景预览缓存；兼容旧实验 |
| `tests/`、`scripts/` | 回归测试、评估和诊断工具 |

GitHub 仓库包含源码、配置、还原工具，以及 NGSIM 原始表、处理片段、训练模型、实验日志和验收产物的完整归档。`archive/` 中的 47 个分卷总计约 1.95 GB，包含 2,569 个文件，均已通过 SHA256 校验并上传。克隆完整仓库后，可执行以下命令还原原有目录结构，无需 Git LFS。归档解压后约 8.8 GB，代表生成归档时的快照，不自动包含后续实验产物：

```bash
python scripts/restore_full_archive.py
```

还原脚本会核对每个分卷和每个原始文件的 SHA256；已有相同文件会跳过，已有不同内容的文件默认不覆盖。仅检查归档可加 `--verify-only`。归档附带的 `archive/manifest.json` 记录完整文件清单与分卷校验值。`.gitignore` 排除的是本地数据和实验产物。Python 缓存、临时工具和凭据不纳入版本管理。数据来源和许可见 [数据许可说明](datasets/DATA_LICENSE.md)，处理流程见 [NGSIM 数据说明](docs/DATASETS.md)。

已保留环境验收的 100k 模型、两种 reward 的 10k 模型，以及用户的 29,475 步直道训练。
这些模型有不同用途；保留并不意味着安全性能已达标。

## 使用与扩展

- [工作台说明](docs/STUDIO.md)：操作、日志下载及算法预算单位。
- [新增算法](rl/README.md)：添加独立模块并在 `rl/algorithms.py` 注册。
- [环境设计](docs/ENVIRONMENT_DESIGN.md)：reward/cost、场景与评估定义。
- [源码审计](ENVIRONMENT_AUDIT.md)。
- [清理记录](docs/CLEANUP_REPORT.json)。

当前环境使用 259 维向量观测、二维连续动作、0.1 秒决策间隔。默认 reward/cost 分离；
训练 seed 为 1000–1999，评估不得重叠。碰撞默认不终止，越界终止，超时截断。
不能用低 cost 代替驾驶完成度；当前配置不保证每个回合都发生交通交互。

## 检查

```bash
python scripts/test_research_env.py
python -m unittest discover -s tests -p 'test_studio*.py'
# 运行中的工作台使用 8765 端口时，执行四算法真实仿真集成测试：
python scripts/validate_studio.py
```

集成测试会创建新的短训练模型和评估结果；这些属于测试产物，不是正式实验结论。
TensorBoard 可分别使用 `tensorboard --logdir models` 或 `tensorboard --logdir results`。

本次清理删除了旧 LLM/shield、项目内 Ollama、失效旧训练脚本和重复试验产物。
历史冻结清单记录的是验收时的旧版本，不代表清理后每个路径都仍存在；当前代码清单见
`docs/STUDIO_IMPLEMENTATION_MANIFEST.json`。环境验收数据保留，环境风险和动力学定义未改动。

真实高速轨迹接入：见 [HighD / NGSIM 使用说明](docs/DATASETS.md)。NGSIM 官方完整轨迹表（11,850,526 行）与筛选依据见 [NGSIM 数据筛选协议](docs/NGSIM_EXPERIMENT_PROTOCOL.md)；四算法固定清单、验证选模和测试一次性评估见 [NGSIM 比较协议 v3](docs/NGSIM_COMPARISON_PROTOCOL_V3.md)，正式入口为 `python scripts/run_ngsim_comparison.py`（dry-run）及加 `--execute` 的实验运行。HighD 本地导入接口可用，真实数据需官方申请授权。
