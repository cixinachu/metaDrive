# MetaDrive 安全强化学习实验室

单智能体驾驶研究环境，支持 SAC、SAC-Lagrangian、WCSAC 和 WCSAC-IQN。
包含真实 MetaDrive 场景预览、实时俯视仿真、训练/评估曲线及文件下载。

## 启动

```bash
conda env create -f environment.yml
conda activate llm_drive
python studio/server.py --port 6002
```

打开 http://localhost:6002 。在训练页选择算法、地图和参数；在评估页选择模型与独立测试场景。
训练从头初始化，不会自动续训评估下拉框中的模型。

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

NGSIM 原始表、处理片段、训练模型、评估结果和本地验收产物不会放入 Git。请按 [NGSIM 数据说明](docs/DATASETS.md) 下载并筛选数据；这些文件由 `.gitignore` 排除，以避免上传数 GB 数据和本地实验产物。

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
