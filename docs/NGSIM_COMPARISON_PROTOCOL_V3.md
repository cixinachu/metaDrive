# 固定 NGSIM 轨迹回放下的策略比较（v3）

本协议将 v1 和 v2 已筛出的片段固定为一个共享实验集，供 SAC、SAC-Lagrangian、WCSAC、WCSAC-IQN 使用。它评估的是**固定 NGSIM 轨迹回放下的策略比较**：自车由策略控制，背景车按记录轨迹开环回放，不会根据自车行为改变路线或避让。结果不能解释为完整闭环交通中的交互安全表现。

## 固定实验对象

清单：[`selected.json`](../datasets/experiments/ngsim-comparison-v3/selected.json)；机器可读配置：[`ngsim_comparison_v3.json`](../configs/datasets/ngsim_comparison_v3.json)。配置哈希为 `cc2b9cb5dd16afd189121cdf0df83b5f625c6931d42b48e65a7cade126f65c7d`，成员清单哈希为 `1c683c840b16826e3c95760317b183b4d85d45ec78fd7f7578631ccc392b335e`。成员清单逐片段保留来源版本、选择配置/代码、官方源文件哈希、回放片段哈希和采集记录块 ID。

| 划分 | 数量 | 用途 | 采集记录 |
| --- | ---: | --- | --- |
| train | 258 | 四算法策略更新 | I-80 一个记录块 |
| validation | 400 | checkpoint/模型选择 | I-80 两个不同记录块 |
| test_ood | 400 | 最终一次性评估 | US-101 三个记录块 |

每个划分中四类各 100 条，但训练清单本身受 v1/v2 物化数量限制：跟驰 100、自车换道 100、切入 38、其他主线 20。为避免训练采样概率进一步受此不平衡影响，训练时先对四类均匀抽样，再在该类内均匀有放回抽片段。相同随机种子使算法在**回合序号**上获得相同的场景类别/片段抽样序列；因策略导致回合长度不同，场景不会保证在每一个环境步上严格同步。

数据仍是从 NGSIM 两个版本各自预先筛选后合并，不代表对所有高速交通的随机抽样。train 和 validation 是 I-80 不同采集记录，test 是 US-101，故测试是跨地点 OOD；各记录时段数量少，区间推断范围有限。

## 公平训练与模型选择

正式默认参数为每个算法、每个种子 100,000 个环境步；种子为 0–4。所有模型共用同一清单、动作/观测、奖励与 cost 定义、环境终止规则、训练步数、随机种子列表和其余公共设置。四种算法自身的网络/优化器和安全约束实现按本地 `rl/` 算法定义，不强行抹平算法特有参数。若改变公共设置，应把修改写入单一 JSON，通过同一个 `--settings` 文件传给所有算法，并生成新的 protocol plan。

每 25,000 步保存 checkpoint。选模时对固定验证子集评估：每个类别 10 条，共 40 条，成员按 SHA256 排序冻结在清单内。先要求平均**折扣 episode cost**不超过训练配置的 `cost_limit`；可行 checkpoint 按碰撞率、越界率、折扣 cost CVaR 依次最小化，再按成功率和路线完成度最大化。若无 checkpoint 满足 cost limit，则先最小化验证平均折扣 cost，再最小化碰撞率、最大化成功率。选择规则在读取测试结果前固定。随后，对选中 checkpoint 的完整 validation 400 条逐片段评估一次；test_ood 的 400 条也逐片段只评估一次。测试集不能参与 checkpoint、超参数或阈值选择。

训练按回合序号使用共同随机数场景序列，而策略终止行为不同会使总回合数和步数上的场景进度有所不同。比较表须同时记录种子及训练步数，不能把算法间每一步理解成完全配对的相同状态。

## 结果指标和样本相关性

逐回合原始记录包含 clip ID、场景类、采集记录块、成功、碰撞、越界、episode cost、折扣 cost、路线完成度、长度及速度。汇总至少报告成功率、碰撞率、越界率、平均未折扣 episode cost、平均折扣 episode cost、平均路线完成度和折扣 cost 的经验 CVaR（默认尾部比例 10%），并同时给出四类分层结果。CVaR 的有效尾部样本数必须注明；总 episode 数少时经验尾部估计尤其不稳定。

不同片段可能共享同一采集时段中的背景交通，故置信区间以采集记录块为单位进行 cluster bootstrap，不能按帧或片段假设 IID。当前 train/validation/test 分别只有 1/2/3 个记录块，因此 bootstrap 区间只作描述性结果，记录块数量不足以支持精确的总体置信结论。还应报告各训练种子的结果和离散程度；种子不是采集记录独立样本的替代物。

## 运行

先 dry-run 核对清单、训练设置及评估数量，不会启动训练：

```bash
python scripts/run_ngsim_comparison.py
```

正式运行默认四算法 × 五种子、每次 100k 步，checkpoint 间隔 25k：

```bash
python scripts/run_ngsim_comparison.py --execute
```

如需采用明确的统一设置文件：

```bash
python scripts/run_ngsim_comparison.py --settings configs/your_common_settings.json --execute
```

模型写入 `models/ngsim-comparison-v3/`，验证/测试结果写入 `results/ngsim-comparison-v3/`。`comparison_results.json/csv` 汇总选中模型结果；每个算法/种子的目录保存验证子集选模记录、完整验证、一次性测试及逐片段 CSV/JSONL。不要将网页单片段演示或随机开发运行混作正式比较结果。

这套结果的准确称谓是“固定 NGSIM 轨迹回放下的策略比较”。只有在背景车能够对自车动作作出反应、碰撞后状态与交互行为都进入闭环后，才适合进一步主张完整闭环交通中的交互安全表现。
