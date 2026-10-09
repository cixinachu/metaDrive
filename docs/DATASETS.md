# HighD / NGSIM 数据接入

支持高速公路主线的真实轨迹驱动环境：自车从记录的位置、速度和尺寸初始化，随后由 RL 算法控制；背景车按时间戳回放，不会针对自车动作主动避让。不是离线 RL，也不是反应式交通模型。

## NGSIM：完整原始表与筛选实验集

已下载 [USDOT / FHWA 官方轨迹表](https://data.transportation.gov/resource/8ect-6jqj.json) 的全部 **11,850,526 行**，原始合并 CSV 为 2,096,223,291 字节。

| 地点 | 官方原始行数 | 本环境用途 |
| --- | ---: | --- |
| I-80 | 4,566,387 | 主线训练、验证 |
| US-101 | 4,802,933 | 跨地点主线测试 |
| Lankershim | 1,607,319 | 保留原始数据；当前不重建城市路口 |
| Peachtree | 873,887 | 保留原始数据；当前不重建城市路口 |

“完整”指该官方轨迹表，不包括全部视频/GIS 附件；表中含重复记录，行数不是车辆数或独立样本数。原始数据放在 `datasets/raw/ngsim/full/`，下载清单、分页 SHA256 和合并文件 SHA256 可追溯；处理后的片段在 `datasets/processed/`。

**筛选规则、论文对照、划分依据与数据处理背景见 [NGSIM 实验协议 v1](NGSIM_EXPERIMENT_PROTOCOL.md)。** 正式多算法比较使用冻结合并的 [v3 比较协议](NGSIM_COMPARISON_PROTOCOL_V3.md)：v1 选中训练 110、验证 120、测试 120 个片段；v2 新增 708 条非重叠窗口；v3 统一 1,058 条，分为 train 258、validation 400、test_ood 400。逐片段来源、哈希和划分保存在 `datasets/experiments/ngsim-comparison-v3/selected.json`。旧的两个 20 秒接入演示片段不属于正式实验集。

## 网页与批量实验

1. 网页切换训练或评估，再选择 NGSIM。
2. 选择片段；名称显示 `train`、`validation` 或 `test_ood` 与场景类别。训练页只显示 train，评估页显示验证及测试片段。
3. 片段决定车道数、道路宽度、背景交通和最大 200 控制步，交通密度不会额外生成随机车辆。
4. 选择第一人称、三维跟车、鸟瞰或二维俯视后预览/运行。四个本地算法共享接口，下载按钮继续导出曲线、日志和实验文件。
5. 一个选中的网页片段只代表一个场景，评估仅运行一次。正式四算法公平比较使用 `scripts/run_ngsim_comparison.py`，统一场景清单、种子、步数、验证集选模和测试集一次性评估，详见 v3 协议；批量入口不生成实时画面。

训练模型保存在 `models/`，评估保存在 `results/`。test_ood 是跨地点、跨车道数评估，不是 IID 测试；由于背景车辆开环回放，结果应称为“固定 NGSIM 轨迹回放下的策略比较”，不能据此主张闭环交通交互安全。

## HighD 与本地导入

HighD 官方要求姓名、机构和研究用途，人工审核后授予访问权限。当前没有下载真实 HighD 数据，未代填身份或提交申请。已实现本地导入器并通过标注为合成的数据测试，不等于真实 HighD 数据验收。
[官方访问申请](https://levelxdata.com/highd-dataset/)、[官方格式说明](https://levelxdata.com/wp-content/uploads/2023/10/highD-Format.pdf)。

网页的“导入 HighD / NGSIM 本地文件”也可接入独立本地记录。HighD 输入 `XX_tracks.csv`，同目录必须有 `XX_tracksMeta.csv` 和 `XX_recordingMeta.csv`。NGSIM 输入标准 CSV 或无表头 18 列 TXT，并明确主线车道数、车道宽度和采集组。

```bash
python scripts/import_trajectories.py --source highd --path /path/to/01_tracks.csv --ego-id 1 --split train
python scripts/download_ngsim.py   # 现在是完整下载入口，可续传
python scripts/select_ngsim.py   # v1 固定样本
python scripts/select_ngsim_v2.py --config configs/datasets/ngsim_selection_v2.json  # v2 非重叠扩充
```

手动导入器禁止同一采集组/原始文件跨 train/eval。完整 NGSIM 筛选器采用官方记录标识和预设记录划分，因此可以从同一个原始总表生成不同划分；不能在手动导入时只换一个采集组名称规避隔离。

## 几何和动力学边界

- HighD 按帧率计算时间，包围盒左上角转车中心；width 为车长、height 为车宽；按 drivingDirection 翻转坐标，只保留同向交通；从标线近似等宽化道路，单车道宽度偏差不超过 0.3m。
- NGSIM 英尺换米、毫秒换秒；Local_Y 车头位置减半车长，Local_X 作为横向位置。只重建高速主线，不恢复真实匝道、曲线、背景影像和所有道路边缘。
- 地图为单向等宽直道，由连续 50m 路段构成，长度上限 900m。背景车使用尺寸一致的包围盒模型，外观不对应原车品牌。
- 保留实测位置，不隐式平滑；回放速度由位置差分得到。物理步 0.02s，RL 控制间隔 0.1s；自车保持 259 维观测及原有 reward/cost 定义。
- 自车由策略控制，不回放。目标为到达记录自车在片段末端的纵向位置且不越界；原轨迹位移不足 5m 时不设置到达成功。回合到窗口末尾截断。
- 背景车在自车偏离原轨迹后仍按记录行驶，也不会因碰撞停止，故可能产生记录中不存在的冲突。这是固定背景轨迹下的策略比较，不能单独代表真实交通闭环安全性。

网页任务保存 `dataset_manifest.json` 与 `dataset_clip.json`，批量任务保存 `dataset_collection.json` 和成员片段校验值。历史接入验收 `outputs/dataset_acceptance.json` 保留；完整数据筛选后的验收另行记录，不能把旧演示验收当作新实验集性能结果。

本次验收：`outputs/ngsim_full_acceptance.json`（350 个片段及 12 组回放）、`outputs/ngsim_algorithm_acceptance.json`（四算法多片段更新/保存/加载）、`outputs/ngsim_cli_acceptance.json`（批量命令端到端）、`outputs/ngsim_ui_acceptance.json`（网页筛选与真实预览）。这些是接口验收，正式算法性能对比尚未运行。
