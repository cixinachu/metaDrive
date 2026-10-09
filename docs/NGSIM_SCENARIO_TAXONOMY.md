# NGSIM 页面类别与数据说明

网页只保留当前已有实测片段的四类：主线普通行驶、跟驰、邻车切入、自车换道；按 US-101 与 I-80 分开筛选。类别由每个片段的 `event_class` 映射，不再把所有 NGSIM 片段都显示为主线。分类定义见 [taxonomy JSON](../configs/datasets/ngsim_scenario_taxonomy.json)。

本地已有 NGSIM 官方完整轨迹表，筛选记录在 `datasets/experiments/ngsim-mainline-v1/` 和 `ngsim-mainline-v2/`。正式比较将两版冻结为 `datasets/experiments/ngsim-comparison-v3/`：共 1,058 条 20 秒片段，train 258、validation 400、test_ood 400。训练是 I-80，验证是 I-80 的不同记录时段，测试为 US-101 跨地点测试；这是有意的 OOD 划分，不能把测试成绩解读成同地点 IID 泛化。四算法的公平比较规则见 [比较协议 v3](NGSIM_COMPARISON_PROTOCOL_V3.md)。

四类采用现有筛选程序定义：

- **跟驰**：持续同车道前车关系达到协议阈值（当前 15 秒）。
- **邻车切入**：周边车辆稳定换入自车车道，且前车关系发生变化。
- **自车换道**：自车稳定车道编号变化；单个窗口保留换道附近的轨迹。
- **主线普通行驶**：不符合以上三类的合格高速主线窗口。

所有样本均为 20 秒、10 Hz。筛选时按车辆类别、轨迹连续性、车道、车辆尺寸、速度/横向跳变和初始重叠做质量检查；不按 TTC 或算法表现挑样本。数据处理和采样细节见 [NGSIM 实验协议](NGSIM_EXPERIMENT_PROTOCOL.md)、[v1 筛选程序](../scripts/select_ngsim.py) 和 [v2 扩充程序](../scripts/select_ngsim_v2.py)。NGSIM 位置经过直路坐标近似，背景车按记录轨迹开环回放，不会对自车策略作反应，所以它适合比较固定轨迹条件下的控制表现，不能等价于完整闭环交通流实验。

页面暂不展示城市路口、匝道汇入/驶出等类别：虽然完整 NGSIM 表包含 Lankershim、Peachtree 和匝道相关记录，但本项目还没有与之匹配、经过验证的 MetaDrive 道路网络及闭环/回放接口。I-80/US-101 的官方资料确认了其匝道和交织研究用途；这些道路事实本身不代表当前网页已有相应可运行片段。[FHWA I-80](https://www.fhwa.dot.gov/publications/research/operations/06137/)；[FHWA US-101](https://www.fhwa.dot.gov/publications/research/operations/07030/)
