# NGSIM 数据来源与许可

本仓库中的 NGSIM 原始数据来自美国交通部 Federal Highway Administration 发布的 Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data。

建议引用：U.S. Department of Transportation Federal Highway Administration. (2016). Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data. [Dataset]. Provided by ITS DataHub through Data.transportation.gov. https://doi.org/10.21949/1504477 。实际下载信息见各数据清单。

- 官方数据目录：https://catalog.data.gov/dataset/next-generation-simulation-ngsim-vehicle-trajectories-and-supporting-data
- 官方数据接口：https://data.transportation.gov/resource/8ect-6jqj.json
- 官方目录标注的许可：Creative Commons Attribution ShareAlike 3.0 Unported (CC BY-SA 3.0)。
- 许可全文：https://creativecommons.org/licenses/by-sa/3.0/legalcode
- 核对日期：2026 年 10 月 9 日。

`raw/ngsim/` 保存下载的数据及分页、合并和校验记录。`processed/` 中的 NGSIM 片段经过窗口筛选、单位转换、坐标处理和训练验证测试划分；它们不是官方原始发布文件。处理方式见 `../docs/DATASETS.md` 及实验协议，来源与校验信息保存在相应清单中。

本仓库中的 NGSIM 数据及其派生片段按上述 CC BY-SA 3.0 许可提供。再分发或改编时须保留来源署名、许可链接并说明修改，对改编数据遵守相同方式共享要求。本说明仅适用于 NGSIM 数据，不为项目源码或其他第三方内容另行授予许可。FHWA 不为本项目或实验结论背书。

本次归档未包含真实 HighD 数据。
