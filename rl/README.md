# 本地 RL 算法

工作台的算法入口全部位于本目录：

| 算法 | 独立模块 | 类 |
|---|---|---|
| SAC | `sac.py` | `SAC` |
| SAC-Lagrangian | `sac_lagrangian.py` | `SACLagrangian` |
| WCSAC | `wcsac.py` | `WCSAC` |
| WCSAC-IQN | `wcsac_iqn.py` | `WCSACIQN` |

`algorithms.py` 是唯一注册表。界面的算法列表、说明、安全属性、训练创建和评估加载均读取它。
`common.py` 复用构造参数，`constrained_sac.py` 复用 replay、critic 和优化循环；
三个安全变体通过各自模块选择期望、高斯或 IQN critic。避免复制相同的 SAC 更新代码。
SAC 模块继承 SB3 SAC，不宣称从零重写 SAC。训练通过 Studio 工作台启动；场景 seed 适配位于 `vec_env.py`。
旧 `rl.constrained_sac` 导入名通过兼容层保留，历史 checkpoint 可继续加载。

## 新增算法

1. 添加 `rl/my_algorithm.py`，定义模型类和 `create_model(settings, env)` 工厂。
2. 在 `rl/algorithms.py` 的 `REGISTRY` 中添加：

```python
'my_algorithm': Algorithm(
    '显示名称', 'rl.my_algorithm', 'MyAlgorithm',
    '算法说明', uses_cost=True,
),
```

3. 重启工作台服务并刷新页面，算法自动出现在选择列表中，不需要在前端硬编码算法名。

工厂可使用自己的构造参数，不要求新算法必须继承 SAC。
目前工作台训练协议兼容 SB3：模型须提供 `set_logger`、`learn(..., callback, log_interval)`、
`save`、类方法 `load(..., device)`、`predict(..., deterministic)`、`num_timesteps`、`device`，
并使用 SB3 callback/logger 与 VecEnv 协议。不兼容的自研模型须在本地模块提供适配器。
若新算法引入页面现有字段之外的新参数，需要同步增加 `studio/config.py` 的默认值/验证和前端表单。
注册本身不会自动生成任意算法的专属参数表单。

训练模型与日志写入 `models/<run>/`；评估结果写入 `results/<run>/`。
算法 ID 是模型元数据的一部分，已有 ID 不要改名或重新绑定到不兼容算法。
