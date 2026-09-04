# Release notes

## `release/multiscale-uno-fm`

本分支是从已完成三阵列实验的干净训练分支整理出的聚焦发布版。

### 保留

- 6.7 GHz、0° 单波束 Multiscale UNO-FM 的统一训练/验证/测试入口；
- 数据 contract、manifest builder、有效区域 mask 与归一化；
- 多尺度条件编码、CA/SA、FNO/UNO 速度网络；
- AdamW、AMP、梯度累积、EMA、早停、严格 checkpoint 恢复；
- 固定哈希噪声、CFG 与 2-step Euler；
- 指标、逐样本 artifact、运行时间和可视化生成逻辑；
- 冻结 schema、架构图、复现说明和固定协议结果。

### 移除

- `tests/`、`pytest.ini` 和测试输出；
- 历史训练/评估入口与服务器绝对路径脚本；
- 稀疏 Task 2、跨频率、common8 多波束等其他实验分支；
- checkpoint、结果目录、数据集、缓存与旧报告图片。

### 低风险整理

`ModelEMA` 从旧总入口 `train.py` 原样迁移到 `training/ema.py`。两处训练器只更改
导入位置，更新公式和 state-dict 行为不变；这样发布代码不再为了 EMA 加载旧 ODE、
旧数据 loader 与旧 CLI。

除上述依赖隔离外，模型结构、随机种子、数据顺序、精度、训练超参数、损失与采样
公式均保持不变。
