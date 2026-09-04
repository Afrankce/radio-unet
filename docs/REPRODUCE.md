# 复现实验

## 固定协议

- 频率：6.7 GHz；
- 波束方向：0° 单波束；
- 阵列：8x8、16x16、32x32；
- 场景划分：seed 42，train/val/test = 560/80/160，场景互斥；
- 分辨率：256x256；
- 条件：Tx mask、height、beam map；
- 优化器：AdamW，learning rate `1e-3`，weight decay `1e-5`；
- 最多 1000 epochs，warmup ratio 0.1，early-stopping patience 20；
- micro-batch 2，accumulation 28，effective batch 56；
- EMA decay 0.999；AMP FP16；
- CFG dropout 0.25，评估 CFG 1.0；
- 固定哈希初始噪声，2-step Euler。

所有关键值都由不可变配置类校验。不要为了“跑通”而直接修改锁定常量，否则已有
checkpoint 的身份哈希将失效。

## 目录约定

```text
/data/multiconfig/                  # dataset root
/data/manifests/
  scene_split_seed42.json
  height_stats_train.json
  manifest_samefreq_6.7ghz_8x8_0deg.jsonl
  manifest_samefreq_6.7ghz_16x16_0deg.jsonl
  manifest_samefreq_6.7ghz_32x32_0deg.jsonl
/runs/uno/{8x8,16x16,32x32}/        # config.json, best.pt, last.pt, metrics.csv
/results/uno/{8x8,16x16,32x32}/     # final metrics, predictions, figures
```

数据目录名可以改变，但 manifest 内的相对路径必须仍然与 dataset root 对齐。

## 1. 创建环境

```bash
git clone https://github.com/Hxxxz0/RadioFlow.git
cd RadioFlow
git switch release/multiscale-uno-fm

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 先安装匹配驱动/CUDA 的 PyTorch
python -m pip install -r requirements.txt
```

参考训练环境使用 PyTorch 2.5.1 + CUDA 12.1；并不要求其他机器完全使用相同驱动，
但论文级严格复现应记录 Python、PyTorch、CUDA、GPU 型号和当前 Git commit。

## 2. 构造三个单波束 manifest

确保 `scene_split_seed42.json` 已位于 manifest 目录，并对每种阵列执行：

```bash
python prepare_same_frequency.py build-manifest \
  --dataset-root /data/multiconfig \
  --split-path /data/manifests/scene_split_seed42.json \
  --array-size 8x8 \
  --frequency-hz 6700000000 \
  --steering-deg 0 \
  --output /data/manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl
```

把两处 `8x8` 分别替换为 `16x16`、`32x32`。构造器会依据冻结 schema 选择各阵列
真正的 0° beam ID，并验证 800 个样本及固定 split。

`height_stats_train.json` 必须只由相同 split 中的 560 个训练场景生成，并包含每个
height 文件的哈希证据；训练预检会验证其 schema、场景数和 split SHA-256。

## 3. 预检

```bash
python run_same_frequency_multiscale_uno.py train \
  --dataset-root /data/multiconfig \
  --manifest-path /data/manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl \
  --height-stats-path /data/manifests/height_stats_train.json \
  --run-root /runs/uno/8x8 \
  --array-size 8x8 \
  --device cuda:0 \
  --resume none \
  --preflight-only
```

预检会读取每个 split 的首个样本、校验 `[3,256,256]` 条件、`[1,256,256]` 目标、
非空 valid mask、数据版本、manifest 哈希、Git ancestry 和 origin。

## 4. 训练/续训

首次训练：

```bash
python run_same_frequency_multiscale_uno.py train \
  --dataset-root /data/multiconfig \
  --manifest-path /data/manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl \
  --height-stats-path /data/manifests/height_stats_train.json \
  --run-root /runs/uno/8x8 \
  --array-size 8x8 \
  --device cuda:0 \
  --resume none
```

断点续训只把最后一项改为：

```bash
--resume auto
```

`last.pt` 保存模型、EMA、optimizer、scheduler、AMP scaler、trainer state 和训练数据
generator 状态。恢复时做严格 identity 校验，避免把其他阵列或其他配置的 checkpoint
误接入当前实验。

三种阵列可分别占用一张 GPU 并行训练：

```bash
DATASET_ROOT=/data/multiconfig \
MANIFEST_DIR=/data/manifests \
HEIGHT_STATS=/data/manifests/height_stats_train.json \
RUN_ROOT=/runs/uno \
RESULTS_ROOT=/results/uno \
bash scripts/run_three_arrays.sh train
```

可用 `GPU_8X8`、`GPU_16X16`、`GPU_32X32` 环境变量覆盖默认的 `0/1/2`。

## 5. 验证集冻结与测试集评估

对每个阵列必须先运行 `select-cfg`，它在 80 个 validation 场景上校验锁定候选
`CFG=1.0`，并将选择与 `best.pt` 的 epoch/identity 绑定。之后运行 `test`，一次性评估
160 个 test 场景。

```bash
bash scripts/run_three_arrays.sh evaluate
```

最终目录包含：

- `metrics_overall.json` 与按频率汇总的 CSV；
- 每个测试样本的预测 artifact；
- 固定场景的对比图与误差图；
- runtime benchmark；
- `run_manifest.json` 和 CFG/checkpoint 哈希证据。

正式结果目录采用“已存在即拒绝覆盖”的策略。需要重跑时请使用新的结果目录，保留旧
结果作为不可变实验记录。

## 6. 指标口径

- dB-RMSE、dB-MAE：只对 `valid_mask` 为真的传播像素统计；
- NMSE：全数据集累计 squared error 除以累计 target energy；
- PSNR：在归一化 `[0,1]` 域计算；
- SSIM：标准局部窗口实现，并按有效区域口径累计；
- 所有 test 指标均使用 EMA `best.pt`、固定噪声、CFG=1.0 和 2-step Euler。
