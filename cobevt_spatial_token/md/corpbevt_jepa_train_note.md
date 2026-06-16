# CorpBEVT-JEPA 实现与训练记录

## 已实现内容

- 新增主模型：`opv2v/opencood/models/corpbevt_jepa.py`
  - `core_method: corpbevt_jepa`
  - 类名：`CorpBEVTJEPA`
  - 主流程：VJEPA2 spatial tokens -> BEV refiner -> optional compressor -> regroup -> STTF -> SwapFusionEncoder -> NaiveDecoder -> BevSegHead。

- 新增 VJEPA2 spatial token adapter：`opv2v/opencood/models/sub_modules/jepa_spatial_encoder.py`
  - 通过 `drivor_root` 复用 DrivoR 包中的 `ImgEncoderVJEPA2`。
  - 支持 `frozen/lora/scene_lora`。
  - `scene_lora` 只用于 attention prefix；最终进入 BEV 的特征固定为 `spatial_tokens = tokens[:, S:]`。
  - 输入帧顺序与 Drive-JEPA 对齐：`[current, previous]`。

- 新增 BEV refiner：`opv2v/opencood/models/sub_modules/jepa_bev_refiner.py`
  - `proposal_query` 固定为 `False`。
  - 输出 local BEV bottleneck 为 `[N, 128, 32, 32]`，与原 `corpbevt` 的 FAX 输出对齐。

- 新增训练配置：`opv2v/opencood/hypes_yaml/opcamera/corpbevt_jepa.yaml`
  - `fusion.args.queue_length: 2`
  - image preprocess: `512 x 512`
  - VJEPA2 resolution: `512 x 512`
  - VJEPA2 spatial map: `32 x 32`
  - final BEV after decoder: `256 x 256`

## 训练启动命令

在 `code/CoBEVT/opv2v` 下执行：

```bash
/home/dataset-assist-0/yinhongbo/miniconda3/envs/cobevt/bin/python \
  opencood/tools/train_camera.py \
  --hypes_yaml opencood/hypes_yaml/opcamera/corpbevt_jepa.yaml
```

当前训练目录：

```text
/home/dataset-local/yinhongbo/code/CoBEVT/opv2v/opencood/logs/corpbevt_jepa_2026_05_13_14_00_26
```

## 训练中修复的问题

- `queue_length=2` 时 `retrieve_base_data()` 返回 list，`BaseCameraDataset.get_sample_random/get_sample()` 已补充逐帧 `get_data_sample()` 兼容。
- DrivoR `pylogger` 依赖 `pytorch_lightning`，在 `jepa_spatial_encoder.py` 中加了轻量 logger stub，避免额外安装 Lightning。
- DrivoR `GridMask` 对非连续 tensor 使用 `.view()` 报错，adapter 调用前补 `.contiguous()`。
- `VanillaSegLoss` 期望 GT 为 `[B, L, H, W]`，`CamIntermediateFusionDataset.collate_batch()` 已在缺少 `L` 维时补成 `[B, 1, H, W]`。
- `queue_length=2` 时不同 timestamp 的有效 CAV 数可能不同。当前实现以当前帧有效 CAV 列表为准；历史帧如果缺少该车辆或该车辆超出通信范围，则重复使用当前帧该车辆作为历史帧，并将 `prev_pose_offset` 置为 identity，保证 VJEPA2 双帧输入 shape 稳定。

## 如何监控

查看训练进程：

```bash
ps -ef | grep 'train_camera.py --hypes_yaml opencood/hypes_yaml/opcamera/corpbevt_jepa.yaml' | grep -v grep
```

启动 TensorBoard：

```bash
cd /home/dataset-local/yinhongbo/code/CoBEVT/opv2v
tensorboard --logdir opencood/logs/corpbevt_jepa_2026_05_13_14_00_26 --host 0.0.0.0 --port 6006
```

如果本机通过 SSH 访问服务器，可以做端口转发：

```bash
ssh -L 6006:127.0.0.1:6006 <server>
```

然后浏览器打开：

```text
http://127.0.0.1:6006
```

查看 checkpoint/event 文件：

```bash
find /home/dataset-local/yinhongbo/code/CoBEVT/opv2v/opencood/logs/corpbevt_jepa_2026_05_13_14_00_26 -maxdepth 1 -type f -ls
```
