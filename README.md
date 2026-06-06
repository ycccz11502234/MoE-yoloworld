# MoE-YOLOWorld

基于 [YOLO-World](https://docs.ultralytics.com/models/yolo-world) 的开放词汇目标检测扩展实现：把 backbone 中的 C2f 模块替换为 **MoE-C2f**（混合专家版本的 C2f），保留原 RepVL-PAN 文本融合 head，得到 **MoE-YOLOWorld**。

模型整体接口、训练流程、数据格式与上游 Ultralytics YOLO-World 完全一致，只是新增了一个 `YOLOWorld_MoE` 入口和对应的 backbone 配置。

> 上游参考：<https://docs.ultralytics.com/models/yolo-world>

---

## 与原版的差异

| 模块 | 原版 YOLO-World | 本仓库 MoE-YOLOWorld |
|------|----------------|---------------------|
| backbone 第 2/4/6/8 层 | `C2f` | `MoE_C2f_final` 或 `MoE_C2f_final_conv` |
| Bottleneck 第二个 3×3 卷积 | 单一 conv | 1 共享专家 + N 个路由专家（Top-K 路由） |
| 路由门控 | — | 跨模态 Max-Sigmoid 门控 + loss-free bias |
| 负载均衡 | — | 局部熵 + 全局熵多层级 loss |
| head | RepVL-PAN（C2fAttn） | RepVL-PAN（C2fAttn），保持不变 |
| Detect | `WorldDetect` | `WorldDetect`，保持不变 |

两个 MoE 变体：

- `MoE_C2f_final`：专家是 **DWConv**，从随机初始化训练。
- `MoE_C2f_final_conv`：专家是标准 **3×3 Conv**，可由 `yolov8x-worldv2.pt` 通过"近复制 + 高斯微扰"方式初始化（论文 §3.7.1 / 公式 3.29）。

默认实验配置：`num_routing=8`，`top_k=2`，`text_dim=512`。

---

## 目录结构

```
yoloworld/
├── ultralytics/                       # 修改版 ultralytics（新增 world_moe 模型族）
│   └── models/yolo/world_moe/         # MoE 训练 / 验证 / 预测
├── yolov8-worldv2.yaml                # 上游原版 backbone
├── yolov8-worldv2-moe-final.yaml      # MoE 变体（DWConv 专家）
├── yolov8-worldv2-moe-final-conv.yaml # MoE 变体（标准 Conv 专家，可继承预训练）
├── init_moe_yoloworld.py              # 从 yolov8x-worldv2.pt 初始化 MoE 权重
├── train_moe_yolo_world.py            # 训练脚本（O365 + GoldG）
├── predict.py                         # 推理示例
├── expert_selection_count.py          # 专家选择频率/概率可视化
├── download_lvis.py                   # LVIS 数据集下载
├── download_objects365.py             # Objects365 数据集下载
├── weights/                           # 模型权重
│   ├── yolov8x-worldv2.pt             # 官方预训练
│   ├── MoE_final_conv_init_x.pt       # 由 init 脚本生成的 MoE 初始权重
│   └── MoE_yoloworld.pt               # 训练好的 MoE 权重
└── test_image/                        # 推理示例图片
```

---

## 环境

依赖与上游 ultralytics 一致：

```bash
pip install -r ultralytics/requirements.txt   # 或单独装
pip install ultralytics torch torchvision pyyaml matplotlib
```

仓库内已包含改造过的 `ultralytics/`，运行脚本时直接以本目录为工作目录即可，无需再 `pip install ultralytics`。

---

## 快速开始

### 推理

```bash
python predict.py
```

或在代码里：

```python
from ultralytics import YOLOWorld_MoE

model = YOLOWorld_MoE("weights/MoE_yoloworld.pt")
model.set_classes(["bus", "person"])

results = model.predict(
    "test_image/000000002006.jpg",
    save=True,
    conf=0.25,
    iou=0.7,
    project="runs/predict",
    name="demo",
    exist_ok=True,
)
```

`set_classes(...)` 决定开放词汇下的类别集合，可任意替换成自定义文本。

### 从预训练初始化 MoE 权重

`MoE_C2f_final_conv` 的专家结构与原 Bottleneck 第二个 3×3 卷积形状一致，可以从 `yolov8x-worldv2.pt` 直接继承：

```bash
python init_moe_yoloworld.py \
  --src weights/yolov8x-worldv2.pt \
  --cfg yolov8x-worldv2-moe-final-conv.yaml \
  --dst weights/MoE_final_conv_init_x.pt \
  --sigma 0.01 --seed 0
```

参数：

- `--sigma`：路由专家的高斯微扰标准差，越大初始多样性越强。
- `--seed`：扰动随机种子。

输出 `MoE_final_conv_init_x.pt` 可直接用作 `model.train(...)` 的初始权重。

### 训练

```bash
python train_moe_yolo_world.py
```

脚本里默认是 O365 + Flickr30k + GQA 联合训练、LVIS 验证：

```python
data = dict(
    train=dict(
        yolo_data=["Objects365.yaml"],
        grounding_data=[
            dict(img_path="datasets/flickr30k/images",
                 json_file="datasets/flickr30k/final_flickr_separateGT_train.json"),
            dict(img_path="datasets/GQA/images",
                 json_file="datasets/GQA/final_mixed_train_no_coco.json"),
        ],
    ),
    val=dict(yolo_data=["lvis.yaml"]),
)

model = YOLOWorld("yolov8-worldv2-moe-final.yaml")
model.train(data=data, epochs=100, batch=64,
            trainer=WorldTrainerFromScratch,
            project="runs/moe_yoloworld", name="moe_yoloworld")
```

需要先准备好 Objects365 / Flickr30k / GQA / LVIS 数据集，可用 `download_objects365.py` 和 `download_lvis.py` 辅助下载。

### 专家激活分析

统计每个 MoE 层在验证集上的专家选择频率/平均概率：

```bash
python expert_selection_count.py
```

会按层（默认 backbone 第 2/4/6/8 层）输出：

- `expert_freq_layer_{i}.png`：专家被 Top-K 选中的次数直方图。
- `average_prob_layer_{i}.png`：专家平均路由概率。

记得修改脚本顶部的 `image_folder` 指向自己的验证集目录。

---

## 配置说明（YAML）

`yolov8-worldv2-moe-final-conv.yaml` 中 MoE 模块参数：

```yaml
- [-1, 6, MoE_C2f_final_conv, [256, True, 8, 2, 512]]
#                              c2  shortcut Nr Tk text_dim
```

- `c2`：输出通道。
- `shortcut`：是否启用 Bottleneck 的残差连接。
- `Nr`：路由专家数量。
- `Tk`：Top-K 路由的 K。
- `text_dim`：文本特征维度（与 head/Detect 对齐，YOLO-World 默认 512）。

`scales` 与 nc 用法和上游一致；切换 `n/s/m/l/x` 即可改尺度，`nc` 一般跟着数据集走（脚本里会用文本类别覆盖）。

---

## 已知限制

- Objects365 / GoldG / LVIS 都比较大，训练对磁盘和带宽有要求。
- 当前 MoE 路由产生的负载均衡 loss 是写在 `v8_world_DetectionLoss` 里的，需在自定义 trainer 里 forward 时一起累加。
- 多卡训练沿用 ultralytics 的 DDP，没有针对 MoE 做 expert-parallel 优化。

---

## 致谢

- [YOLO-World](https://github.com/AILab-CVC/YOLO-World)：开放词汇检测原方法。
- [Ultralytics](https://github.com/ultralytics/ultralytics)：YOLOv8 / YOLO-World 工程实现，本仓库 backbone & 训练流程基于其改造。
- 论文：MoE-YOLOWorld（北京理工大学本科毕设，2026）。

---

## License

继承上游 Ultralytics AGPL-3.0。使用本仓库代码请同步遵守 AGPL-3.0 条款。
