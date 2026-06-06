import os
import torch
from glob import glob
import matplotlib.pyplot as plt

# -------------------------------
# 配置参数
# -------------------------------
image_folder = "/home/wfx/yoloworld/datasets/lvis/images/val2017"  # 修改为你的图片文件夹
output_plot_path = "./expert_selection_analysis.png"    # 输出图表路径
device = "cuda:0" if torch.cuda.is_available() else "cpu"
top_k = 2


from ultralytics import YOLOWorld_MoE
import yaml

# 读取 YAML 文件
with open('./ultralytics/cfg/datasets/lvis.yaml', 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file)

# 获取 names 字典
names = data['names']

# 转成有序列表（按索引排序）
names_list = [names[i] for i in sorted(names.keys())]

# model = YOLOWorld("weights/base.pt") 

# model = YOLOWorld_MoE("weights/MoE_attn.pt") 

# Initialize a YOLO-World model
model = YOLOWorld_MoE("weights/MoE_gate.pt")  # or choose yolov8m/l-world.pt

# model.set_classes(["person",])

model.set_classes(names_list)
# -------------------------------
# 收集所有图片路径
# -------------------------------
img_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
img_paths = []
for fmt in img_formats:
    img_paths.extend(glob(os.path.join(image_folder, fmt)))
img_paths.sort()

print(f"找到 {len(img_paths)} 张图片")

# -------------------------------
# 存储结果
# -------------------------------
all_gate_logits = []
all_expert_indices = []

# -------------------------------
# 遍历每张图片
# -------------------------------
for embed_layer in [2,4,6,8]:
    all_gate_logits = []
    all_expert_indices = []
    for img_path in img_paths:
        result = model.predict(img_path, embed=[embed_layer], visualize=False)
        # 提取 gate logits: shape [1, num_experts]
        gate_logits = result[0]  # 因为返回的是 list[tensor]
        if gate_logits.dim() == 1:
            gate_logits = gate_logits.unsqueeze(0)
        topk_vals, topk_idx = torch.topk(gate_logits, top_k, dim=-1)

        # 存储结果
        all_gate_logits.append(gate_logits.cpu())
        all_expert_indices.append(topk_idx.cpu())
        selected_experts = topk_idx[0].tolist()  # e.g. [7, 4]
        print(f"处理: {os.path.basename(img_path)} -> 选择专家 {selected_experts}")


    # -------------------------------
    # 转换为 Tensor
    # -------------------------------
    if len(all_gate_logits) == 0:
        raise ValueError("没有成功处理任何图片，请检查路径或模型")

    gate_logits_tensor = torch.cat(all_gate_logits, dim=0)  # [N, num_experts]
    expert_indices_tensor = torch.cat(all_expert_indices, dim=0)  # [N, top_k]

    # 转为概率
    probs = torch.softmax(gate_logits_tensor, dim=-1)  # [N, num_experts]
    mean_probs = probs.mean(dim=0).numpy()  # 每个专家的平均选择概率

    # 统计每个专家被选中的次数
    num_experts = probs.shape[1]
    # 🔥 关键：展平所有选中的专家索引（因为每个样本选 top_k 个）
    all_selected_flat = expert_indices_tensor.view(-1)  # [N * top_k]
    hist_counts = torch.bincount(all_selected_flat, minlength=num_experts).numpy()


    # -------------------------------
    # 绘图：选择频率
    # -------------------------------
    plt.figure(figsize=(8, 6))
    colors = ['red' if c == max(hist_counts) else 'gray' for c in hist_counts]
    plt.bar(range(num_experts), hist_counts, color=colors, alpha=0.8)
    plt.title(f"Expert Selection Frequency (Layer {embed_layer})", fontsize=14)
    plt.xlabel("Expert Index", fontsize=12)
    plt.ylabel("Selection Count", fontsize=12)
    plt.xticks(range(num_experts))
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    freq_plot_path = f"expert_freq_layer_{embed_layer}.png"
    plt.savefig(freq_plot_path, dpi=150)
    plt.close()

    # -------------------------------
    # 绘图：平均概率
    # -------------------------------
    plt.figure(figsize=(8, 6))
    plt.bar(range(num_experts), mean_probs, color='salmon', alpha=0.8)
    plt.title(f"Average Selection Probability (Layer {embed_layer})", fontsize=14)
    plt.xlabel("Expert Index", fontsize=12)
    plt.ylabel("Average Probability", fontsize=12)
    plt.xticks(range(num_experts))
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    prob_plot_path = f"average_prob_layer_{embed_layer}.png"
    plt.savefig(prob_plot_path, dpi=150)
    plt.close()

    print(f"\n=== Layer {embed_layer} 分析完成 ===")
    print(f"平均选择概率: {mean_probs.round(3)}")
    print(f"选择次数统计: {hist_counts.tolist()}")
    print(f"频率图保存至: {freq_plot_path}")
    print(f"概率图保存至: {prob_plot_path}\n")