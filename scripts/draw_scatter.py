import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 数据准备 (来自 Table 1: Llama-3.1-8B)
# ==========================================
# X轴: Constraint/Safety (IFEval Score) -> 越右越好
# Y轴: Reasoning/Alignment (CSQA Accuracy) -> 越上越好

data = {
    "Baseline": {"x": 0.305, "y": 0.740, "color": "#333333", "marker": "*", "size": 300, "label": "Baseline"},
    "Random_index": {"x": 0.306, "y": 0.743, "color": "#999999", "marker": "o", "size": 100, "label": "Random"},
    "Smooth_abs": {"x": 0.312, "y": 0.735, "color": "#2ca02c", "marker": "s", "size": 120, "label": "Smooth_abs"},
    "Abs_select": {"x": 0.321, "y": 0.736, "color": "#1f77b4", "marker": "^", "size": 120, "label": "Abs_select"},
    "Grad_dir": {"x": 0.223, "y": 0.784, "color": "#d62728", "marker": "D", "size": 120, "label": "Grad_dir"},
}

# ==========================================
# 2. 绘图设置
# ==========================================
plt.figure(figsize=(8, 6), dpi=150)
ax = plt.gca()

# 绘制中心参考线 (以 Baseline 为原点)
base_x = data["Baseline"]["x"]
base_y = data["Baseline"]["y"]
plt.axvline(x=base_x, color='gray', linestyle='--', alpha=0.5, linewidth=1)
plt.axhline(y=base_y, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# ==========================================
# 3. 绘制散点与箭头
# ==========================================
for name, point in data.items():
    # 绘制点
    plt.scatter(point["x"], point["y"],
                color=point["color"],
                marker=point["marker"],
                s=point["size"],
                label=point["label"],
                edgecolors='white',
                zorder=10)

    # 绘制从 Baseline 指向该点的箭头 (Baseline自己除外)
    if name != "Baseline":
        plt.annotate("",
                     xy=(point["x"], point["y"]),
                     xytext=(base_x, base_y),
                     arrowprops=dict(arrowstyle="->", color=point["color"], alpha=0.6, lw=1.5))

    # 添加文字标签 (带一点偏移量，避免遮挡)
    offset_x = 0.005
    offset_y = 0.002

    # 特殊处理文字位置以防重叠
    ha = 'left'
    if name == "Grad_dir":
        offset_x = -0.005  # 文字放左边
        ha = 'right'
    elif name == "Baseline":
        offset_y = -0.005  # 文字放下面

    plt.text(point["x"] + offset_x, point["y"] + offset_y, name,
             fontsize=9, fontweight='bold', color=point["color"], ha=ha, zorder=15)

# ==========================================
# 4. 装饰与标注 (体现分析深度)
# ==========================================

# 区域标注 (象限含义)
plt.text(0.24, 0.79, "Trade-off Region\n(High Reward, High Risk)",
         fontsize=10, color='#d62728', alpha=0.8, ha='center',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.text(0.32, 0.73, "Conservative Region\n(Safe but Lower Gain)",
         fontsize=9, color='#1f77b4', alpha=0.8, ha='left')

# 坐标轴标签
plt.xlabel("Constraint Adherence (IFEval Score) $\\rightarrow$", fontsize=12, fontweight='bold')
plt.ylabel("Reasoning Alignment (CSQA Accuracy) $\\rightarrow$", fontsize=12, fontweight='bold')
plt.title("Performance Trade-off: Alignment vs. Constraints (Llama-3.1)", fontsize=14, pad=15)

# 设置坐标范围 (留出余地)
plt.xlim(0.20, 0.34)
plt.ylim(0.72, 0.80)

# 网格与美化
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# 显示图表
plt.show()