import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# =====================================================================
# 第一部分：极度严谨的数据构建 (Data Assembly)
# =====================================================================
print("⏳ 1/4 正在构建多尺度特征排名矩阵...")

# 将你提供的三个尺度 Top 10 特征整理成字典。
# 数字代表排名 (1 是最重要的第一归因)。如果在某个尺度跌出前 10，使用 np.nan 表示。
data = {
    'Feature': [
        'H3K4me3', 
        'gc_content_flank_50bp',
        'gc_content_flank_25bp',
        'gc_content_flank_100bp',
        'gc_content_flank_200bp',
        'H3K4me2',
        'log2_dist5prime_exonicTranscripts',
        'log2_dist5prime_promoters',
        'log2_dist3prime_promoters',
        'relativePOS_promoters',
        'relativePOS_exonicFivePrimeUTR',
        'meta_t_topology',
        'gc_content_exonicFivePrimeUTR'
    ],
    '21bp':  [1,  6,  8, 10, np.nan, np.nan, 3,  2, 4, 5,  7,      9,      np.nan],
    '51bp':  [3,  7,  6,  9, np.nan, np.nan, 1,  2, 4, 5,  np.nan, 8,      10],
    '101bp': [1,  3,  7,  8, 9,      10,     2,  4, 5, 6,  np.nan, np.nan, np.nan]
}

df = pd.DataFrame(data)
df.set_index('Feature', inplace=True)
scales = ['21bp', '51bp', '101bp']

# =====================================================================
# 第二部分：生物学逻辑驱动的色彩与视觉层级设计 (Visual Hierarchy)
# =====================================================================
print("\n🎨 2/4 正在配置生物学逻辑配色与图层权重...")

# 大牛视角的色彩规划：绝不能随机给颜色！必须按生物学意义分组
color_dict = {
    # 1. 绝对核心 (表观遗传) -> 鲜艳的警示红/橙
    'H3K4me3': '#e74c3c', 
    'H3K4me2': '#e67e22', 
    
    # 2. 远端结构约束 (侧翼 GC 含量) -> 渐变深邃蓝
    'gc_content_flank_50bp': '#2980b9', 
    'gc_content_flank_25bp': '#3498db',
    'gc_content_flank_100bp': '#5dade2',
    'gc_content_flank_200bp': '#85c1e9',
    
    # 3. 拓扑距离基准 (距离类特征) -> 统一使用沉稳的灰色，作为背景衬托
    'log2_dist5prime_exonicTranscripts': '#7f8c8d',
    'log2_dist5prime_promoters': '#95a5a6',
    'log2_dist3prime_promoters': '#bdc3c7',
    'relativePOS_promoters': '#bdc3c7',
    
    # 4. 局部/微观噪音 (跌出榜单的特征) -> 极浅的灰色虚线
    'relativePOS_exonicFivePrimeUTR': '#d5dbdb',
    'meta_t_topology': '#d5dbdb',
    'gc_content_exonicFivePrimeUTR': '#d5dbdb'
}

# =====================================================================
# 第三部分：绘制高精度 Bump Chart
# =====================================================================
print("\n🖌️ 3/4 正在渲染图表结构 (点、线、描边)...")

fig, ax = plt.subplots(figsize=(12, 9))

# 遍历每一个特征，画出它的生命周期轨迹
for feature in df.index:
    ranks = df.loc[feature].values
    
    # 判断该特征是否是我们的“主角”
    is_protagonist = feature in ['H3K4me3', 'gc_content_flank_50bp']
    lw = 4.0 if is_protagonist else 2.0
    alpha = 1.0 if is_protagonist else 0.7
    zorder = 10 if is_protagonist else 5
    ls = '-' if not np.isnan(ranks).all() else '--' # 完全跌出的用虚线
    color = color_dict.get(feature, '#95a5a6')

    # 1. 画出连接线 (带有白色描边以防交叉时混淆)
    ax.plot(scales, ranks, marker='o', markersize=10, linewidth=lw, 
            color=color, alpha=alpha, linestyle=ls, zorder=zorder,
            path_effects=[pe.Stroke(linewidth=lw+2, foreground='white'), pe.Normal()])
    
    # 2. 智能标签标注 (极度考验排版功底)
    # 左侧标注 (21bp)
    if not np.isnan(ranks[0]):
        # 如果是主角，字体加粗加大
        fw = 'bold' if is_protagonist else 'normal'
        fs = 12 if is_protagonist else 10
        ax.text(-0.1, ranks[0], feature, ha='right', va='center', 
                fontsize=fs, fontweight=fw, color=color, zorder=zorder)
    
    # 右侧标注 (101bp)
    if not np.isnan(ranks[2]):
        fw = 'bold' if is_protagonist else 'normal'
        fs = 12 if is_protagonist else 10
        ax.text(2.1, ranks[2], feature, ha='left', va='center', 
                fontsize=fs, fontweight=fw, color=color, zorder=zorder)
    
    # 处理只在中间出现的特殊特征 (例如 gc_content_exonicFivePrimeUTR)
    if np.isnan(ranks[0]) and np.isnan(ranks[2]) and not np.isnan(ranks[1]):
        ax.text(1.0, ranks[1]-0.3, feature, ha='center', va='top', 
                fontsize=9, color=color, zorder=zorder)

# =====================================================================
# 第四部分：图表美化与坐标系翻转
# =====================================================================
print("\n✨ 4/4 正在进行最终排版与导出...")

# 【核心】：排名图的 Y 轴必须是倒置的，第 1 名在最上面！
ax.set_ylim(10.5, 0.5) 
ax.set_yticks(range(1, 11))
ax.set_yticklabels([f"Rank {i}" for i in range(1, 11)], fontsize=11)

# X 轴美化
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(scales, fontsize=14, fontweight='bold')
ax.set_xlim(-0.8, 2.8) # 留出巨大空间给两边的文字标签

# 标题与装饰
ax.set_title("Multi-scale Feature Rank Evolution: The Rise of Macro-Topology", 
             fontsize=18, fontweight='bold', pad=25)
ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=1)

# 隐藏边框，让画面更干净呼吸
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()

# 保存
output_file = "m6A_MultiScale_BumpChart.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"🎉 完美收官！秩次迁移图已保存至 '{output_file}'")

plt.show()