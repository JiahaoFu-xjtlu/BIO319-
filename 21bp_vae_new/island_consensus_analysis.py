import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =====================================================================
# 第一步：极其严谨的数据加载与对齐检查
# =====================================================================
print("⏳ 正在加载并校验数据底座...")

labels_file = "m6A_21bp_umap_hdbscan_labels.csv"
features_file = "m6A_21bp_onehot_features.csv"

# 1. 加载聚类标签
labels_df = pd.read_csv(labels_file)

# 2. 加载原始的扁平化 One-hot 特征矩阵 (135300, 84)
raw_onehot = pd.read_csv(features_file, header=None).values

# 🔐 【核心严谨性检查】：确保标签数量与特征矩阵行数绝对一致
assert len(labels_df) == raw_onehot.shape[0], \
    f"❌ 数据不对齐！标签有 {len(labels_df)} 行，但特征矩阵有 {raw_onehot.shape[0]} 行。"
print(f"✅ 数据对齐校验通过！总样本量: {len(labels_df)}")
print(f"✅ 原始特征维度校验通过！特征数: {raw_onehot.shape[1]} (对应 21bp * 4)")

# =====================================================================
# 第二步：精准打捞 Cluster 0 (孤岛)
# =====================================================================
print("\n🤿 正在深潜打捞孤岛 (Cluster 0) 序列...")

TARGET_CLUSTER = 0

# 获取孤岛序列在原始数据集中的绝对索引
island_indices = labels_df.index[labels_df['Cluster_Label'] == TARGET_CLUSTER].tolist()
island_size = len(island_indices)

if island_size == 0:
    raise ValueError(f"❌ 错误：在标签中找不到 Cluster {TARGET_CLUSTER}，请检查聚类结果！")

print(f"🎯 成功锁定孤岛！共提取出 {island_size} 条序列。")

# 根据索引切片，提取孤岛的 One-hot 特征
island_raw_data = raw_onehot[island_indices] # 形状: (island_size, 84)

# 将扁平的 84 维张量还原为生信可读的 (样本数, 序列长度, 碱基种类)
island_tensor = island_raw_data.reshape(island_size, 21, 4)

# =====================================================================
# 第三步：计算位置权重矩阵 (Position Weight Matrix, PWM)
# =====================================================================
print("\n🧮 正在计算碱基位置频率...")

# 沿着样本维度 (axis=0) 求平均值，得到 21bp 上每个位置、每种碱基的出现频率
# 结果形状: (21, 4)
motif_frequencies = np.mean(island_tensor, axis=0)

# 验证概率和是否为 1 (容许极小的浮点数误差)
assert np.allclose(np.sum(motif_frequencies, axis=1), 1.0), "❌ PWM 概率计算异常！"

# =====================================================================
# 第四步：提取共识序列 (Consensus Sequence)
# =====================================================================
bases = ['A', 'C', 'G', 'T'] # 依据你的 R 代码，第四列代表 T/U，这里统一标记为 T
consensus_seq = ""

for pos in range(21):
    dominant_base_idx = np.argmax(motif_frequencies[pos])
    dominant_freq = motif_frequencies[pos][dominant_base_idx]
    
    # 严格的共识标准：频率 > 50% 才认为是明确的基序，否则用小写或 N 表示不确定性
    if dominant_freq >= 0.75:
        consensus_seq += bases[dominant_base_idx].upper() # 极度保守 (>75%)
    elif dominant_freq >= 0.50:
        consensus_seq += bases[dominant_base_idx].lower() # 相对保守 (50%-75%)
    else:
        consensus_seq += "N" # 缺乏共识 (<50%)

print(f"\n👑 【孤岛解码结果】")
print(f"   共识序列 (Consensus): 5'- {consensus_seq} -3'")

# =====================================================================
# 第五步：生成科研级 PWM 热图 (出版物质量)
# =====================================================================
print("\n🎨 正在绘制高分辨率序列频率热图...")

# 构造 DataFrame 供 Seaborn 绘图
positions = [f"P{i+1}" for i in range(21)]
positions[10] = "P11(m6A)" # 标记中心位点
freq_df = pd.DataFrame(motif_frequencies.T, index=bases, columns=positions)

plt.figure(figsize=(16, 4)) # 长条形画布，适合展示序列
ax = sns.heatmap(freq_df, cmap="Reds", annot=True, fmt=".2f", 
                 annot_kws={"size": 8}, linewidths=0.5, linecolor='gray',
                 cbar_kws={'label': 'Base Frequency'})

# 图表装饰
plt.title(f'Nucleotide Frequency (PWM) for Isolated Cluster 0 (N = {island_size})', fontsize=16, pad=15)
plt.xlabel('Sequence Position (21bp Context)', fontsize=14, labelpad=10)
plt.ylabel('Nucleotide', fontsize=14)

# 强制 Y 轴文字水平显示，更易读
plt.yticks(rotation=0, fontsize=12)
plt.xticks(rotation=45, fontsize=10)

# 保存图片
output_img = f'm6A_21bp_Cluster0_Island_PWM.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"🎉 绘图完成！热图已保存为 '{output_img}'")

plt.show()