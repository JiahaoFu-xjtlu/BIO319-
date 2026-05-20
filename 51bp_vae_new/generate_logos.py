import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logomaker

# =====================================================================
# 第一部分：严谨地加载纯净 4D 数据与标签
# =====================================================================
print("⏳ 正在加载原始的纯净 4D One-hot 序列矩阵...")
raw_csv = pd.read_csv("m6A_51bp_onehot_features.csv", header=None, dtype=np.float32)
# 将其折叠为 4D 张量 (135300, 21, 4)
raw_data = raw_csv.values.reshape(-1, 51, 4)

print("⏳ 正在加载之前保存的聚类标签 (保证绝对一致性)...")
labels_df = pd.read_csv("m6A_51bp_umap_hdbscan_labels.csv")
labels = labels_df['Cluster_Label'].values

# 🔐 严谨性断言
assert len(labels) == raw_data.shape[0], "❌ 错误：序列数据与标签数量不匹配！"

unique_clusters = sorted([lbl for lbl in set(labels) if lbl != -1])
print(f"✅ 成功锁定 {len(unique_clusters)} 个高置信度 Motif 亚型！")

# =====================================================================
# 第三部分：极其重要的生命线 —— 保存序列与类别的映射关系
# =====================================================================
print("\n💾 正在保存序列与类别的映射关系...")
mapping_df = pd.DataFrame({
    'Original_Row_Index': range(len(labels)), # 记录原始 CSV 里的行号
    'Cluster_ID': labels                      # 记录它被分到了哪一个簇 (-1 代表噪音)
})

mapping_file = 'm6A_51bp_Sequence_Cluster_Mapping.csv'
mapping_df.to_csv(mapping_file, index=False)
print(f"✅ 映射关系已安全保存至 '{mapping_file}'！")

# =====================================================================
# 第四部分：计算信息熵矩阵并绘制 Sequence Logo (4D 专属版)
# =====================================================================
print("\n🎨 正在为每一个 Motif 亚型绘制 4D Sequence Logo...")

# 去除 'N' 碱基，定义 RNA 的 4 个核心通道
bases = ['A', 'C', 'G', 'U']
rna_color_scheme = {'A': '#109648', 'C': '#255C99', 'G': '#F7B32B', 'U': '#D62839'}

cols = 3
rows = int(np.ceil(len(unique_clusters) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
axes = axes.flatten() 

for idx, cluster_id in enumerate(unique_clusters):
    ax = axes[idx]
    
    cluster_seqs = raw_data[labels == cluster_id]
    
    ppm = cluster_seqs.mean(axis=0)
    ppm_df = pd.DataFrame(ppm, columns=bases)
    
    info_df = logomaker.transform_matrix(ppm_df, from_type='probability', to_type='information')
    
    logo = logomaker.Logo(info_df, ax=ax, color_scheme=rna_color_scheme)
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    
    ax.set_title(f'Cluster {cluster_id} (n={len(cluster_seqs)})', fontsize=14, fontweight='bold')
    ax.set_ylabel('Information (bits)')
    
    ax.set_xticks(range(21))
    xtick_labels = [str(i-10) if i == 10 else '' for i in range(21)]
    xtick_labels[10] = "m6A(0)"
    ax.set_xticklabels(xtick_labels, rotation=45)
    
    ax.axvline(x=10, color='red', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 2.0)

for idx in range(len(unique_clusters), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
output_name = 'm6A_51bp_Motif_Logos.png'
plt.savefig(output_name, dpi=300, bbox_inches='tight')
print(f"\n🎉 大功告成！全景密码已成功绘制并保存为 '{output_name}'")