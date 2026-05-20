import pandas as pd
import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt

# =====================================================================
# 第一部分：加载纯净的 21bp (4D) 潜空间特征
# =====================================================================
print("⏳ 正在加载 VAE 潜空间特征...")
# 对接上一步的最新文件名
input_file = "m6A_51bp_latent_features_32d.csv"
latent_features = pd.read_csv(input_file, header=None).values

print(f"✅ 特征加载成功！数据形状: {latent_features.shape}")

# =====================================================================
# 第二部分：UMAP 降维 (保留原有绝佳参数)
# =====================================================================
print("\n🚀 正在运行 UMAP 降维 (这可能需要几分钟，请耐心等待)...")

reducer = umap.UMAP(
    n_neighbors=50,       
    min_dist=0.0,         
    n_components=2,       
    metric='euclidean',   
    random_state=42       
)

embedding = reducer.fit_transform(latent_features)
print("✅ UMAP 降维完成！")

# =====================================================================
# 第三部分：HDBSCAN 密度聚类 (微调以适应连续流形)
# =====================================================================
print("\n🔍 正在运行 HDBSCAN 聚类...")

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=1000, 
    min_samples=50, # 【专家微调】：从 100 下调至 50，以包容 VAE 特有的连续过渡带
    metric='euclidean',
    cluster_selection_method='eom' 
)

cluster_labels = clusterer.fit_predict(embedding)

num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
noise_points = list(cluster_labels).count(-1)
print(f"✅ 聚类完成！共发现 {num_clusters} 个高置信度的 Motif 亚型。")
print(f"⚠️ 过滤掉了 {noise_points} 条难以归类的边缘/噪音序列。")

# =====================================================================
# 第四部分：极其关键 —— 保存聚类结果 (为算分和 SHAP 做准备)
# =====================================================================
print("\n💾 正在保存降维坐标与聚类标签...")

# 将 UMAP 的二维坐标和对应的亚型标签拼成一张表
results_df = pd.DataFrame({
    'UMAP_1': embedding[:, 0],
    'UMAP_2': embedding[:, 1],
    'Cluster_Label': cluster_labels
})

# 保存为 CSV
output_csv = "m6A_51bp_umap_hdbscan_labels.csv"
results_df.to_csv(output_csv, index=False)
print(f"🎉 标签保存成功！文件: {output_csv} (这是下一步计算 Silhouette Score 的核心弹药！)")

# =====================================================================
# 第五部分：科研级可视化
# =====================================================================
print("\n🎨 正在绘制聚类结果图...")

plt.figure(figsize=(12, 10))

mask_noise = cluster_labels == -1
mask_core = cluster_labels != -1

# 先画噪音
plt.scatter(embedding[mask_noise, 0], embedding[mask_noise, 1], 
            c='lightgray', s=1, alpha=0.3, label='Noise / Transition')

# 再画核心的 Motif 亚型簇
scatter = plt.scatter(embedding[mask_core, 0], embedding[mask_core, 1], 
                      c=cluster_labels[mask_core], cmap='Spectral', 
                      s=3, alpha=0.8)

plt.title('UMAP Projection of 51bp VAE Latent Space', fontsize=16)
plt.xlabel('UMAP Dimension 1', fontsize=12)
plt.ylabel('UMAP Dimension 2', fontsize=12)
plt.colorbar(scatter, label='Motif Subtype Cluster ID')
plt.grid(False)

# 更新了图片命名，标明这是 21bp 且剔除了 N 碱基的纯净版
plt.savefig('m6A_51bp_vae_umap_hdbscan_clusters.png', dpi=300, bbox_inches='tight')
print("🎉 绘图完成！图片已保存为 'm6A_21bp_vae_umap_clusters.png'")

plt.show()