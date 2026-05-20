import pandas as pd
import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
import time

print("🚀 启动终极基线测试：Raw Data UMAP + HDBSCAN")

# =====================================================================
# 第一部分：加载并“拍扁”原始 21bp One-hot 数据
# =====================================================================
print("⏳ 正在加载原始 One-hot 特征矩阵...")
input_file = "m6A_21bp_onehot_features.csv" # 确保这里是你的 4通道纯净原始数据
raw_data_df = pd.read_csv(input_file, header=None, dtype="float32")

# 当前 raw_data_df 应该是 135300 行 x 84 列 (21 * 4)
raw_features = raw_data_df.values

print(f"✅ 原始数据加载成功！")
print(f"📄 数据形状: {raw_features.shape[0]} 条序列, {raw_features.shape[1]} 维扁平特征")

# =====================================================================
# 第二部分：直接对 84 维原始空间进行 UMAP 降维
# =====================================================================
print("\n🔥 正在对高维原始数据运行 UMAP (由于维度较高，耗时将比 VAE 特征长，请耐心等待)...")
start_time = time.time()

reducer = umap.UMAP(
    n_neighbors=50,       
    min_dist=0.0,         
    n_components=2,       
    metric='euclidean',   
    random_state=42       
)

# 直接把 84 维的原始数据喂进去！
embedding = reducer.fit_transform(raw_features)

end_time = time.time()
print(f"✅ UMAP 降维完成！耗时: {end_time - start_time:.1f} 秒")

# =====================================================================
# 第三部分：HDBSCAN 密度聚类
# =====================================================================
print("\n🔍 正在对原始数据降维结果运行 HDBSCAN 聚类...")

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=1000, 
    min_samples=50, 
    metric='euclidean',
    cluster_selection_method='eom' 
)

cluster_labels = clusterer.fit_predict(embedding)

num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
noise_points = list(cluster_labels).count(-1)
print(f"✅ 聚类完成！在未经神经网络处理的数据中，发现了 {num_clusters} 个亚型。")
print(f"⚠️ 原始数据产生的噪音/边缘点数量: {noise_points}")

# =====================================================================
# 第四部分：保存基准结果
# =====================================================================
print("\n💾 正在保存 Raw Data 的聚类标签...")

results_df = pd.DataFrame({
    'UMAP_1': embedding[:, 0],
    'UMAP_2': embedding[:, 1],
    'Cluster_Label': cluster_labels
})

output_csv = "m6A_21bp_umap_hdbscan_labels.csv"
results_df.to_csv(output_csv, index=False)
print(f"🎉 标签保存成功！文件: {output_csv} (请用它去计算 Raw Data 的 Silhouette Score)")

# =====================================================================
# 第五部分：原始空间的可视化揭秘
# =====================================================================
print("\n🎨 正在绘制 Raw Data 聚类结果图...")

plt.figure(figsize=(12, 10))

mask_noise = cluster_labels == -1
mask_core = cluster_labels != -1

# 噪音点
plt.scatter(embedding[mask_noise, 0], embedding[mask_noise, 1], 
            c='lightgray', s=1, alpha=0.3, label='Noise')

# 核心簇
scatter = plt.scatter(embedding[mask_core, 0], embedding[mask_core, 1], 
                      c=cluster_labels[mask_core], cmap='Spectral', 
                      s=3, alpha=0.8)

plt.title('UMAP Projection of RAW 21bp One-hot Data', fontsize=16)
plt.xlabel('UMAP Dimension 1', fontsize=12)
plt.ylabel('UMAP Dimension 2', fontsize=12)
plt.colorbar(scatter, label='Subtype Cluster ID')
plt.grid(False)

plt.savefig('m6A_21bp_umap_clusters.png', dpi=300, bbox_inches='tight')
print("🎉 绘图完成！去看看没有深度学习的帮助，原始数据糊成了什么样子吧！")