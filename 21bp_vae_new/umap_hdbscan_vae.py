import os
import pandas as pd
import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.image as mpimg

# =====================================================================
# 第一部分：基础设置
# =====================================================================

scale = "21bp"

input_file = f"m6A_{scale}_latent_features_16d.csv"
output_csv = f"m6A_{scale}_umap_hdbscan_labels.csv"

# 你的 sequence logo 总图文件
motif_logo_file = "m6A_21bp_Motif_Logos.png"

# 根据你已经生成的 sequence logo 图，为每个 cluster 手动添加 motif 注释
# 注意：这里的 motif 名称是用于图中标注的简洁版本。
# 如果你后续有更严格的 consensus motif 结果，只需要修改这个字典即可。
motif_annotations = {
    0: "CAG\nm6Am-like",
    1: "GGACU",
    2: "AGACU",
    3: "AGACU",
    4: "GGACU",
    5: "GGACU",
    6: "GGACU",
    7: "AGACU",
    8: "AAACU",
    9: "GAGAC",
    10: "AGACU"
}

# 如果某些 label 在图上互相重叠，可以在这里微调文本偏移量
# 格式：cluster_id: (x_offset, y_offset)
label_offsets = {
    0: (0.00, -0.25),
    1: (0.10, 0.20),
    2: (0.10, 0.20),
    3: (-0.10, -0.20),
    4: (0.10, 0.20),
    5: (0.10, -0.20),
    6: (0.00, 0.20),
    7: (0.10, 0.20),
    8: (-0.10, 0.20),
    9: (-0.15, -0.20),
    10: (0.15, -0.20)
}

# =====================================================================
# 第二部分：加载 VAE 潜空间特征
# =====================================================================

print("⏳ 正在加载 VAE 潜空间特征...")

latent_features = pd.read_csv(input_file, header=None).values

print(f"✅ 特征加载成功！数据形状: {latent_features.shape}")

# =====================================================================
# 第三部分：UMAP 降维
# =====================================================================

print("\n🚀 正在运行 UMAP 降维...")

reducer = umap.UMAP(
    n_neighbors=50,
    min_dist=0.0,
    n_components=2,
    metric="euclidean",
    random_state=42
)

embedding = reducer.fit_transform(latent_features)

print("✅ UMAP 降维完成！")

# =====================================================================
# 第四部分：HDBSCAN 密度聚类
# =====================================================================

print("\n🔍 正在运行 HDBSCAN 聚类...")

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=1000,
    min_samples=50,
    metric="euclidean",
    cluster_selection_method="eom"
)

cluster_labels = clusterer.fit_predict(embedding)

num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
noise_points = list(cluster_labels).count(-1)

print(f"✅ 聚类完成！共发现 {num_clusters} 个高置信度的 Motif 亚型。")
print(f"⚠️ 过滤掉了 {noise_points} 条难以归类的边缘/噪音序列。")

# =====================================================================
# 第五部分：保存 UMAP 坐标与聚类标签
# =====================================================================

print("\n💾 正在保存降维坐标与聚类标签...")

results_df = pd.DataFrame({
    "UMAP_1": embedding[:, 0],
    "UMAP_2": embedding[:, 1],
    "Cluster_Label": cluster_labels
})

results_df.to_csv(output_csv, index=False)

print(f"🎉 标签保存成功！文件: {output_csv}")

# =====================================================================
# 第六部分：定义绘图函数 —— 带 motif 标注的 UMAP 图
# =====================================================================

def plot_umap_with_motif_annotations(ax, embedding, cluster_labels, motif_annotations):
    """
    在 UMAP/HDBSCAN 聚类图上标注每个 cluster 对应的 motif 信息。
    """

    mask_noise = cluster_labels == -1
    mask_core = cluster_labels != -1

    # 先画噪音点
    ax.scatter(
        embedding[mask_noise, 0],
        embedding[mask_noise, 1],
        c="lightgray",
        s=1,
        alpha=0.25,
        label="Noise / Transition"
    )

    # 再画核心 cluster
    scatter = ax.scatter(
        embedding[mask_core, 0],
        embedding[mask_core, 1],
        c=cluster_labels[mask_core],
        cmap="Spectral",
        s=3,
        alpha=0.80
    )

    ax.set_title(
        f"UMAP Projection of {scale} VAE Latent Space\nwith HDBSCAN Motif Subtype Annotations",
        fontsize=16,
        fontweight="bold"
    )
    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    ax.grid(False)

    # 给每个 cluster 加 motif 标注
    unique_clusters = sorted([c for c in np.unique(cluster_labels) if c != -1])

    for cid in unique_clusters:
        cluster_points = embedding[cluster_labels == cid]

        # 用中位数作为 cluster 中心，避免被离群点影响
        x_center = np.median(cluster_points[:, 0])
        y_center = np.median(cluster_points[:, 1])

        dx, dy = label_offsets.get(cid, (0.0, 0.0))

        motif_text = motif_annotations.get(cid, "Motif NA")
        n_points = cluster_points.shape[0]

        label_text = f"Cluster {cid}\n{motif_text}\nn={n_points}"

        txt = ax.text(
            x_center + dx,
            y_center + dy,
            label_text,
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="black",
                alpha=0.82,
                linewidth=0.8
            )
        )

        # 给文字加白色描边，提高可读性
        txt.set_path_effects([
            path_effects.Stroke(linewidth=2.5, foreground="white"),
            path_effects.Normal()
        ])

    return scatter

# =====================================================================
# 第七部分：图 1 —— 单独保存带 motif 标注的 UMAP 聚类图
# =====================================================================

print("\n🎨 正在绘制带 motif 标注的 UMAP 聚类图...")

fig, ax = plt.subplots(figsize=(13, 10))

scatter = plot_umap_with_motif_annotations(
    ax=ax,
    embedding=embedding,
    cluster_labels=cluster_labels,
    motif_annotations=motif_annotations
)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Motif Subtype Cluster ID", fontsize=11)

annotated_umap_png = f"m6A_{scale}_vae_umap_hdbscan_clusters_motif_annotated.png"

plt.savefig(
    annotated_umap_png,
    dpi=300,
    bbox_inches="tight"
)

print(f"🎉 带 motif 标注的 UMAP 图已保存为: {annotated_umap_png}")

plt.show()

# =====================================================================
# 第八部分：图 2 —— UMAP 聚类图 + sequence logo 总图并排展示
# =====================================================================

print("\n🎨 正在绘制 UMAP + sequence logo panel 组合图...")

fig = plt.figure(figsize=(22, 11))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])

# 左侧：UMAP 聚类图
ax_umap = fig.add_subplot(gs[0, 0])

scatter = plot_umap_with_motif_annotations(
    ax=ax_umap,
    embedding=embedding,
    cluster_labels=cluster_labels,
    motif_annotations=motif_annotations
)

cbar = plt.colorbar(scatter, ax=ax_umap, fraction=0.046, pad=0.04)
cbar.set_label("Motif Subtype Cluster ID", fontsize=11)

# 右侧：sequence logo 总图
ax_logo = fig.add_subplot(gs[0, 1])

if os.path.exists(motif_logo_file):
    logo_img = mpimg.imread(motif_logo_file)
    ax_logo.imshow(logo_img)
    ax_logo.set_title(
        "Sequence Logos of HDBSCAN-derived Motif Subtypes",
        fontsize=15,
        fontweight="bold"
    )
    ax_logo.axis("off")
else:
    ax_logo.axis("off")
    ax_logo.text(
        0.5,
        0.5,
        f"Motif logo file not found:\n{motif_logo_file}",
        ha="center",
        va="center",
        fontsize=14
    )

combined_png = f"m6A_{scale}_vae_umap_with_motif_logo_panel.png"

plt.savefig(
    combined_png,
    dpi=300,
    bbox_inches="tight"
)

print(f"🎉 UMAP + motif logo 组合图已保存为: {combined_png}")

plt.show()