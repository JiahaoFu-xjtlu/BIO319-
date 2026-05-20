import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import time
import os

# =====================================================================
# 第一部分：极度严谨的轮廓系数评估引擎
# =====================================================================
def calculate_rigorous_silhouette(feature_matrix, labels, dataset_name):
    """
    计算严谨的轮廓系数（自动剥离噪声，并对计算耗时和内存进行优化保护）
    """
    print(f"\n⏳ 正在评估: {dataset_name} ...")
    
    # 1. 噪声剥离 (HDBSCAN 的 -1 不属于任何亚型，必须剔除)
    mask = (labels != -1)
    
    clean_features = feature_matrix[mask]
    clean_labels = labels[mask]
    
    total_samples = len(labels)
    clean_samples = len(clean_labels)
    noise_ratio = 1 - (clean_samples / total_samples)
    
    print(f"   📊 总样本: {total_samples} | 去噪后聚类样本: {clean_samples} | 噪声率: {noise_ratio:.2%}")
    
    # 2. 极端异常拦截
    unique_clusters = np.unique(clean_labels)
    if len(unique_clusters) < 2:
        print(f"   ⚠️ 警告: {dataset_name} 去除噪声后只剩下 {len(unique_clusters)} 个亚型。无法计算轮廓系数！")
        return None, noise_ratio

    # 3. 蒙特卡洛近似计算 (防止 13 万条数据导致内存核爆)
    print(f"   🧮 正在高维空间计算 Silhouette Score (采用 20000 随机采样)...")
    start_time = time.time()
    
    sample_size = min(20000, clean_samples) 
    
    score = silhouette_score(
        clean_features, 
        clean_labels, 
        metric='euclidean', 
        sample_size=sample_size, 
        random_state=42 # 固定种子，保证可复现
    )
    
    end_time = time.time()
    print(f"   ✅ 计算完成！耗时: {end_time - start_time:.2f} 秒")
    print(f"   🏆 {dataset_name} Average Silhouette Score: {score:.4f}")
    
    return score, noise_ratio

# =====================================================================
# 第二部分：主控程序 (双文件加载机制 - 彻底解决列名和空间陷阱)
# =====================================================================
if __name__ == "__main__":
    
    # 【核心修改区】：采用“特征文件”与“标签文件”分离的加载模式
    # 请确保 paths 字典中的文件路径与你的实际情况一致
    evaluation_config = {
        "1. Traditional AE (21bp)": {
            # 高维特征：必须使用你之前用 AE 提取出的潜变量 (比如 32 维) CSV
            "features_path": "m6A_21bp_AE_latent_features_16d.csv", 
            # 标签文件：你刚给我的文件
            "labels_path": "m6A_21bp_ae_umap_hdbscan_labels.csv" 
        },
        "2. VAE (21bp)": {
            "features_path": "m6A_21bp_vae_latent_features_16d.csv",
            "labels_path": "m6A_21bp_vae_umap_hdbscan_labels.csv"
        },
        "3. Raw_Data (21bp One-Hot)": {
            "features_path": "m6A_21bp_onehot_features.csv",
            "labels_path": "m6A_21bp_raw_umap_hdbscan_labels.csv"
        }
    }
    
    results = []
    
    print("🚀 开始 21bp 表征空间聚类质量终极评估...")
    print("=" * 60)
    
    for name, paths in evaluation_config.items():
        feat_path = paths["features_path"]
        lbl_path = paths["labels_path"]
        
        if not os.path.exists(feat_path) or not os.path.exists(lbl_path):
            print(f"\n❌ 跳过 {name}: 找不到特征文件或标签文件，请检查路径。")
            continue
            
        try:
            # 1. 安全加载 Label 文件
            df_labels = pd.read_csv(lbl_path)
            
            # 【完美兼容】：智能识别 Cluster_Label 或 cluster_label
            if 'Cluster_Label' in df_labels.columns:
                labels = df_labels['Cluster_Label'].values
            elif 'cluster_label' in df_labels.columns:
                labels = df_labels['cluster_label'].values
            elif 'Cluster_ID' in df_labels.columns:
                 labels = df_labels['Cluster_ID'].values
            else:
                raise ValueError(f"在 {lbl_path} 中找不到标签列！存在的列名有: {df_labels.columns.tolist()}")

            # 2. 安全且强制的 Feature 文件加载逻辑
            # 如果是 One-Hot 原始特征文件，强制无表头读取；否则默认有表头（或让 Pandas 推断）
            if 'onehot' in feat_path.lower():
                df_features = pd.read_csv(feat_path, header=None)
            else:
                df_features = pd.read_csv(feat_path, header=None if 'latent' in feat_path.lower() else 'infer')
            
            # 智能清理：如果特征文件里不小心混入了画图的列，强行剥离
            cols_to_drop = [col for col in df_features.columns if str(col).startswith('UMAP_') or str(col) in ['Cluster_Label', 'cluster_label', 'Original_Row_Index']]
            features = df_features.drop(columns=cols_to_drop, errors='ignore').values
            
            # 3. 数据对齐校验 (防呆设计)
            if len(features) != len(labels):
                raise ValueError(f"严重错误: 特征数量 ({len(features)}) 与 标签数量 ({len(labels)}) 不一致！")
            
            if features.shape[1] <= 3:
                print(f"   ⚠️ 严重警告: {name} 的特征维度只有 {features.shape[1]} 维，你可能错误地在使用 UMAP 坐标计算轮廓系数！")
                
            # 执行底层计算
            score, noise = calculate_rigorous_silhouette(features, labels, name)
            
            if score is not None:
                results.append({"Representation": name, "Silhouette_Score": score, "Noise_Ratio": noise})
                
        except Exception as e:
            print(f"\n❌ {name} 运行出错: {str(e)}")
            
    # =====================================================================
    # 第三部分：生成最终学术对比报告
    # =====================================================================
    if results:
        print("\n" + "=" * 60)
        print("🎯 最终对比报告 (Final Evaluation Report):")
        print("=" * 60)
        report_df = pd.DataFrame(results)
        report_df = report_df.sort_values(by="Silhouette_Score", ascending=False).reset_index(drop=True)
        print(report_df.to_string(index=False))
        print("=" * 60)