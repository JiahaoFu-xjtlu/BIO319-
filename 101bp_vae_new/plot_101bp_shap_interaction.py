import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import gc

# =====================================================================
# 第一部分：数据加载与重装甲模型训练 (101bp 专属)
# =====================================================================
print("⏳ 1/4 正在加载 101bp 全量特征矩阵...")
# 【请确保这是你 101bp 的全量特征文件】
input_file = "Final_Feature_Matrix_for_ML.csv" 
df = pd.read_csv(input_file)

# 核心特征安全检查
target_features = ['H3K4me3', 'gc_content_flank_50bp', 'Cluster_ID']
for feat in target_features:
    if feat not in df.columns:
        raise ValueError(f"❌ 数据集中缺失核心特征: {feat}，请检查文件！")

# 数据清洗与内存压缩
latent_cols_to_drop = [col for col in df.columns if col.startswith('Latent_') or col.startswith('UMAP_')]
cols_to_drop = ['start', 'end', 'Original_Row_Index', 'Cluster_ID'] + latent_cols_to_drop

X = df.select_dtypes(include=['number']).drop(columns=cols_to_drop, errors='ignore')
X = X.astype('float32') 
y = (df['Cluster_ID'] == 0).astype(np.int8)

print("\n🚀 2/4 正在训练 101bp 随机森林模型 (对抗极度不平衡)...")
model = RandomForestClassifier(
    n_estimators=150, 
    max_depth=15, 
    class_weight='balanced', # 核心护城河
    random_state=42, 
    n_jobs=-1 
)
model.fit(X, y)

# =====================================================================
# 第二部分：SHAP 值解析 (构建 1:1 对等解释集)
# =====================================================================
print("\n🧠 3/4 正在执行 SHAP 解析提取特征贡献度...")
explainer = shap.TreeExplainer(model)

# 提取所有的孤岛正样本，并抽样等量的负样本作为背景基准
positive_indices = y[y == 1].index
negative_indices = y[y == 0].sample(n=len(positive_indices), random_state=42).index
explain_indices = positive_indices.union(negative_indices)

X_explain = X.loc[explain_indices].copy()

# 计算 SHAP 值 (关闭 additivity 检查以防止精度截断报错)
shap_values_raw = explainer.shap_values(X_explain, check_additivity=False)

# 【极度严谨】：兼容不同版本 SHAP 的输出格式，安全提取正类 (Cluster 0) 的 SHAP 值
if isinstance(shap_values_raw, list):
    shap_pos = shap_values_raw[1]  
elif len(shap_values_raw.shape) == 3:
    shap_pos = shap_values_raw[:, :, 1]
else:
    shap_pos = shap_values_raw

# 获取我们要分析的 X 轴特征在矩阵中的列索引
target_feat_name = 'gc_content_flank_50bp'
feat_idx = X_explain.columns.get_loc(target_feat_name)

# 提取作图所需的三个核心维度：
# 1. 真实的 GC 含量数值 (X轴)
# 2. GC 含量对应的 SHAP 贡献值 (Y轴)
# 3. 真实的 H3K4me3 数值 (用于颜色分类)
plot_data = pd.DataFrame({
    'GC_Content': X_explain[target_feat_name].values,
    'SHAP_Value': shap_pos[:, feat_idx],
    'H3K4me3_Status': np.where(X_explain['H3K4me3'].values > 0.5, 'Positive (+)', 'Negative (-)')
})

# =====================================================================
# 第三部分：科研级高保真交互依赖图 (Dependence Plot) 渲染
# =====================================================================
print("\n🎨 4/4 正在绘制高精度 SHAP 交互依赖图...")

sns.set_theme(style="ticks", context="paper", font_scale=1.3)
fig, ax = plt.subplots(figsize=(10, 7))

# 画一条 SHAP = 0 的基准线（生死线）
ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

# 【核心 Z-order 管理】：
# 1. 先把 H3K4me3 阴性的序列（没有门票的倒霉蛋）画在底层，用冷灰色
neg_data = plot_data[plot_data['H3K4me3_Status'] == 'Negative (-)']
ax.scatter(
    neg_data['GC_Content'], neg_data['SHAP_Value'],
    color='#bdc3c7', alpha=0.6, s=30, label='H3K4me3 Negative (-)', zorder=2
)

# 2. 再把 H3K4me3 阳性的序列（拿到门票的幸运儿）画在最上层，用极度鲜艳的警示红，加上黑边增强立体感
pos_data = plot_data[plot_data['H3K4me3_Status'] == 'Positive (+)']
ax.scatter(
    pos_data['GC_Content'], pos_data['SHAP_Value'],
    color='#e74c3c', alpha=0.9, s=45, edgecolor='black', linewidth=0.4, label='H3K4me3 Positive (+)', zorder=5
)

# 图表装饰
ax.set_title("SHAP Interaction: H3K4me3 Gates the Effect of Flanking GC Content (101bp Model)", 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("Flanking 50bp GC Content (Raw Feature Value)", fontsize=14)
ax.set_ylabel("SHAP Value (Impact on Cluster 0 Classification)", fontsize=14)

# 优化图例
legend = ax.legend(title="Epigenetic Environment", loc="upper left", frameon=True, fontsize=12)
legend.get_title().set_fontweight('bold')

sns.despine()
plt.tight_layout()

# 输出保存
output_file = "m6A_SHAP_Interaction_GC_H3K4me3.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"🎉 大功告成！交互依赖图已完美保存至 '{output_file}'")

# 清理内存
del X, X_explain, shap_values_raw, shap_pos
gc.collect()

plt.show()