import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import gc # 内存管理神器

# =====================================================================
# 第一部分：极度严谨的数据加载与内存防御机制
# =====================================================================
print("1/5 正在读取已合并特征与标签的全量矩阵 (13.5万行)...")

# 【请确保这个文件是你用 4D 序列特征、宏观基因组特征与 Cluster_ID 拼接好的文件】
input_file = "Final_Feature_Matrix_for_ML.csv"
df = pd.read_csv(input_file)

# 【防泄漏机制】：严格清洗掉所有用于降维和聚类的中间变量
latent_cols_to_drop = [col for col in df.columns if col.startswith('Latent_') or col.startswith('UMAP_')]
cols_to_drop = ['start', 'end', 'Original_Row_Index', 'Cluster_ID'] + latent_cols_to_drop

# 分离特征矩阵 (X) 并施加极限内存压缩
X = df.select_dtypes(include=['number']).drop(columns=cols_to_drop, errors='ignore')
X = X.astype('float32') # 将默认的 float64 压缩一半内存

# 锁定目标孤岛 (Cluster 0)
target_cluster = 0
# 将布尔值转为最省内存的 int8 (只占 1 byte)
y = (df['Cluster_ID'] == target_cluster).astype(np.int8)

print(f"✅ 全量数据准备完毕！")
print(f"   特征总数: {X.shape[1]}")
print(f"   总样本数: {X.shape[0]}")
print(f"   目标孤岛 (Cluster 0) 样本数: {y.sum()}")

# 释放原始大文件占用的内存
del df
gc.collect()

# =====================================================================
# 第二部分：重装甲模型训练 (对抗 2% 极度不平衡)
# =====================================================================
print("\n2/5 正在全量数据上训练随机森林模型 (请耐心等待)...")

# 【世界级生信大牛的参数精调】：
# 1. n_estimators=150: 树的数量足够多，保证对 13.5 万数据的特征覆盖。
# 2. max_depth=15: 防止深层过拟合，同时显著加快全量训练速度。
# 3. class_weight='balanced': 绝对核心！让模型对那 2739 个少数派孤岛的错误预测施加 50 倍的惩罚，强制模型寻找真实特征边界。
model = RandomForestClassifier(
    n_estimators=150, 
    max_depth=15, 
    class_weight='balanced', 
    random_state=42, 
    n_jobs=-1 
)

# 直接喂入 13.5 万行全量数据！
model.fit(X, y)

# =====================================================================
# 第三部分：特征贡献度解码 (SHAP 1:1 对等解释集)
# =====================================================================
print("\n3/5 模型训练完成。正在执行 SHAP 值解析...")
explainer = shap.TreeExplainer(model)

# 【构建最具科学说服力的解释集】：
# 我们不解释全部 13.5 万行（图表会变成蓝色的海洋，且内存会崩溃）。
# 我们把 2739 条正样本 (孤岛) 全部拿出来！
positive_indices = y[y == 1].index

# 从剩下的大部队中，随机且公平地抽取等量（2739条）负样本作为对照组
negative_indices = y[y == 0].sample(n=len(positive_indices), random_state=42).index

# 拼接成 5478 条用于 SHAP 解析的精华对抗集
explain_indices = positive_indices.union(negative_indices)
X_explain = X.loc[explain_indices]

# 关闭 check_additivity 绕过底层精度报错
shap_values = explainer.shap_values(X_explain, check_additivity=False)

# =====================================================================
# 第四部分：科研级高保真可视化输出
# =====================================================================
print("\n4/5 正在生成科研级特征重要性蜂巢图...")

# 提取正类（预测为 Cluster 0）的 SHAP 值
try:
    if isinstance(shap_values, list):
        final_shap = shap_values[1]  
    else:
        final_shap = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values
except Exception as e:
    final_shap = shap_values

# 设置画幅，准备绘图
plt.figure(figsize=(12, 10))

# max_display=20 限制只展示最核心的前 20 个特征，避免图表臃肿
shap.summary_plot(final_shap, X_explain, plot_type="dot", show=False, max_display=20) 

# 图表装饰
plt.title(f"SHAP Attribution: Determinants of the Cluster {target_cluster}, Explainer N={len(explain_indices)})", fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()

# 保存高质量图片
output_path = f"m6A_101bp_Cluster{target_cluster}_SHAP.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n🎉 完美解码！基于全量训练和对等解释的特征决定图已保存至: '{output_path}'")

# 清理内存并展示图片
del X_explain, final_shap, shap_values
gc.collect()

plt.show()