import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# =====================================================================
# 第一部分：绝对严谨的三尺度数据加载与主键校验
# =====================================================================
print("⏳ 1/4 正在加载 21bp, 51bp 和 101bp 三个尺度的特征矩阵...")

# 必须加载全部三个文件，才能求出真正的“三尺度完全重叠”
file_21bp = "21bp_Final_Feature_Matrix_for_ML.csv"
file_51bp = "51bp_Final_Feature_Matrix_for_ML.csv"
file_101bp = "101bp_Final_Feature_Matrix_for_ML.csv"

try:
    df_21 = pd.read_csv(file_21bp)
    df_51 = pd.read_csv(file_51bp)
    df_101 = pd.read_csv(file_101bp)
except FileNotFoundError as e:
    raise FileNotFoundError(f"❌ 找不到文件，请确保三个尺度的 CSV 文件均在当前目录下: {e}")

ID_COLUMN = 'Original_Row_Index'  
CLUSTER_COLUMN = 'Cluster_ID'

# 防御性校验
for df, name in zip([df_21, df_51, df_101], ['21bp', '51bp', '101bp']):
    if ID_COLUMN not in df.columns or CLUSTER_COLUMN not in df.columns:
        raise ValueError(f"❌ 致命错误：{name} 数据集中缺失主键 '{ID_COLUMN}' 或聚类标签 '{CLUSTER_COLUMN}'！")

# =====================================================================
# 第二部分：真·三尺度集合运算 (修正 N 数不匹配的 Bug)
# =====================================================================
print("\n🧮 2/4 正在执行严格的三尺度流形追踪 (Three-Scale Manifold Tracking)...")

# 1. 提取三个尺度下各自 Cluster 0 的原始物理编号
c0_21bp_ids = set(df_21[df_21[CLUSTER_COLUMN] == 0][ID_COLUMN])
c0_51bp_ids = set(df_51[df_51[CLUSTER_COLUMN] == 0][ID_COLUMN])
c0_101bp_ids = set(df_101[df_101[CLUSTER_COLUMN] == 0][ID_COLUMN])

print(f"   📊 21bp 初始识别数量: {len(c0_21bp_ids)}")

# 2. 核心计算：三重交集与差集
# 【真正的 Core】：只有在 21bp AND 51bp AND 101bp 中全都是 Cluster 0 的序列
core_ids = c0_21bp_ids.intersection(c0_51bp_ids).intersection(c0_101bp_ids) 

# 【真正的 Dropout】：在 21bp 时是 Cluster 0，但在后续（51bp 或 101bp）扩增视野时被踢出的序列
dropout_ids = c0_21bp_ids.difference(core_ids)

print(f"   🎯 真·绝对核心群体 (三尺度稳定): {len(core_ids)} 条")
print(f"   🚫 真·多余边缘群体 (后续尺度剔除): {len(dropout_ids)} 条")

# =====================================================================
# 第三部分：特征提取
# =====================================================================
print("\n🔍 3/4 正在提取目标特征 (gc_content_flank_50bp) 进行底层对比...")

feature_name = 'gc_content_flank_50bp'
if feature_name not in df_101.columns:
    raise ValueError(f"❌ 在 101bp 数据集中找不到特征 '{feature_name}'！")

df_core = df_101[df_101[ID_COLUMN].isin(core_ids)][[feature_name]].copy()
df_core['Subpopulation'] = 'Core (Stable across 3 scales)'

df_dropout = df_101[df_101[ID_COLUMN].isin(dropout_ids)][[feature_name]].copy()
df_dropout['Subpopulation'] = 'Dropout (Rejected at 51/101bp)'

df_plot = pd.concat([df_core, df_dropout], axis=0)

# =====================================================================
# 第四部分：绘制顶刊级别对比箱线图
# =====================================================================
print("\n🎨 4/4 正在绘制高阶对比箱线图...")

sns.set_theme(style="ticks", context="paper", font_scale=1.2)
plt.figure(figsize=(7, 7)) 

custom_palette = {
    'Core (Stable across 3 scales)': '#27ae60',   
    'Dropout (Rejected at 51/101bp)': '#e67e22' 
}

ax = sns.boxplot(
    data=df_plot, 
    x='Subpopulation', 
    y=feature_name, 
    palette=custom_palette,
    width=0.5,
    linewidth=1.5,
    fliersize=3 
)

plt.title("Macroscopic GC Topologies: Core vs. Dropout Sequences", fontsize=16, fontweight='bold', pad=25)
plt.xlabel("") 
plt.ylabel("Flanking 50bp GC Content", fontsize=14)

# 动态将 N 数融入 X 轴标签
ax.set_xticklabels([
    f"Core (Stable across 3 scales)\n(N = {len(core_ids)})",
    f"Dropout (Rejected at 51/101bp)\n(N = {len(dropout_ids)})"
])

# 统计检验
val_core = df_core[feature_name]
val_dropout = df_dropout[feature_name]
u_stat, p_val = stats.mannwhitneyu(val_core, val_dropout, alternative='two-sided')

if p_val == 0.0 or p_val < 1e-300:
    p_str = "P < 1.0e-300"
elif p_val < 0.0001:
    p_str = "P < 0.0001"
else:
    p_str = f"P = {p_val:.4e}"

# 绘制 SCI 显著性支架
y_max = df_plot[feature_name].max()
y_min = df_plot[feature_name].min()
y_range = y_max - y_min
plt.ylim(y_min - y_range * 0.05, y_max + y_range * 0.18)

x1, x2 = 0, 1 
bracket_y = y_max + y_range * 0.04  
bracket_h = y_range * 0.03          
plt.plot([x1, x1, x2, x2], [bracket_y, bracket_y+bracket_h, bracket_y+bracket_h, bracket_y], lw=1.5, color='black')
plt.text((x1+x2)*.5, bracket_y + bracket_h + y_range*0.01, p_str, ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')

sns.despine()
plt.tight_layout()

img_out = "m6A_TrueCore_vs_Dropout_GC_Content.png"
plt.savefig(img_out, dpi=300, bbox_inches='tight')
print(f"   ✅ 真·三尺度核心对比图已成功保存至 '{img_out}'")