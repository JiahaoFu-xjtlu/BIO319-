import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# =====================================================================
# 第一部分：数据加载与预处理 (针对 51bp 尺度)
# =====================================================================
print("⏳ 1/3 正在加载 51bp 全量数据集...")
# 【严格注意】：请确保你的 51bp 特征矩阵文件名与此处一致
input_file_51bp = "51bp_Final_Feature_Matrix_for_ML.csv" 
try:
    df_51bp = pd.read_csv(input_file_51bp)
except FileNotFoundError:
    raise FileNotFoundError(f"❌ 找不到文件 {input_file_51bp}，请检查当前路径或修改文件名！")

target_features = ['H3K4me3', 'Cluster_ID']
for feat in target_features:
    if feat not in df_51bp.columns:
        raise ValueError(f"❌ 缺失核心特征: {feat}！")

# 清理缺失值与异常值
df_51bp = df_51bp.replace([np.inf, -np.inf], np.nan)
df_clean = df_51bp.dropna(subset=target_features).copy()

# 将目标和特征分类化 (严格保持与 21bp 完全一致的命名规则)
df_clean['Group'] = np.where(df_clean['Cluster_ID'] == 0, 'Cluster 0 (Island)', 'Others (Background)')
df_clean['H3K4me3_Status'] = np.where(df_clean['H3K4me3'] > 0.5, 'Positive (+)', 'Negative (-)')

print(f"✅ 51bp 数据准备完毕。总样本数: {len(df_clean)}")

# =====================================================================
# 第二部分：Fisher 精确检验计算
# =====================================================================
print("\n🧮 2/3 正在构建 2x2 列联表并执行 Fisher's Exact Test...")

c0_pos = len(df_clean[(df_clean['Group'] == 'Cluster 0 (Island)') & (df_clean['H3K4me3_Status'] == 'Positive (+)')])
c0_neg = len(df_clean[(df_clean['Group'] == 'Cluster 0 (Island)') & (df_clean['H3K4me3_Status'] == 'Negative (-)')])
others_pos = len(df_clean[(df_clean['Group'] == 'Others (Background)') & (df_clean['H3K4me3_Status'] == 'Positive (+)')])
others_neg = len(df_clean[(df_clean['Group'] == 'Others (Background)') & (df_clean['H3K4me3_Status'] == 'Negative (-)')])

contingency_table = [[c0_pos, c0_neg], [others_pos, others_neg]]
print(f"   📊 51bp 列联表分布: Cluster 0 (+:{c0_pos}, -:{c0_neg}) | Others (+:{others_pos}, -:{others_neg})")

# 执行 Fisher 精确检验 (生物学富集分析的统计金标准)
oddsratio, p_val_h3 = stats.fisher_exact(contingency_table, alternative='two-sided')

# =====================================================================
# 第三部分：绘制极简阳性率柱状图 (完美复刻 21bp 视觉 UI)
# =====================================================================
print("\n🎨 3/3 正在绘制 51bp H3K4me3 阳性率柱状图...")

# 计算百分比
h3k4me3_summary = df_clean.groupby(['Group', 'H3K4me3_Status']).size().unstack(fill_value=0)
h3k4me3_pct = h3k4me3_summary.div(h3k4me3_summary.sum(axis=1), axis=0) * 100

sns.set_theme(style="ticks", context="paper", font_scale=1.2)
plt.figure(figsize=(7, 6))

# 【UI 锁定】：强制使用图1的颜色和顺序
target_order = ['Cluster 0 (Island)', 'Others (Background)']
plot_data = h3k4me3_pct['Positive (+)'].reindex(target_order)

ax1 = plot_data.plot(
    kind='bar', 
    color=['#e74c3c', '#95a5a6'], # 绝对锁定：红灰配色
    edgecolor='black',            # 黑色描边增强学术感
    linewidth=1.2,
    rot=0                         # X轴标签水平
)

# 动态修改标题，明确指出这是 51bp 的结果
plt.title("Proportion of H3K4me3 Positive Sequences (51bp)", fontsize=16, fontweight='bold', pad=15)
plt.ylabel("Percentage (%)", fontsize=14)
plt.xlabel("") # 底部标签已自解释
plt.ylim(0, 105) # 锁死 Y 轴比例，确保与 21bp 图在视觉上直接可比

# P-value 科学计数法格式化 (处理底层的极端下溢问题)
if p_val_h3 == 0.0 or p_val_h3 < 1e-300:
    p_str_h3 = "P < 1.0e-300 "
elif p_val_h3 < 0.0001:
    p_str_h3 = "P < 0.0001 "
else:
    p_str_h3 = f"P = {p_val_h3:.4e}"

# 标注 P-value
plt.text(0.5, 98, p_str_h3, ha='center', va='top', fontsize=14, fontweight='bold')

# 标注具体的百分比数值在柱子上方
for p in ax1.patches:
    ax1.annotate(f"{p.get_height():.1f}%", 
                 (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='bottom', fontsize=12, xytext=(0, 5), textcoords='offset points')

sns.despine()
plt.tight_layout()

# 保存图片，命名带有 51bp 标识
img_out = "m6A_51bp_Cluster0_H3K4me3_Bar_Fisher.png"
plt.savefig(img_out, dpi=300)
print(f"   ✅ 51bp 柱状图已保存至 '{img_out}'")
plt.close()
print("\n🎉 51bp 分析完美收官！")