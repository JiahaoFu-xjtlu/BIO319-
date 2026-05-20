library(dplyr)

# 1. 读取 RDS (此时加载进来的是一个 GRanges 对象)
# 请替换为你的实际文件名
gr_obj <- readRDS("D:/FYP/m6A_OrthogonallyValidatedSites_Combined_hg38.rds")

# 2. 关键步骤：将 GRanges 对象转换为普通的 data.frame 表格
orig_df <- as.data.frame(gr_obj)

# (可选) 你可以运行 head(orig_df) 看一下，
# 此时 <Rle> 格式消失了，变成了带有 "seqnames", "start", "end", "width", "strand" 表头的普通表格

# 3. 读取你的聚类结果 CSV
cluster_df <- read.csv("D:/FYP/101bp_vae_new/m6A_101bp_Sequence_Cluster_Mapping.csv")

# 4. 为转换后的原始数据添加行索引 (从 0 开始，与 Python 对齐)
orig_df$Original_Row_Index <- 0:(nrow(orig_df) - 1)

# 5. 根据索引完美合并
merged_df <- inner_join(orig_df, cluster_df, by = "Original_Row_Index")

# 6. 导出为跨平台的 CSV 文件
write.csv(merged_df, "D:/FYP/101bp_vae_new/Coordinates_101bp_with_Clusters.csv", row.names = FALSE)