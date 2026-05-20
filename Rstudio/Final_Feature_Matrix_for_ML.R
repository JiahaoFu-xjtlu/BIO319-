library(data.table)

print("1. 读取你的坐标文件...")
my_data <- fread("D:/FYP/101bp_vae_new/Coordinates_101bp_with_Clusters.csv")

print("2. 读取 m6AConquer 大数据库 (由于你刚才读过，这次可能只要几秒钟)...")
big_data <- fread("D:/FYP/m6Aconquer_omicsFeaturesV2_hg38.csv", 
                  quote = "", fill = TRUE, showProgress = TRUE)

print("3. 【关键修复】正在清洗暗号上的双引号...")
# (1) 清洗列名
colnames(big_data) <- gsub("\"", "", colnames(big_data))

# (2) 清洗数据里的双引号 (让 "-" 变回 -，让 "chr1" 变回 chr1)
big_data[, seqnames := gsub("\"", "", seqnames)]
big_data[, strand := gsub("\"", "", strand)]

# (3) 确保起止点被当作纯数字对待
big_data[, start := as.integer(start)]
big_data[, end := as.integer(end)]

print("4. 再次执行合并...")
# 这次暗号绝对一模一样了！
merged_data <- merge(my_data, big_data, 
                     by = c("seqnames", "start", "end", "strand"),
                     all = FALSE)

print(paste("✅ 成功！这次真正合并到了", nrow(merged_data), "行数据！"))

# 5. 覆盖之前那个错误的空表
fwrite(merged_data, "D:/FYP/101bp_vae_new/Final_Feature_Matrix_for_ML.csv")
print("🎉 覆盖保存成功，赶紧去看看吧！")

# 按照你原始的索引号重新排序
merged_data <- merged_data[order(Original_Row_Index)]