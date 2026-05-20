# ==========================================
# 1. 加载核心生信包
# ==========================================
suppressMessages(library(BSgenome.Hsapiens.UCSC.hg38))
suppressMessages(library(GenomicRanges))

# ==========================================
# 2. 配置文件路径 (请确认路径准确)
# ==========================================
# 你的原始 m6A 坐标 RDS 文件
rds_file <- "D:/FYP/m6A_OrthogonallyValidatedSites_Combined_hg38.rds"

# 你的聚类结果 CSV 文件 (包含索引和 Cluster_ID)
# TODO: 请把这里的名字换成你实际的聚类结果文件！
cluster_file <- "D:/FYP/21BP(VAE)/m6A_Sequence_Cluster_Mapping.csv" 

# 最终要生成的画图文件
output_file <- "D:/FYP/21BP(VAE)/21bp_Cluster_Results.csv"

# ==========================================
# 3. 加载数据并提取序列
# ==========================================
cat("Step 1: Loading coordinates and cluster results...\n")
m6a_sites <- readRDS(rds_file)
cluster_df <- read.csv(cluster_file)

# 将 1bp 的 m6A 中心坐标向两端扩展，形成完整的 21bp 窗口
# (上游 10bp + m6A位点本身 1bp + 下游 10bp = 21bp)
cat("Step 2: Resizing coordinates to 21bp window...\n")
m6a_21bp <- resize(m6a_sites, width = 21, fix = "center")

# 使用 hg38 提取真实的 ATCG 序列
cat("Step 3: Extracting high-fidelity sequences from hg38...\n")
seqs <- getSeq(BSgenome.Hsapiens.UCSC.hg38, m6a_21bp)

# 将 DNAStringSet 转换为普通的字符串向量
seqs_char <- as.character(seqs)

# ==========================================
# 4. 安全检查与数据拼接
# ==========================================
cat("Step 4: Performing safety check on row counts...\n")

# 【绝对防御】：必须保证序列条数和你的聚类结果行数严丝合缝
if(length(seqs_char) == nrow(cluster_df)) {
  cat(" -> SUCCESS: Match confirmed! Both have", length(seqs_char), "rows.\n")
  
  # 将真实的序列作为新的一列，插入到聚类结果的数据框中
  cluster_df$Sequence <- seqs_char
  
  # 保存为 Python 画图所需的文件
  write.csv(cluster_df, output_file, row.names = FALSE)
  cat("\n🎉 ALL DONE! File successfully saved to:\n", output_file, "\n")
  cat("You can now safely run the Python K-mer plotting script!\n")
  
} else {
  # 如果行数对不上，立刻报错，阻止错误数据生成
  stop(paste("ERROR: Row mismatch detected!\n",
             "Extracted Sequences:", length(seqs_char), "rows.\n",
             "Cluster DataFrame:", nrow(cluster_df), "rows.\n",
             "Please check if your cluster CSV was filtered or shuffled!"))
}