library(GenomicRanges)
library(Biostrings)
library(BSgenome.Hsapiens.UCSC.hg38)

print("⏳ 正在加载 m6A 位点数据...")
m6a_gr <- readRDS("D:/FYP/m6A_OrthogonallyValidatedSites_Combined_hg38.rds")

# 【修改点 1】：将截取窗口宽度修改为 21bp
print("✂️ 正在截取 21bp 上下文窗口...")
window_21mer <- resize(m6a_gr, width = 101, fix = "center")

m6a_seqs <- getSeq(BSgenome.Hsapiens.UCSC.hg38, window_21mer)

# 转化为字符矩阵
seq_matrix <- do.call(rbind, strsplit(as.character(m6a_seqs), ""))
N <- nrow(seq_matrix)

# 【修改点 2】：将序列长度 L 修改为 21
L <- 101

print("🧮 正在构建纯净的 4D One-hot 矩阵...")
# 初始化全 0 矩阵。遇到 'N' 会天然保持 [0, 0, 0, 0] 的完美状态
onehot_matrix <- matrix(0, nrow = N, ncol = L * 4)

# 严谨的 4 通道赋值
onehot_matrix[, seq(1, L*4, by = 4)] <- (seq_matrix == "A") * 1
onehot_matrix[, seq(2, L*4, by = 4)] <- (seq_matrix == "C") * 1
onehot_matrix[, seq(3, L*4, by = 4)] <- (seq_matrix == "G") * 1
onehot_matrix[, seq(4, L*4, by = 4)] <- (seq_matrix == "T" | seq_matrix == "U") * 1

# 【修改点 3】：修改输出文件名，防止覆盖 101bp 的数据
output_path <- "D:/FYP/101bp_vae_new/m6A_101bp_onehot_features.csv"
write.table(onehot_matrix, 
            file = output_path, 
            sep = ",", row.names = FALSE, col.names = FALSE)

print(sprintf("🎉 大功告成！21bp 数据已保存至: %s", output_path))
print(sprintf("📊 当前矩阵形状: %d 行, %d 列 (无缝对接 PyTorch 输入层！)", N, L*4))