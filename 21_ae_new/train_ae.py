import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# =====================================================================
# 第一部分：数据加载 (与 VAE 绝对保持一致)
# =====================================================================

class m6A_Dataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file, header=None, dtype="float32", engine='python')
        raw_data = torch.tensor(df.values)
        self.data = raw_data.view(-1, 21, 4)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# =====================================================================
# 第二部分：标准 AE 模型架构 (剥离概率采样，保留等量参数)
# =====================================================================

class m6A_AE(nn.Module):
    def __init__(self, bottleneck_dim=16):
        super(m6A_AE, self).__init__()
        
        # 1. 卷积编码器 (参数与 VAE 100% 对齐)
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten() # 展平后维度为 32 * 21 = 672
        )
        
        # 2. 中间过渡层
        self.fc_encoder = nn.Sequential(
            nn.Linear(32 * 21, 256), 
            nn.ReLU()
        )
        
        # 【修改点 A】：剥离 mu 和 logvar，替换为直接的瓶颈层映射
        self.fc_bottleneck = nn.Linear(256, bottleneck_dim)      
        
        # 3. 解码器全连接部分
        self.decoder_linear = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32 * 21), 
            nn.ReLU()
        )
        
        # 4. 反卷积解码器
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=4, kernel_size=3, padding=1) 
        )

    # 【修改点 B】：删除了 reparameterize 函数，AE 是确定性映射

    def forward(self, x):
        x_cnn = x.permute(0, 2, 1) # [Batch, 4, 21]
        
        hidden = self.encoder_cnn(x_cnn) # [Batch, 672]
        hidden = self.fc_encoder(hidden) # [Batch, 256]
        
        # 【修改点 C】：直接输出高维特征 z
        z = self.fc_bottleneck(hidden)   # [Batch, 16]
        
        dec = self.decoder_linear(z)     # [Batch, 672]
        dec = dec.view(-1, 32, 21)       # 变回图片格式 [Batch, 32, 21]
        out_cnn = self.decoder_conv(dec) # [Batch, 4, 21]
        
        return out_cnn, z


# =====================================================================
# 第三部分：标准重构损失 (剥离 KL 散度)
# =====================================================================
def ae_loss_function(recon_x, x):
    # 将 [Batch, 21, 4] 的 One-hot 转为索引
    target_indices = torch.argmax(x, dim=2) 
    
    # 【修改点 D】：仅保留交叉熵重构误差，没有任何正则化约束
    CE_loss = F.cross_entropy(recon_x, target_indices, reduction='sum')
    
    return CE_loss

# =====================================================================
# 第四部分：正式训练大循环 (移除退火魔法)
# =====================================================================
if __name__ == '__main__':
    csv_file_path = "m6A_21bp_onehot_features.csv" 
    dataset = m6A_Dataset(csv_file_path)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 开始训练 AE！当前使用设备: {device}")
    
    # 召唤 AE 模型
    model = m6A_AE(bottleneck_dim=16).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 30 
    
    for epoch in range(1, epochs + 1):
        model.train() 
        train_loss = 0
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播：提取特征 -> 重构
            recon_batch, z = model(data)
            
            # 让裁判打分 (仅看重构质量)
            loss = ae_loss_function(recon_batch, data)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_loss = train_loss / len(dataset)
        
        # 打印日志
        print(f"Epoch [{epoch}/{epochs}] | Reconstruction Loss (CE): {avg_loss:.4f}")

    # 保存 AE 权重，严格区分文件名
    torch.save(model.state_dict(), "m6A_ae_weights.pth")
    print("\n🎉 AE 训练圆满完成！模型权重已保存为 'm6A_ae_weights.pth'") 

# =====================================================================
# 第五部分：采摘果实 (特征提取与保存)
# =====================================================================

if __name__ == '__main__':
    csv_file_path = "m6A_21bp_onehot_features.csv" 
    dataset = m6A_Dataset(csv_file_path)
    extract_loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 开始提取 AE 高维特征！当前使用设备: {device}")
    
    # 加载 AE 模型
    model = m6A_AE(bottleneck_dim=16).to(device)
    model.load_state_dict(torch.load("m6A_ae_weights.pth"))
    model.eval() 
    
    all_latent_features = [] 
    
    with torch.no_grad():
        for batch_idx, data in enumerate(extract_loader):
            data = data.to(device)
            
            x_cnn = data.permute(0, 2, 1)
            hidden = model.encoder_cnn(x_cnn)
            hidden = model.fc_encoder(hidden)
            
            # 【修改点 E】：直接获取绝对质心 z
            z = model.fc_bottleneck(hidden)
            
            all_latent_features.append(z.cpu().numpy())
            
            if batch_idx % 20 == 0:
                print(f"⏳ 正在提取 AE 特征... 已处理 {batch_idx * 1024} 条序列")

    final_features_matrix = np.vstack(all_latent_features)
    print(f"\n✅ AE 特征提取完毕！特征矩阵最终形状: {final_features_matrix.shape}")
    
    # 极其关键的文件名区分
    output_filename = "m6A_21bp_latent_features_16d.csv"
    
    df_out = pd.DataFrame(final_features_matrix)
    df_out.to_csv(output_filename, index=False, header=False)
    
    print(f"🎉 大功告成！纯净的 AE 特征已安全存入: {output_filename}")