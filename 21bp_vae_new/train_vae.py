import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# =====================================================================
# 第一部分：数据加载 (完全适配纯净 4D 格式)
# =====================================================================

class m6A_Dataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file, header=None, dtype="float32", engine='python')
        raw_data = torch.tensor(df.values)
        # 【修改点 1】：将原本的 21, 4 严格对齐（这里你原来写的很对，保留）
        self.data = raw_data.view(-1, 21, 4)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# =====================================================================
# 第二部分：VAE 模型架构 (加入中间层，严格对齐 4D)
# =====================================================================

class m6A_VAE(nn.Module):
    def __init__(self, bottleneck_dim=32):
        super(m6A_VAE, self).__init__()
        
        # 1. 卷积编码器
        self.encoder_cnn = nn.Sequential(
            # 【修改点 2】：in_channels 从 5 改为真实的 4 (A, C, G, T)
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten() # 展平后维度为 32 * 21 = 672
        )
        
        # 【修改点 3】：加入你提议的中间过渡层，保持架构跨尺度一致性
        self.fc_encoder = nn.Sequential(
            nn.Linear(32 * 21, 256), # 672 -> 256 平滑过渡
            nn.ReLU()
        )
        
        # 连接到均值和方差
        self.fc_mu = nn.Linear(256, bottleneck_dim)      
        self.fc_logvar = nn.Linear(256, bottleneck_dim)  
        
        # 2. 解码器全连接部分（严格对称地还原）
        self.decoder_linear = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32 * 21), # 256 -> 672
            nn.ReLU()
        )
        
        # 3. 反卷积解码器
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            # 【修改点 4】：out_channels 从 5 改回真实的 4
            nn.ConvTranspose1d(in_channels=16, out_channels=4, kernel_size=3, padding=1) 
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        x_cnn = x.permute(0, 2, 1) # [Batch, 4, 21]
        
        hidden = self.encoder_cnn(x_cnn) # [Batch, 672]
        hidden = self.fc_encoder(hidden) # [Batch, 256] 经过新的中间层
        
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        
        dec = self.decoder_linear(z)     # [Batch, 672]
        dec = dec.view(-1, 32, 21)       # 变回图片格式 [Batch, 32, 21]
        out_cnn = self.decoder_conv(dec) # [Batch, 4, 21]
        
        return out_cnn, mu, logvar


# =====================================================================
# 第三部分：双重裁判系统 (Loss Function)
# =====================================================================
def vae_loss_function(recon_x, x, mu, logvar, beta):
    # recon_x 现在的形状是 [Batch, 4, 21]
    # x 的形状是 [Batch, 21, 4]
    
    # 将 [Batch, 21, 4] 的 One-hot 转为索引
    target_indices = torch.argmax(x, dim=2) 
    
    # 计算交叉熵，完美适配 PyTorch 标准
    CE_loss = F.cross_entropy(recon_x, target_indices, reduction='sum')
    
    # 计算 KL 散度
    KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = CE_loss + beta * KLD_loss
    
    return total_loss, CE_loss, KLD_loss

# =====================================================================
# 第四部分：正式训练大循环 (Training Loop)
# =====================================================================
if __name__ == '__main__':
    # 1. 准备数据和模型
    csv_file_path = "m6A_21bp_onehot_features.csv" # 确保这里是你的正确文件名
    dataset = m6A_Dataset(csv_file_path)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # 如果电脑有显卡就用显卡，没有就用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 开始训练！当前使用设备: {device}")
    
    model = m6A_VAE(bottleneck_dim=16).to(device)
    
    # 2. 设置优化器 (Adam 是最常用的学习算法)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. 训练超参数
    epochs = 30           # 总共训练 30 轮
    warmup_epochs = 15    # 前 15 轮用来“温水煮青蛙”
    
    # 4. 开启训练
    for epoch in range(1, epochs + 1):
        model.train() # 告诉模型：现在是训练时间，开启重参数化（加噪声）魔法！
        
        train_loss = 0
        train_ce = 0
        train_kld = 0
        
        # 【核心技巧：KL 退火 (温水煮青蛙)】
        # 随着 epoch 增加，beta 从 0 慢慢涨到 1
        if epoch <= warmup_epochs:
            beta = epoch / warmup_epochs
        else:
            beta = 1.0
            
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            
            # 清空上一轮的梯度
            optimizer.zero_grad()
            
            # 前向传播：提取特征 -> 采样 -> 重构
            recon_batch, mu, logvar = model(data)
            
            # 让裁判打分
            loss, ce, kld = vae_loss_function(recon_batch, data, mu, logvar, beta)
            
            # 反向传播：根据得分修改网络参数
            loss.backward()
            optimizer.step()
            
            # 记录分数值用于打印
            train_loss += loss.item()
            train_ce += ce.item()
            train_kld += kld.item()
            
        # 计算平均每条序列的 Loss
        avg_loss = train_loss / len(dataset)
        avg_ce = train_ce / len(dataset)
        avg_kld = train_kld / len(dataset)
        
        # 打印日志
        print(f"Epoch [{epoch}/{epochs}] | Beta: {beta:.2f} | Total Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, KLD: {avg_kld:.4f})")

    # 5. 训练结束，保存模型权重
    torch.save(model.state_dict(), "m6A_vae_weights.pth")
    print("\n🎉 训练圆满完成！模型权重已保存为 'm6A_vae_weights.pth'") 

# =====================================================================
# 第五部分：采摘果实 (特征提取与保存)
# 目标：关闭随机噪声，将 135300 条序列转化为 16 维质心坐标，保存为 CSV。
# =====================================================================

if __name__ == '__main__':
    # 1. 准备原始数据
    csv_file_path = "m6A_21bp_onehot_features.csv" 
    dataset = m6A_Dataset(csv_file_path)
    
    extract_loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 开始提取高维特征！当前使用设备: {device}")
    
    # 2. 召唤我们刚刚训练好的模型大脑
    # 【修复点 2】：必须显式指定 bottleneck_dim=16，确保与训练时保存的权重结构完全一致
    model = m6A_VAE(bottleneck_dim=16).to(device)
    
    model.load_state_dict(torch.load("m6A_vae_weights.pth"))
    model.eval() 
    
    all_latent_features = [] 
    
    with torch.no_grad():
        for batch_idx, data in enumerate(extract_loader):
            data = data.to(device)
            
            # 手动把数据送进 Encoder
            x_cnn = data.permute(0, 2, 1)
            
            # 经过卷积层，输出 [Batch, 672]
            hidden = model.encoder_cnn(x_cnn)
            
            # 【修复点 1】：必须经过中间层！将 672 降维到 256
            hidden = model.fc_encoder(hidden)
            
            # 拿到最纯净的靶心坐标 mu (输入 256，输出 16)
            mu = model.fc_mu(hidden)
            
            all_latent_features.append(mu.cpu().numpy())
            
            if batch_idx % 20 == 0:
                print(f"⏳ 正在提取特征... 已处理 {batch_idx * 1024} 条序列")

    # 4. 拼接矩阵
    final_features_matrix = np.vstack(all_latent_features)
    print(f"\n✅ 特征提取完毕！特征矩阵最终形状: {final_features_matrix.shape}")
    
    # 5. 保存
    # 【细节优化】：建议在文件名中明确标出 21bp 和 16d，防止和 51bp 的特征混淆
    output_filename = "m6A_21bp_latent_features_16d.csv"
    
    df_out = pd.DataFrame(final_features_matrix)
    df_out.to_csv(output_filename, index=False, header=False)
    
    print(f"🎉 大功告成！纯净的 VAE 特征已安全存入: {output_filename}")