import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float, metrics
from tqdm import tqdm
import os

# ==========================================
# 1. 定义带有 Dropout 的 U-Net (S2S 的灵魂)
# ==========================================
class DropoutUNet(nn.Module):
    def __init__(self, drop_rate=0.3):
        super(DropoutUNet, self).__init__()
        self.drop_rate = drop_rate
        
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.pool = nn.MaxPool2d(2)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(64 + 32, 32)
        
        self.out = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def conv_block(self, in_c, out_c):
        # 相比 DIP，这里强行加入了 Dropout 层
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(self.drop_rate), # 核心！
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(self.drop_rate)  # 核心！
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.dec1(torch.cat([self.up1(e2), e1], dim=1))
        return self.out(d1)

# ==========================================
# 2. 数据准备与加噪 (与刚才保持完全一致)
# ==========================================
clean_img = img_as_float(data.camera())
sigma = 25 / 255.0  
np.random.seed(42) # 固定随机种子以保证实验可重复
noisy_img = clean_img + np.random.normal(0, sigma, clean_img.shape)
noisy_img = np.clip(noisy_img, 0, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noisy_tensor = torch.from_numpy(noisy_img).float().unsqueeze(0).unsqueeze(0).to(device)

print(f"使用的设备: {device}")
print(f"初始带噪图像 PSNR: {metrics.peak_signal_noise_ratio(clean_img, noisy_img, data_range=1.0):.2f} dB")

# ==========================================
# 3. Self2Self 训练逻辑
# ==========================================
net = DropoutUNet(drop_rate=0.3).to(device)
optimizer = optim.Adam(net.parameters(), lr=0.005)

num_iter = 1500 # S2S 训练次数 (它不容易像 DIP 那样严重过拟合)
p = 0.3 # 遮挡掉 30% 的像素

print("\n开始 Self2Self 训练...")
# 注意：我们一直保持 net.train() 模式，让 Dropout 一直生效
net.train() 
for i in tqdm(range(num_iter)):
    optimizer.zero_grad()
    
    # 1. 伯努利采样生成 Mask (1表示保留，0表示遮挡)
    mask = torch.bernoulli(torch.full_like(noisy_tensor, 1 - p)).to(device)
    
    # 2. 用 Mask 遮挡带噪图像，作为输入
    masked_input = noisy_tensor * mask
    
    # 3. 网络预测完整图像
    out = net(masked_input)
    
    # 4. 【灵魂机制】Loss 仅仅计算那些被遮挡掉的像素点！
    # 网络被迫通过周围的像素去猜测被遮挡的像素，从而学到了去噪
    loss = torch.sum(((out - noisy_tensor) ** 2) * (1 - mask)) / torch.sum(1 - mask)
    
    loss.backward()
    optimizer.step()

# ==========================================
# 4. 集成推理 (Ensemble Inference)
# ==========================================
print("\n训练完成，开始集成推理 (多次预测取平均)...")
num_ensembles = 50 # 推理 50 次
preds = []

# 推理时必须保持 train 模式！这是 S2S 论文强调的，以保留随机性
net.train() 

with torch.no_grad(): # 停止计算梯度，节省显存
    for _ in tqdm(range(num_ensembles)):
        # 推理时也加 Mask 和 Dropout，产生差异化的预测
        mask = torch.bernoulli(torch.full_like(noisy_tensor, 1 - p)).to(device)
        masked_input = noisy_tensor * mask
        pred = net(masked_input)
        preds.append(pred)

# 把 50 次“各不相同”的猜测取平均，得到极其平滑的结果
final_denoised_tensor = torch.mean(torch.stack(preds), dim=0)
final_denoised = final_denoised_tensor.cpu().squeeze().numpy()

# 计算 S2S 最终 PSNR
s2s_psnr = metrics.peak_signal_noise_ratio(clean_img, final_denoised, data_range=1.0)
print(f"\nSelf2Self 最终去噪 PSNR: {s2s_psnr:.2f} dB")

# ==========================================
# 5. 可视化与保存
# ==========================================
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(clean_img, cmap='gray')
axes[0].set_title('Clean Image')
axes[1].imshow(noisy_img, cmap='gray')
axes[1].set_title('Noisy Image (sigma=25)')

# 展示一张被遮挡的中间图，让老师知道你彻底懂了 S2S 原理
sample_masked = (noisy_tensor * mask).cpu().squeeze().numpy()
axes[2].imshow(sample_masked, cmap='gray')
axes[2].set_title('S2S Masked Input (30% dropped)')

axes[3].imshow(final_denoised, cmap='gray')
axes[3].set_title(f'Self2Self Result (PSNR: {s2s_psnr:.2f})')

for ax in axes:
    ax.axis('off')

os.makedirs('results', exist_ok=True)
save_path = 'results/s2s_result.png'
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"\n结果已保存至: {os.path.abspath(save_path)}")