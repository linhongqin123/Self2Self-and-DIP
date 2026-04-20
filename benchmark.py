import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import bm3d
import pandas as pd
from skimage import io, img_as_float, color, metrics
from tqdm import tqdm

# ==========================================
# 1. 网络结构定义区 (三大金刚集结)
# ==========================================

# [1] DIP 使用的极简 U-Net
class DIP_UNet(nn.Module):
    def __init__(self):
        super(DIP_UNet, self).__init__()
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(64 + 32, 32)
        self.out = nn.Sequential(nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid())

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

# [2] Self2Self 使用的带 Dropout 的 U-Net
class DropoutUNet(nn.Module):
    def __init__(self, drop_rate=0.3):
        super(DropoutUNet, self).__init__()
        self.drop_rate = drop_rate
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(64 + 32, 32)
        self.out = nn.Sequential(nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid())

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(self.drop_rate),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(self.drop_rate)
        )
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.dec1(torch.cat([self.up1(e2), e1], dim=1))
        return self.out(d1)

# [3] N2N 使用的 U-Net (精准复刻你上周的代码)
class N2N_UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=32):
        super(N2N_UNet, self).__init__()
        self.enc1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = self._block(features * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = self._block(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = self._block(features * 2, features)
        self.out = nn.Conv2d(features, out_channels, kernel_size=1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.upconv2(b)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)

# ==========================================
# 2. 算法推理封装函数
# ==========================================

def run_bm3d(noisy_img, sigma):
    return bm3d.bm3d(noisy_img, sigma_psd=sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES)

def run_dip(noisy_tensor, clean_img, device, iters=800):
    net = DIP_UNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    mse_loss = nn.MSELoss()
    z = torch.rand(1, 1, clean_img.shape[0], clean_img.shape[1]).to(device)
    best_psnr, best_img = 0, None
    for i in range(iters):
        optimizer.zero_grad()
        out = net(z)
        loss = mse_loss(out, noisy_tensor)
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            out_np = out.detach().cpu().squeeze().numpy()
            current_psnr = metrics.peak_signal_noise_ratio(clean_img, out_np, data_range=1.0)
            if current_psnr > best_psnr:
                best_psnr = current_psnr
                best_img = out_np.copy()
    return best_img

def run_s2s(noisy_tensor, device, iters=800, ensembles=30):
    net = DropoutUNet(drop_rate=0.3).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    p = 0.3
    net.train()
    for _ in range(iters):
        optimizer.zero_grad()
        mask = torch.bernoulli(torch.full_like(noisy_tensor, 1 - p)).to(device)
        out = net(noisy_tensor * mask)
        loss = torch.sum(((out - noisy_tensor) ** 2) * (1 - mask)) / torch.sum(1 - mask)
        loss.backward()
        optimizer.step()
    preds = []
    net.train()
    with torch.no_grad():
        for _ in range(ensembles):
            mask = torch.bernoulli(torch.full_like(noisy_tensor, 1 - p)).to(device)
            preds.append(net(noisy_tensor * mask))
    return torch.mean(torch.stack(preds), dim=0).cpu().squeeze().numpy()

def run_n2n(noisy_tensor, device, model_path='n2n_weights.pth'):
    if not os.path.exists(model_path):
        return None
    net = N2N_UNet().to(device)
    try:
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.eval()
        with torch.no_grad():
            out = net(noisy_tensor)
        # N2N 原始代码没有 Sigmoid，需要手动 clip 保证像素在 0-1 之间
        return np.clip(out.cpu().squeeze().numpy(), 0, 1)
    except Exception as e:
        print(f"\n⚠️ 加载 N2N 权重报错: {e}。跳过 N2N。")
        return None

# ==========================================
# 3. 自动化评测主流程
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 启动终极自动化评测引擎 | 使用设备: {device}")
    
    n2n_weight_path = 'n2n_weights.pth'
    if os.path.exists(n2n_weight_path):
        print(f"✅ 成功检测到 N2N 权重 ({n2n_weight_path})，四大算法已集结！")
    else:
        print(f"⚠️ 未检测到 '{n2n_weight_path}'，本次测试将跳过 N2N。")
    
    sigmas = [15, 25, 35, 50]
    
    dataset_path = 'Set12'
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs('benchmark_results', exist_ok=True)
    
    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.bmp'):
        image_paths.extend(glob.glob(os.path.join(dataset_path, ext)))
        
    if not image_paths:
        print(f"\n❌ 警告：未在 {dataset_path} 文件夹中找到任何图片！请放入测试图片后再运行。")
        return
        
    print(f"\n📂 发现 {len(image_paths)} 张测试图像。开始执行全矩阵测试...")
    results_records = []

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        print(f"\n--- 正在处理图像: {img_name} ---")
        
        raw_img = io.imread(img_path)
        if len(raw_img.shape) == 3:
            raw_img = color.rgb2gray(raw_img)
        clean_img = img_as_float(raw_img)
        
        for sigma_val in sigmas:
            print(f"  [测试级别] 噪声 σ = {sigma_val} ...", end="", flush=True)
            sigma = sigma_val / 255.0
            noisy_img = clean_img + np.random.normal(0, sigma, clean_img.shape)
            noisy_img = np.clip(noisy_img, 0, 1)
            noisy_tensor = torch.from_numpy(noisy_img).float().unsqueeze(0).unsqueeze(0).to(device)
            
            res_bm3d = run_bm3d(noisy_img, sigma)
            res_dip = run_dip(noisy_tensor, clean_img, device, iters=800)
            res_s2s = run_s2s(noisy_tensor, device, iters=800, ensembles=30)
            res_n2n = run_n2n(noisy_tensor, device, model_path=n2n_weight_path)
            
            current_results = {'Noisy': noisy_img, 'BM3D': res_bm3d, 'DIP': res_dip, 'Self2Self': res_s2s}
            if res_n2n is not None:
                current_results['N2N'] = res_n2n
            
            print(" 完成！")
            
            num_cols = len(current_results) + 1 
            fig, axes = plt.subplots(1, num_cols, figsize=(3.5 * num_cols, 4))
            axes[0].imshow(clean_img, cmap='gray'); axes[0].set_title("Clean Image")
            
            for idx, (algo_name, out_img) in enumerate(current_results.items()):
                psnr_val = metrics.peak_signal_noise_ratio(clean_img, out_img, data_range=1.0)
                ssim_val = metrics.structural_similarity(clean_img, out_img, data_range=1.0)
                
                if algo_name != 'Noisy':
                    results_records.append({
                        'Image': img_name, 'Sigma': sigma_val, 'Algorithm': algo_name,
                        'PSNR': psnr_val, 'SSIM': ssim_val
                    })
                
                axes[idx+1].imshow(out_img, cmap='gray')
                axes[idx+1].set_title(f"{algo_name}\nPSNR: {psnr_val:.2f} dB")
            
            for ax in axes: ax.axis('off')
            plt.tight_layout()
            plt.savefig(f'benchmark_results/{img_name}_sigma{sigma_val}.png', dpi=150)
            plt.close()

    # ==========================================
    # 4. 生成报告级 Markdown 表格
    # ==========================================
    print("\n✅ 所有测试完成！正在生成数据报告...")
    df = pd.DataFrame(results_records)
    
    summary_df = df.groupby(['Sigma', 'Algorithm'])[['PSNR', 'SSIM']].mean().reset_index()
    summary_df['PSNR'] = summary_df['PSNR'].apply(lambda x: f"{x:.2f}")
    summary_df['SSIM'] = summary_df['SSIM'].apply(lambda x: f"{x:.4f}")
    
    pivot_psnr = summary_df.pivot(index='Algorithm', columns='Sigma', values='PSNR')
    
    print("\n" + "="*60)
    print("🏆 实验结果：各噪声水平下的 平均 PSNR (dB) 对比")
    print("="*60)
    print(pivot_psnr.to_markdown())
    print("="*60)
    
    summary_df.to_csv('benchmark_results/final_metrics.csv', index=False)
    print("\n💾 详细数据表已保存至: benchmark_results/final_metrics.csv")
    print("🖼️ 所有的对比组图已保存至: benchmark_results/ 文件夹")

if __name__ == '__main__':
    main()