import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SCINet01']

class IlluminationBranch(nn.Module):
    def __init__(self, channels):
        super(IlluminationBranch, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        feat = self.relu(self.conv1(x))
        L_raw = self.sigmoid(self.conv2(feat))
        return L_raw

class ReflectanceBranch(nn.Module):
    def __init__(self, channels):
        super(ReflectanceBranch, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        r = self.relu(self.conv1(x))
        r = self.conv2(r)
        return r

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, H, W = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)

class EnhancedIllumination(nn.Module):
    def __init__(self):
        super(EnhancedIllumination, self).__init__()
        self.log_gamma = nn.Parameter(torch.zeros(1))  # Learnable gamma parameter
        self.se = SEBlock(1, reduction=4)  # SE Block for single-channel illumination map
    
    def forward(self, L_raw):
        gamma = torch.exp(self.log_gamma)
        L_gamma = torch.pow(L_raw + 1e-6, gamma)
        L_enhanced = self.se(L_gamma)
        return L_enhanced

class GuidedDiffusionBlock(nn.Module):
    def __init__(self, channels, time_dim=64):
        super(GuidedDiffusionBlock, self).__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels)
        )
        self.depthwise_conv = nn.Conv2d(channels + 3, channels + 3, kernel_size=3, padding=1, groups=channels + 3)
        self.pointwise_conv = nn.Conv2d(channels + 3, channels, kernel_size=1)
        self.channel_attention = ChannelAttention(channels)
        self.diffusion_coeff = nn.Parameter(torch.tensor(0.1))
    
    def compute_gradient(self, x):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)
        return torch.cat([grad_x, grad_y], dim=1)
    
    def forward(self, reflectance, illumination, t_emb):
        B = reflectance.size(0)
        t_feat = self.time_mlp(t_emb).view(B, -1, 1, 1)
        x = reflectance + t_feat
        
        illum_grad = self.compute_gradient(illumination)
        condition = torch.cat([x, illumination, illum_grad], dim=1)
        
        out = self.depthwise_conv(condition)
        out = self.pointwise_conv(out)
        
        out = self.channel_attention(out)
        
        reflectance = reflectance + self.diffusion_coeff * out
        return reflectance

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, _, _ = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)

class SCINet01(nn.Module):
    def __init__(self, channels=64, num_diffusion_steps=3, time_dim=64):
        super(SCINet01, self).__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.illumination_branch = IlluminationBranch(channels)
        self.reflectance_branch = ReflectanceBranch(channels)
        self.enhanced_illumination = EnhancedIllumination()  # Enhanced illumination module
        self.diffusion_steps = nn.ModuleList([
            GuidedDiffusionBlock(channels, time_dim) for _ in range(num_diffusion_steps)
        ])
        self.time_embed = nn.Parameter(torch.randn(1, time_dim))
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels + 1, 3, kernel_size=3, padding=1),  # Adjusted input channels
            nn.Sigmoid()
        )
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        features = self.shared_conv(x)
        illum_raw = self.illumination_branch(features)
        reflectance = self.reflectance_branch(features)
        
        # Enhance illumination
        illum_enhanced = self.enhanced_illumination(illum_raw)
        
        time_emb = self.time_embed.repeat(x.size(0), 1)
        for step in self.diffusion_steps:
            reflectance = step(reflectance, illum_raw, time_emb)  # Use raw illumination as condition
        
        # Fusion of enhanced illumination and enhanced reflectance
        fused = torch.cat([reflectance, illum_enhanced], dim=1)
        enhanced = self.out_conv(fused)
        return torch.clamp(enhanced, 0, 1)

if __name__ == "__main__":
    model = SCINet01()
    input_img = torch.randn(1, 3, 256, 256)
    output_img = model(input_img)
    print(output_img.shape)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# __all__ = ['SCINet01']


# class IlluminationBranch(nn.Module):
#     def __init__(self, channels):
#         super(IlluminationBranch, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         illum = self.sigmoid(self.conv2(x))
#         return illum


# class ReflectanceBranch(nn.Module):
#     def __init__(self, channels):
#         super(ReflectanceBranch, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

#     def forward(self, x):
#         r = self.relu(self.conv1(x))
#         r = self.conv2(r)
#         return r


# class ChannelAttention(nn.Module):
#     # reduction 通道数的压缩比例
#     def __init__(self, channels, reduction=16):
#         super(ChannelAttention, self).__init__()
#         # 自适应平均池化层，将输入特征图池化为 1x1 的大小。这一步将特征图的空间维度压缩为 1x1，保留通道信息。
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)

#         # fc: 一个由三个全连接层组成的序列，用于计算通道注意力权重。
#         self.fc = nn.Sequential(
#             # 第一个全连接层将通道数压缩为 channels // reduction，减少计算量。
#             nn.Linear(channels, channels // reduction, bias=False),
#             nn.ReLU(),
#             # 第二个全连接层将通道数恢复为原始大小。
#             nn.Linear(channels // reduction, channels, bias=False),
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         # B: 批量大小。  C: 输入特征图的通道数。
#         B, C, _, _ = x.size()
#         # avg_pool(x): 对输入特征图 x 应用自适应平均池化，得到大小为 1x1 的特征图。
#         # .view(B, C): 将池化后的特征图重新调整为形状 (B, C)，以便输入到全连接层。
#         y = self.avg_pool(x).view(B, C)
#         # self.fc(y): 通过全连接层计算通道注意力权重。
#         # .view(B, C, 1, 1): 将注意力权重重新调整为形状 (B, C, 1, 1)，以便与输入特征图进行逐通道乘法。
#         y = self.fc(y).view(B, C, 1, 1)
#         # y.expand_as(x): 将注意力权重扩展到与输入特征图 x 相同的形状。
#         # x * y.expand_as(x): 将注意力权重逐通道应用于输入特征图，增强重要通道的特征，抑制不重要通道的特征。
#         return x * y.expand_as(x)

# # 通过引导扩散过程细化反射成分
# class GuidedDiffusionBlock(nn.Module):
#     def __init__(self, channels, time_dim=64):
#         super(GuidedDiffusionBlock, self).__init__()
#         # time_mlp: 一个多层感知机（MLP），用于处理时间嵌入。
#         self.time_mlp = nn.Sequential(
#             nn.Linear(time_dim, channels * 2),
#             nn.GELU(),
#             nn.Linear(channels * 2, channels)
#         )

#         # depthwise_conv: 深度卷积层(groups=输入通道数)，对每个通道独立进行卷积，保留通道间的独立性。深度卷积对每个通道独立进行卷积操作。
#         self.depthwise_conv = nn.Conv2d(channels + 3, channels + 3, kernel_size=3, padding=1, groups=channels + 3)
#         # pointwise_conv: 逐点卷积层，将通道数从 channels + 3 减少到 channels，使用 1x1 卷积核。
#         self.pointwise_conv = nn.Conv2d(channels + 3, channels, kernel_size=1)
#         self.channel_attention = ChannelAttention(channels)
#         # diffusion_coeff: 一个可学习的参数，控制扩散的强度，默认值为 0.1。
#         self.diffusion_coeff = nn.Parameter(torch.tensor(0.1))

#     def compute_gradient(self, x):
#         # Sobel 滤波器，分别用于计算水平和垂直方向的梯度。
#         sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
#         sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
#         # 使用 Sobel 滤波器对输入特征图 x 进行卷积，得到水平和垂直方向的梯度。
#         grad_x = F.conv2d(x, sobel_x, padding=1)
#         grad_y = F.conv2d(x, sobel_y, padding=1)
#         # 将水平和垂直梯度拼接成一个两通道的梯度图。
#         return torch.cat([grad_x, grad_y], dim=1)

#     def forward(self, reflectance, illumination, t_emb):
#         B = reflectance.size(0)
#         # t_feat: 时间特征，通过时间嵌入 t_emb 经过 time_mlp 处理后，调整为与反射特征相同的形状。
#         t_feat = self.time_mlp(t_emb).view(B, -1, 1, 1)
#         # x: 将时间特征加到反射特征上。
#         x = reflectance + t_feat

#         # illum_grad: 计算光照的梯度。
#         illum_grad = self.compute_gradient(illumination)
#         # condition: 将反射特征、光照和梯度拼接在一起，形成条件特征。
#         condition = torch.cat([x, illumination, illum_grad], dim=1)

#         # 条件特征经过深度卷积和逐点卷积处理。
#         out = self.depthwise_conv(condition)
#         out = self.pointwise_conv(out)
#         out = self.channel_attention(out)

#         reflectance = reflectance + self.diffusion_coeff * out
#         return reflectance


# class SCINet01(nn.Module):
#     def __init__(self, channels=64, num_diffusion_steps=3, time_dim=64):
#         super(SCINet01, self).__init__()
#         # shared_conv: 共享卷积层，提取初始特征。
#         self.shared_conv = nn.Sequential(
#             nn.Conv2d(3, channels, kernel_size=3, padding=1),
#             nn.ReLU()
#         )
#         # illumination_branch 和 reflectance_branch: 分解光照和反射。
#         self.illumination_branch = IlluminationBranch(channels)
#         self.reflectance_branch = ReflectanceBranch(channels)

#         # diffusion_steps: 多个引导扩散块。
#         self.diffusion_steps = nn.ModuleList([
#             GuidedDiffusionBlock(channels, time_dim) for _ in range(num_diffusion_steps)
#         ])
#         # time_embed: 时间嵌入参数。
#         self.time_embed = nn.Parameter(torch.randn(1, time_dim))
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(channels, 3, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )

#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)

#     def forward(self, x):
#         features = self.shared_conv(x)
#         illumination = self.illumination_branch(features)
#         reflectance = self.reflectance_branch(features)

#         # 时间嵌入，重复用于每个样本。 x.size(0): 获取输入张量 x 的批量大小
#         # repeat(x.size(0), 1): 在第一个维度（批量大小）上重复
#         time_emb = self.time_embed.repeat(x.size(0), 1)
#         for step in self.diffusion_steps:
#             reflectance = step(reflectance, illumination, time_emb)

#         enhanced = self.out_conv(reflectance * illumination)
#         return torch.clamp(enhanced, 0, 1)


# if __name__ == "__main__":
#     model = SCINet01()
#     input_img = torch.randn(1, 3, 256, 256)
#     output_img = model(input_img)
#     print(output_img.shape)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import warnings
#
# __all__ = ['SCINet01']
#
#
# def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
#     # 简单实现：使用截断正态分布初始化
#     with torch.no_grad():
#         return tensor.normal_(mean, std).clamp_(min=a, max=b)
#
#
# class PreNorm(nn.Module):
#     def __init__(self, channels, fn):
#         super(PreNorm, self).__init__()
#         self.norm = nn.LayerNorm([channels, 1, 1])
#         self.fn = fn
#
#     def forward(self, x):
#         # x shape: [B, C, H, W]，对每个通道归一化（先 reshape 后归一化）
#         B, C, H, W = x.shape
#         x = x.view(B, C, -1)
#         x = self.norm(x).view(B, C, H, W)
#         return self.fn(x)
#
#
# # ------------------------
# # 照明估计模块：参考 RetinexFormer 中 Illumination_Estimator
# # ------------------------
# class IlluminationEstimator(nn.Module):
#     def __init__(self, channels, mid_channels):
#         super(IlluminationEstimator, self).__init__()
#         self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=True)
#         self.depth_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=5, padding=2, bias=True, groups=channels)
#         self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # 输入 x: [B, C, H, W]
#         x1 = self.conv1(x)
#         x2 = self.depth_conv(x1)
#         illum = self.sigmoid(self.conv2(x2))
#         return x2, illum  # 返回中间特征和照明图
#
#
# # ------------------------
# # 反射分支：在原 ReflectanceBranch 基础上增加残差连接和非线性处理
# # ------------------------
# class ReflectanceBranch(nn.Module):
#     def __init__(self, channels):
#         super(ReflectanceBranch, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.gelu = nn.GELU()
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         residual = x
#         x = self.gelu(self.conv1(x))
#         x = self.conv2(x)
#         return x + residual
#
#
# # ------------------------
# # 引导注意力模块：借鉴 IG_MSA 与多尺度融合思想
# # ------------------------
# class GuidedAttentionBlock(nn.Module):
#     def __init__(self, channels, time_dim=64):
#         super(GuidedAttentionBlock, self).__init__()
#         # 时间嵌入 MLP
#         self.time_mlp = nn.Sequential(
#             nn.Linear(time_dim, channels * 2),
#             nn.GELU(),
#             nn.Linear(channels * 2, channels)
#         )
#         self.register_parameter("time_embed", nn.Parameter(torch.randn(1, time_dim)))
#
#         # 多尺度卷积（引入 illumination 信息时，通道数+2）
#         self.conv3x3 = nn.Conv2d(channels + 2, channels, kernel_size=3, padding=1)
#         self.conv5x5 = nn.Conv2d(channels + 2, channels, kernel_size=5, padding=2)
#         self.conv7x7 = nn.Conv2d(channels + 2, channels, kernel_size=7, padding=3)
#
#         # 注意力融合层：1×1 卷积获得多尺度特征权重
#         self.attn_conv = nn.Sequential(
#             nn.Conv2d(channels * 3, channels * 3, kernel_size=1),
#             nn.Sigmoid()
#         )
#         # 输出融合卷积
#         self.out_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#
#     @staticmethod
#     def compute_gradient(x):
#         # 计算水平和垂直梯度
#         grad_x = F.pad(x[:, :, :, :-1] - x[:, :, :, 1:], (0, 1, 0, 0))
#         grad_y = F.pad(x[:, :, :-1, :] - x[:, :, 1:, :], (0, 0, 0, 1))
#         return torch.abs(grad_x) + torch.abs(grad_y)
#
#     def forward(self, reflectance, illumination):
#         B = reflectance.size(0)
#         # 计算时间嵌入
#         t_emb = self.time_mlp(self.time_embed.repeat(B, 1))
#         t_emb = t_emb.view(B, -1, 1, 1)
#         # 给反射特征添加调制
#         x = reflectance + t_emb
#
#         # 计算 illumination 梯度
#         illum_grad = self.compute_gradient(illumination)
#         # 将 x, illumination, illum_grad 融合
#         condition = torch.cat([x, illumination, illum_grad], dim=1)
#
#         # 多尺度特征提取
#         feat1 = self.conv3x3(condition)
#         feat2 = self.conv5x5(condition)
#         feat3 = self.conv7x7(condition)
#
#         # 融合多尺度特征
#         cat_feats = torch.cat([feat1, feat2, feat3], dim=1)
#         attn_weights = self.attn_conv(cat_feats)
#         c = reflectance.size(1)
#         w1, w2, w3 = torch.split(attn_weights, c, dim=1)
#         fused = w1 * feat1 + w2 * feat2 + w3 * feat3
#         fused = self.out_conv(fused)
#
#         # 残差更新（扩散步长）
#         out = reflectance + 0.1 * fused
#         return out
#
#
# # ------------------------
# # 主网络：SCINet01
# # ------------------------
# class SCINet01(nn.Module):
#     def __init__(self, channels=64, num_diffusion_steps=3, time_dim=64):
#         super(SCINet01, self).__init__()
#         # 共享特征提取
#         self.shared_conv = nn.Sequential(
#             nn.Conv2d(3, channels, kernel_size=3, padding=1),
#             nn.ReLU()
#         )
#         # 改进后的照明估计器
#         self.illumination_estimator = IlluminationEstimator(channels, mid_channels=channels)
#         # 改进后的反射分支
#         self.reflectance_branch = ReflectanceBranch(channels)
#
#         # Diffusion 模块：使用多个引导注意力块
#         self.diffusion_steps = nn.ModuleList([
#             GuidedAttentionBlock(channels, time_dim=time_dim) for _ in range(num_diffusion_steps)
#         ])
#
#         # 重构模块
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(channels, 3, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )
#
#         self.initialize_weights()
#
#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 trunc_normal_(m.weight, std=0.02)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 trunc_normal_(m.weight, std=0.02)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#
#     def forward(self, x):
#         # 共享特征提取
#         features = self.shared_conv(x)
#         # 照明估计
#         illum_fea, illumination = self.illumination_estimator(features)
#         # 反射分支
#         reflectance = self.reflectance_branch(features)
#
#         # Diffusion 过程：引导注意力块迭代更新反射分量
#         for block in self.diffusion_steps:
#             reflectance = block(reflectance, illumination)
#
#         # Retinex 融合：将增强后的反射分量与照明指导相乘
#         out = reflectance * illumination
#         enhanced = self.out_conv(out)
#         return torch.clamp(enhanced, 0, 1)


# # Example usage
# if __name__ == "__main__":
#     model = SCINet01()
#     input_img = torch.randn(1, 3, 256, 256)
#     output_img = model(input_img)
#     print(output_img.shape)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import cv2
# import numpy as np
#
# __all__ = ['SCINet01']
#
#
# def tv_loss(x):
#     loss_h = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
#     loss_v = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
#     return loss_h + loss_v
#
#
# class IlluminationBranch(nn.Module):
#     def __init__(self, channels):
#         super(IlluminationBranch, self).__init__()
#         self.conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         illum = self.sigmoid(self.conv(x))
#         return illum
#
#
# class ReflectanceBranch(nn.Module):
#     def __init__(self, channels):
#         super(ReflectanceBranch, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         r = self.relu(self.conv1(x))
#         r = self.conv2(r)
#         return r
#
#
# class GuidedDiffusionBlock(nn.Module):
#     def __init__(self, channels, time_dim=64):
#         super(GuidedDiffusionBlock, self).__init__()
#         self.time_mlp = nn.Sequential(
#             nn.Linear(time_dim, channels * 2),
#             nn.GELU(),
#             nn.Linear(channels * 2, channels)
#         )
#         self.conv3x3 = nn.Conv2d(channels + 2, channels, kernel_size=3, padding=1)
#         self.conv5x5 = nn.Conv2d(channels + 2, channels, kernel_size=5, padding=2)
#         self.conv7x7 = nn.Conv2d(channels + 2, channels, kernel_size=7, padding=3)
#         self.attention = nn.Sequential(
#             nn.Conv2d(channels * 3, channels * 3, kernel_size=1),
#             nn.Sigmoid()
#         )
#         self.out_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#
#         self.diffusion_coef = nn.Parameter(torch.tensor(0.1))
#
#         self.time_embed = nn.Parameter(torch.randn(1, time_dim))
#
#     @staticmethod
#     def compute_gradient(x):
#         grad_x = F.pad(x[:, :, :, :-1] - x[:, :, :, 1:], (0, 1, 0, 0))
#         grad_y = F.pad(x[:, :, :-1, :] - x[:, :, 1:, :], (0, 0, 0, 1))
#         return torch.abs(grad_x) + torch.abs(grad_y)
#
#     def forward(self, reflectance, illumination):
#         B = reflectance.size(0)
#
#         t_emb = self.time_mlp(self.time_embed.repeat(B, 1))
#         t_emb = t_emb.view(B, -1, 1, 1)
#         x = reflectance + t_emb
#
#         illum_grad = self.compute_gradient(illumination)
#
#         condition = torch.cat([x, illumination, illum_grad], dim=1)
#
#         out1 = self.conv3x3(condition)
#         out2 = self.conv5x5(condition)
#         out3 = self.conv7x7(condition)
#
#         cat_feats = torch.cat([out1, out2, out3], dim=1)
#         attn_weights = self.attention(cat_feats)
#         c = reflectance.size(1)
#         w1, w2, w3 = torch.split(attn_weights, c, dim=1)
#         fused = w1 * out1 + w2 * out2 + w3 * out3
#         fused = self.out_conv(fused)
#
#         out = reflectance + self.diffusion_coef * fused
#         return out
#
#
# class SCINet01(nn.Module):
#     def __init__(self, channels=64, num_diffusion_steps=3, time_dim=64):
#         super(SCINet01, self).__init__()
#
#         self.shared_conv = nn.Sequential(
#             nn.Conv2d(3, channels, kernel_size=3, padding=1),
#             nn.ReLU()
#         )
#         self.illumination_branch = IlluminationBranch(channels)
#         self.reflectance_branch = ReflectanceBranch(channels)
#
#         self.diffusion_steps = nn.ModuleList([
#             GuidedDiffusionBlock(channels, time_dim) for _ in range(num_diffusion_steps)
#         ])
#
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(channels, 3, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )
#         self.initialize_weights()
#
#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#
#     def forward(self, x):
#         features = self.shared_conv(x)
#         illumination = self.illumination_branch(features)
#         reflectance = self.reflectance_branch(features)
#         for step in self.diffusion_steps:
#             reflectance = step(reflectance, illumination)
#
#         out = reflectance * illumination
#         enhanced = self.out_conv(out)
#         return torch.clamp(enhanced, 0, 1)
#
#     def test_time_optimization(self, x, num_iters=200, lr=1e-3, lambda_tv=0.01):
#         optimized_model = SCINetZeroIG()
#         optimized_model.load_state_dict(self.state_dict())
#         optimized_model.train()
#
#         optimizer = torch.optim.Adam(optimized_model.parameters(), lr=lr)
#         for _ in range(num_iters):
#             optimizer.zero_grad()
#             enhanced = optimized_model(x)
#
#             rec_loss = F.l1_loss(enhanced, x)
#             features = optimized_model.shared_conv(x)
#             illumination = optimized_model.illumination_branch(features)
#             reg_loss = tv_loss(illumination)
#             loss = rec_loss + lambda_tv * reg_loss
#             loss.backward()
#             optimizer.step()
#
#         optimized_model.eval()
#         with torch.no_grad():
#             result = optimized_model(x)
#         return torch.clamp(result, 0, 1)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# __all__ = ['SCINet01']
#
#
# class IlluminationAwareEnhancement(nn.Module):
#     """
#     光照感知特征增强模块
#     根据光照分量的多尺度梯度信息动态调整特征权重。
#     """
#
#     def __init__(self, channels):
#         super(IlluminationAwareEnhancement, self).__init__()
#         self.conv1 = nn.Conv2d(1, channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(1, channels, kernel_size=5, padding=2)
#         self.conv3 = nn.Conv2d(1, channels, kernel_size=7, padding=3)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, feature, illumination):
#         grad_illumination = self.compute_gradient(illumination)
#         weight1 = self.sigmoid(self.conv1(grad_illumination))
#         weight2 = self.sigmoid(self.conv2(grad_illumination))
#         weight3 = self.sigmoid(self.conv3(grad_illumination))
#         enhanced_feature = feature * (weight1 + weight2 + weight3) / 3
#         return enhanced_feature
#
#     @staticmethod
#     def compute_gradient(x):
#         grad_x = F.pad(x[:, :, :, :-1] - x[:, :, :, 1:], (0, 1, 0, 0))
#         grad_y = F.pad(x[:, :, :-1, :] - x[:, :, 1:, :], (0, 0, 0, 1))
#         return torch.abs(grad_x) + torch.abs(grad_y)
#
#
# class ReflectionEnhancement(nn.Module):
#     """
#     反射特征增强模块
#     通过反射分量的多尺度高频信息增强特征细节。
#     """
#
#     def __init__(self, channels):
#         super(ReflectionEnhancement, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
#         self.conv3 = nn.Conv2d(channels, channels, kernel_size=7, padding=3)
#         self.relu = nn.ReLU()
#
#     def forward(self, feature, reflectance):
#         high_freq1 = self.conv1(reflectance)
#         high_freq2 = self.conv2(reflectance)
#         high_freq3 = self.conv3(reflectance)
#         enhanced_feature = feature + self.relu((high_freq1 + high_freq2 + high_freq3) / 3)
#         return enhanced_feature
#
#
# class FeatureEnhancementBlock(nn.Module):
#     def __init__(self, channels):
#         super(FeatureEnhancementBlock, self).__init__()
#         # 多尺度特征提取
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
#         self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4)
#
#         # 光照引导的注意力机制
#         self.attention = nn.Sequential(
#             nn.Conv2d(channels * 3 + 1, channels * 3, kernel_size=1),  # +1 for illumination guidance
#             nn.Sigmoid()
#         )
#         self.relu = nn.ReLU()
#
#     def forward(self, x, illumination):
#         # 计算光照梯度
#         grad_illum = IlluminationAwareEnhancement.compute_gradient(illumination)
#
#         # 多尺度特征提取
#         conv1_out = self.relu(self.conv1(x))
#         conv2_out = self.relu(self.conv2(x))
#         conv3_out = self.relu(self.conv3(x))
#
#         # 拼接多尺度特征和光照梯度
#         fused_features = torch.cat([conv1_out, conv2_out, conv3_out, grad_illum], dim=1)
#
#         # 生成光照引导的注意力权重
#         attention_weights = self.attention(fused_features)
#         a1, a2, a3 = torch.split(attention_weights, x.size(1), dim=1)
#
#         # 加权特征融合
#         enhanced_features = a1 * conv1_out + a2 * conv2_out + a3 * conv3_out
#         return enhanced_features
#
#
# class LightweightDiffusion(nn.Module):
#     """轻量级扩散增强模块，保留原有双分支结构"""
#
#     def __init__(self, channels, time_dim=64):
#         super().__init__()
#         self.time_mlp = nn.Sequential(
#             nn.Linear(time_dim, channels * 2),
#             nn.GELU(),
#             nn.Linear(channels * 2, channels)
#         )
#
#         # 扩散增强卷积组
#         self.diff_conv1 = nn.Sequential(
#             nn.Conv2d(channels + 1, channels, 3, padding=1),  # 包含光照条件
#             nn.InstanceNorm2d(channels),
#             nn.GELU(),
#             nn.Conv2d(channels, channels, 3, padding=1)
#         )
#         self.diff_conv2 = nn.Sequential(
#             nn.Conv2d(channels + 1, channels, 5, padding=2),  # 包含光照条件
#             nn.InstanceNorm2d(channels),
#             nn.GELU(),
#             nn.Conv2d(channels, channels, 5, padding=2)
#         )
#         self.diff_conv3 = nn.Sequential(
#             nn.Conv2d(channels + 1, channels, 7, padding=3),  # 包含光照条件
#             nn.InstanceNorm2d(channels),
#             nn.GELU(),
#             nn.Conv2d(channels, channels, 7, padding=3)
#         )
#
#         # 时间嵌入参数
#         self.time_embed = nn.Parameter(torch.randn(1, time_dim))
#
#     def forward(self, x, illumination):
#         """
#         x: 特征图 [B, C, H, W]
#         illumination: 光照分量 [B, 1, H, W]
#         """
#         B = x.size(0)
#
#         # 生成时间条件 (共享参数)
#         t_emb = self.time_mlp(self.time_embed.repeat(B, 1))  # [B, C]
#         t_emb = t_emb.view(B, -1, 1, 1)  # [B, C, 1, 1]
#
#         # 将时间条件添加到特征图
#         conditioned_x = x + t_emb  # [B, C, H, W]
#
#         # 拼接光照条件
#         condition = torch.cat([conditioned_x, illumination], dim=1)  # [B, C+1, H, W]
#
#         # 扩散增强处理
#         noise_pred1 = self.diff_conv1(condition)
#         noise_pred2 = self.diff_conv2(condition)
#         noise_pred3 = self.diff_conv3(condition)
#         noise_pred = (noise_pred1 + noise_pred2 + noise_pred3) / 3
#         return x + 0.1 * noise_pred  # 控制增强强度
#
#
# class SCINet01(nn.Module):
#     """集成轻量扩散增强的SCI网络，保留双分支结构"""
#
#     def __init__(self, channels=64, layers=3):
#         super().__init__()
#         # 原始组件保持不变
#         self.in_conv = nn.Sequential(
#             nn.Conv2d(3, channels, 3, padding=1),
#             nn.ReLU()
#         )
#         self.illumination_branch = nn.Conv2d(channels, 1, 3, padding=1)
#         self.reflectance_branch = nn.Conv2d(channels, channels, 3, padding=1)
#         self.illumination_aware = IlluminationAwareEnhancement(channels)
#         self.reflection_enhance = ReflectionEnhancement(channels)
#         self.feb = FeatureEnhancementBlock(channels)
#
#         # 新增轻量扩散模块
#         self.diffusion_enhance = LightweightDiffusion(channels)
#
#         # 增强后的处理层
#         self.blocks = nn.ModuleList([self.conv_block(channels) for _ in range(layers)])
#
#         # 输出层保持原有结构
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(channels, 3, 3, padding=1),
#             nn.Sigmoid()
#         )
#
#     def conv_block(self, channels):
#         return nn.Sequential(
#             nn.Conv2d(channels, channels, 3, padding=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         # 保持原有前向传播流程
#         fea = self.in_conv(x)
#         illumination = self.illumination_branch(fea)
#         reflectance = self.reflectance_branch(fea)
#
#         # 原有特征增强
#         fea = self.illumination_aware(fea, illumination)
#         fea = self.reflection_enhance(fea, reflectance)
#         fea = self.feb(fea, illumination)
#
#         # 新增扩散增强步骤
#         fea = self.diffusion_enhance(fea, illumination)  # 添加扩散增强
#
#         # 后续处理
#         for block in self.blocks:
#             fea = fea + block(fea)
#
#         # 最终重建
#         enhanced = self.out_conv(fea)
#
#         # 保持Retinex约束
#         illumination_3ch = illumination.repeat(1, 3, 1, 1)
#         reflectance_3ch = reflectance[:, :3, :, :].contiguous()
#         enhanced = illumination_3ch * enhanced + reflectance_3ch
#         return torch.clamp(enhanced, 0, 1)
# #
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# __all__ = ['SCINet01']
#
# class IlluminationAwareEnhancement(nn.Module):
#     """
#     光照感知特征增强模块
#     根据光照分量的多尺度梯度信息动态调整特征权重。
#     """
#     def __init__(self, channels):
#         super(IlluminationAwareEnhancement, self).__init__()
#         self.conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, feature, illumination):
#         grad_illumination = self.compute_gradient(illumination)
#         weight = self.sigmoid(self.conv(grad_illumination))
#         enhanced_feature = feature * weight
#         return enhanced_feature
#     @staticmethod
#     def compute_gradient(x):
#         grad_x = F.pad(x[:, :, :, :-1] - x[:, :, :, 1:], (0, 1, 0, 0))
#         grad_y = F.pad(x[:, :, :-1, :] - x[:, :, 1:, :], (0, 0, 0, 1))
#         return torch.abs(grad_x) + torch.abs(grad_y)
#
#
# class ReflectionEnhancement(nn.Module):
#     def __init__(self, channels):
#         super(ReflectionEnhancement, self).__init__()
#         self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#
#     def forward(self, feature, reflectance):
#         high_freq = self.conv(reflectance)
#         enhanced_feature = feature + self.relu(high_freq)
#         return enhanced_feature
#
# class FeatureEnhancementBlock(nn.Module):
#     def __init__(self, channels):
#         super(FeatureEnhancementBlock, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
#         self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4)
#         self.attention = nn.Sequential(
#             nn.Conv2d(channels * 3 + 1, channels * 3, kernel_size=1),  # +1 for illumination guidance
#             nn.Sigmoid()
#         )
#         self.relu = nn.ReLU()
#
#     def forward(self, x, illumination):
#         grad_illum = IlluminationAwareEnhancement.compute_gradient(illumination)
#         conv1_out = self.relu(self.conv1(x))
#         conv2_out = self.relu(self.conv2(x))
#         conv3_out = self.relu(self.conv3(x))
#         fused_features = torch.cat([conv1_out, conv2_out, conv3_out, grad_illum], dim=1)
#         attention_weights = self.attention(fused_features)
#         a1, a2, a3 = torch.split(attention_weights, x.size(1), dim=1)
#         enhanced_features = a1 * conv1_out + a2 * conv2_out + a3 * conv3_out
#         return enhanced_features
#
#
# class InvertibleDiffusion(nn.Module):
#     def __init__(self, channels, time_dim=64):
#         super().__init__()
#         self.time_mlp = nn.Sequential(
#             nn.Linear(time_dim, channels * 2),
#             nn.GELU(),
#             nn.Linear(channels * 2, channels)
#         )
#         self.diff_conv = nn.Sequential(
#             nn.Conv2d(channels + 1, channels, 3, padding=1),
#             nn.InstanceNorm2d(channels),
#             nn.GELU(),
#             nn.Conv2d(channels, channels, 3, padding=1)
#         )
#         self.time_embed = nn.Parameter(torch.randn(1, time_dim))
#
#     def forward(self, x, illumination):
#         B = x.size(0)
#         t_emb = self.time_mlp(self.time_embed.repeat(B, 1)).view(B, -1, 1, 1)
#         conditioned_x = x + t_emb
#         condition = torch.cat([conditioned_x, illumination], dim=1)
#         noise_pred = self.diff_conv(condition)
#         return x + 0.1 * noise_pred
#
# class SCINet01(nn.Module):
#
#     def __init__(self, channels=64, layers=3):
#         super().__init__()
#         self.in_conv = nn.Sequential(
#             nn.Conv2d(3, channels, 3, padding=1),
#             nn.ReLU()
#         )
#         self.illumination_branch = nn.Conv2d(channels, 1, 3, padding=1)
#         self.reflectance_branch = nn.Conv2d(channels, channels, 3, padding=1)
#         self.illumination_aware = IlluminationAwareEnhancement(channels)
#         self.reflection_enhance = ReflectionEnhancement(channels)
#         self.feb = FeatureEnhancementBlock(channels)
#         self.diffusion_enhance = InvertibleDiffusion(channels)
#         self.blocks = nn.ModuleList([self.conv_block(channels) for _ in range(layers)])
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(channels, 3, 3, padding=1),
#             nn.Sigmoid()
#         )
#
#     def conv_block(self, channels):
#         return nn.Sequential(
#             nn.Conv2d(channels, channels, 3, padding=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         fea = self.in_conv(x)
#         illumination = self.illumination_branch(fea)
#         reflectance = self.reflectance_branch(fea)
#         fea = self.illumination_aware(fea, illumination)
#         fea = self.reflection_enhance(fea, reflectance)
#         fea = self.feb(fea, illumination)
#         fea = self.diffusion_enhance(fea, illumination)
#         for block in self.blocks:
#             fea = fea + block(fea)
#         enhanced = self.out_conv(fea)
#         illumination_3ch = illumination.repeat(1, 3, 1, 1)
#         reflectance_3ch = reflectance[:, :3, :, :].contiguous()
#         enhanced = illumination_3ch * enhanced + reflectance_3ch
#         return torch.clamp(enhanced, 0, 1)


# SCINet04
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# __all__ = ['SCINet01']
#
# class IlluminationAwareEnhancement(nn.Module):
#     """
#     光照感知特征增强模块
#     根据光照分量的梯度信息动态调整特征权重。
#     """
#
#     def __init__(self, channels):
#         super(IlluminationAwareEnhancement, self).__init__()
#         self.conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)  # 输入通道数改为 1
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, feature, illumination):
#         grad_illumination = self.compute_gradient(illumination)
#         weight = self.sigmoid(self.conv(grad_illumination))
#         enhanced_feature = feature * weight
#         return enhanced_feature
#
#     @staticmethod
#     def compute_gradient(x):
#         grad_x = F.pad(x[:, :, :, :-1] - x[:, :, :, 1:], (0, 1, 0, 0))
#         grad_y = F.pad(x[:, :, :-1, :] - x[:, :, 1:, :], (0, 0, 0, 1))
#         return torch.abs(grad_x) + torch.abs(grad_y)
#
#
# class ReflectionEnhancement(nn.Module):
#     """
#     反射特征增强模块
#     通过反射分量的高频信息增强特征细节。
#     """
#
#     def __init__(self, channels):
#         super(ReflectionEnhancement, self).__init__()
#         self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#
#     def forward(self, feature, reflectance):
#         high_freq = self.conv(reflectance)
#         enhanced_feature = feature + self.relu(high_freq)
#         return enhanced_feature
#
#
# class FeatureEnhancementBlock(nn.Module):
#     def __init__(self, channels):
#         super(FeatureEnhancementBlock, self).__init__()
#         # 多尺度特征提取
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
#         self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4)
#
#         # 光照引导的注意力机制
#         self.attention = nn.Sequential(
#             nn.Conv2d(channels * 3 + 1, channels * 3, kernel_size=1),  # +1 for illumination guidance
#             nn.Sigmoid()
#         )
#         self.relu = nn.ReLU()
#
#     def forward(self, x, illumination):
#         # 计算光照梯度
#         grad_illum = IlluminationAwareEnhancement.compute_gradient(illumination)
#
#         # 多尺度特征提取
#         conv1_out = self.relu(self.conv1(x))
#         conv2_out = self.relu(self.conv2(x))
#         conv3_out = self.relu(self.conv3(x))
#
#         # 拼接多尺度特征和光照梯度
#         fused_features = torch.cat([conv1_out, conv2_out, conv3_out, grad_illum], dim=1)
#
#         # 生成光照引导的注意力权重
#         attention_weights = self.attention(fused_features)
#         a1, a2, a3 = torch.split(attention_weights, x.size(1), dim=1)
#
#         # 加权特征融合
#         enhanced_features = a1 * conv1_out + a2 * conv2_out + a3 * conv3_out
#         return enhanced_features
#
#
# class LightweightDiffusion(nn.Module):
#     """轻量级扩散增强模块，保留原有双分支结构"""
#
#     def __init__(self, channels, time_dim=64):
#         super().__init__()
#         self.time_mlp = nn.Sequential(
#             nn.Linear(time_dim, channels * 2),
#             nn.GELU(),
#             nn.Linear(channels * 2, channels)
#         )
#
#         # 扩散增强卷积组
#         self.diff_conv = nn.Sequential(
#             nn.Conv2d(channels + 1, channels, 3, padding=1),  # 包含光照条件
#             nn.InstanceNorm2d(channels),
#             nn.GELU(),
#             nn.Conv2d(channels, channels, 3, padding=1)
#         )
#
#         # 时间嵌入参数
#         self.time_embed = nn.Parameter(torch.randn(1, time_dim))
#
#     def forward(self, x, illumination):
#         """
#         x: 特征图 [B, C, H, W]
#         illumination: 光照分量 [B, 1, H, W]
#         """
#         B = x.size(0)
#
#         # 生成时间条件 (共享参数)
#         t_emb = self.time_mlp(self.time_embed.repeat(B, 1))  # [B, C]
#         t_emb = t_emb.view(B, -1, 1, 1)  # [B, C, 1, 1]
#
#         # 将时间条件添加到特征图
#         conditioned_x = x + t_emb  # [B, C, H, W]
#
#         # 拼接光照条件
#         condition = torch.cat([conditioned_x, illumination], dim=1)  # [B, C+1, H, W]
#
#         # 扩散增强处理
#         noise_pred = self.diff_conv(condition)
#         return x + 0.1 * noise_pred  # 控制增强强度
#
#
# class SCINet01(nn.Module):
#     """集成轻量扩散增强的SCI网络，保留双分支结构"""
#
#     def __init__(self, channels=64, layers=3):
#         super().__init__()
#         # 原始组件保持不变
#         self.in_conv = nn.Sequential(
#             nn.Conv2d(3, channels, 3, padding=1),
#             nn.ReLU()
#         )
#         self.illumination_branch = nn.Conv2d(channels, 1, 3, padding=1)
#         self.reflectance_branch = nn.Conv2d(channels, channels, 3, padding=1)
#         self.illumination_aware = IlluminationAwareEnhancement(channels)
#         self.reflection_enhance = ReflectionEnhancement(channels)
#         self.feb = FeatureEnhancementBlock(channels)
#
#         # 新增轻量扩散模块
#         self.diffusion_enhance = LightweightDiffusion(channels)
#
#         # 增强后的处理层
#         self.blocks = nn.ModuleList([self.conv_block(channels) for _ in range(layers)])
#
#         # 输出层保持原有结构
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(channels, 3, 3, padding=1),
#             nn.Sigmoid()
#         )
#
#     def conv_block(self, channels):
#         return nn.Sequential(
#             nn.Conv2d(channels, channels, 3, padding=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         # 保持原有前向传播流程
#         fea = self.in_conv(x)
#         illumination = self.illumination_branch(fea)
#         reflectance = self.reflectance_branch(fea)
#
#         # 原有特征增强
#         fea = self.illumination_aware(fea, illumination)
#         fea = self.reflection_enhance(fea, reflectance)
#         fea = self.feb(fea, illumination)
#
#         # 新增扩散增强步骤
#         fea = self.diffusion_enhance(fea, illumination)  # 添加扩散增强
#
#         # 后续处理
#         for block in self.blocks:
#             fea = fea + block(fea)
#
#         # 最终重建
#         enhanced = self.out_conv(fea)
#
#         # 保持Retinex约束
#         illumination_3ch = illumination.repeat(1, 3, 1, 1)
#         reflectance_3ch = reflectance[:, :3, :, :].contiguous()
#         enhanced = illumination_3ch * enhanced + reflectance_3ch
#         return torch.clamp(enhanced, 0, 1)
# #



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class IlluminationAwareEnhancement(nn.Module):
#     """
#     光照感知特征增强模块
#     根据光照分量的梯度信息动态调整特征权重。
#     """
#
#     def __init__(self, channels):
#         super(IlluminationAwareEnhancement, self).__init__()
#         self.conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)  # 输入通道数改为 1
#         self.sigmoid = nn.Sigmoid()
#         self.feb = FeatureEnhancementBlock(channels)  # 嵌入FEB模块
#
#     def forward(self, feature, illumination):
#         grad_illumination = self.compute_gradient(illumination)
#         weight = self.sigmoid(self.conv(grad_illumination))
#         enhanced_feature = feature * weight
#         enhanced_feature = self.feb(enhanced_feature)  # 调用FEB模块
#         return enhanced_feature
#
#     @staticmethod
#     def compute_gradient(x):
#         grad_x = F.pad(x[:, :, :, :-1] - x[:, :, :, 1:], (0, 1, 0, 0))
#         grad_y = F.pad(x[:, :, :-1, :] - x[:, :, 1:, :], (0, 0, 0, 1))
#         return torch.abs(grad_x) + torch.abs(grad_y)
#
#
# class ReflectionEnhancement(nn.Module):
#     """
#     反射特征增强模块
#     通过反射分量的高频信息增强特征细节。
#     """
#
#     def __init__(self, channels):
#         super(ReflectionEnhancement, self).__init__()
#         self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#
#     def forward(self, feature, reflectance):
#         high_freq = self.conv(reflectance)
#         enhanced_feature = feature + self.relu(high_freq)
#         return enhanced_feature
#
#
# class FeatureEnhancementBlock(nn.Module):
#     def __init__(self, channels):
#         super(FeatureEnhancementBlock, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
#         self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4)
#         self.attention = nn.Conv2d(channels * 3, channels * 3, kernel_size=1)  # 注意力权重的通道数为 3 * channels
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         conv1_out = self.relu(self.conv1(x))
#         conv2_out = self.relu(self.conv2(x))
#         conv3_out = self.relu(self.conv3(x))
#
#         fused_features = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)
#
#         attention_weights = self.attention(fused_features)
#         attention_weights = torch.split(attention_weights, self.conv1.out_channels, dim=1)  # 拆分为三个部分
#
#         enhanced_features = attention_weights[0] * conv1_out + attention_weights[1] * conv2_out + attention_weights[2] * conv3_out
#
#         return enhanced_features
#
#
#
# class SCINet01(nn.Module):
#     """
#     引入Retinex特征增强和FEB模块的SCI模块
#     """
#
#     def __init__(self, channels=64, layers=3):
#         super(SCINet01, self).__init__()
#         self.in_conv = nn.Sequential(
#             nn.Conv2d(3, channels, kernel_size=3, padding=1),
#             nn.ReLU()
#         )
#         self.illumination_branch = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
#         self.reflectance_branch = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.illumination_aware_enhancement = IlluminationAwareEnhancement(channels)
#         self.reflection_enhancement = ReflectionEnhancement(channels)
#         self.blocks = nn.ModuleList([self.conv_block(channels) for _ in range(layers)])
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(channels, 3, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )
#
#     def conv_block(self, channels):
#         return nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         fea = self.in_conv(x)
#         illumination = self.illumination_branch(fea)  # shape: [B, 1, H, W]
#         reflectance = self.reflectance_branch(fea)  # shape: [B, 64, H, W]
#
#         fea = self.illumination_aware_enhancement(fea, illumination)
#         fea = self.reflection_enhancement(fea, reflectance)
#
#         for block in self.blocks:
#             fea = fea + block(fea)
#
#         enhanced_image = self.out_conv(fea)  # shape: [B, 3, H, W]
#
#         # 扩展 illumination 和 reflectance 的通道数
#         illumination_expanded = illumination.repeat(1, 3, 1, 1)  # shape: [B, 3, H, W]
#         reflectance_reduced = reflectance[:, :3, :, :]  # 取前 3 个通道，shape: [B, 3, H, W]
#
#         enhanced_image = illumination_expanded * enhanced_image + reflectance_reduced
#         enhanced_image = torch.clamp(enhanced_image, 0, 1)
#         return enhanced_image
#
#
# # 测试代码
# if __name__ == "__main__":
#     import torch
#
#     # 创建一个随机输入图像
#     image_size = (1, 3, 256, 256)
#     input_image = torch.rand(*image_size)
#
#     # 初始化 SCI 网络
#     model = SCINet01(channels=64, layers=3)
#
#     # 前向传播
#     enhanced_image = model(input_image)
#
#     # 输出增强图像的形状
#     print("Enhanced Image Shape:", enhanced_image.shape)






# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# __all__ = ['SCINet01']
#
#
# class IlluminationAwareEnhancement(nn.Module):
#     """
#     光照感知特征增强模块
#     根据光照分量的梯度信息动态调整特征权重。
#     """
#
#     def __init__(self, channels):
#         super(IlluminationAwareEnhancement, self).__init__()
#         self.conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)  # 输入通道数改为 1
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, feature, illumination):
#         grad_illumination = self.compute_gradient(illumination)
#         weight = self.sigmoid(self.conv(grad_illumination))
#         enhanced_feature = feature * weight
#         return enhanced_feature
#
#     @staticmethod
#     def compute_gradient(x):
#         grad_x = F.pad(x[:, :, :, :-1] - x[:, :, :, 1:], (0, 1, 0, 0))
#         grad_y = F.pad(x[:, :, :-1, :] - x[:, :, 1:, :], (0, 0, 0, 1))
#         return torch.abs(grad_x) + torch.abs(grad_y)
#
#
# class ReflectionEnhancement(nn.Module):
#     """
#     反射特征增强模块
#     通过反射分量的高频信息增强特征细节。
#     """
#
#     def __init__(self, channels):
#         super(ReflectionEnhancement, self).__init__()
#         self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#
#     def forward(self, feature, reflectance):
#         high_freq = self.conv(reflectance)
#         enhanced_feature = feature + self.relu(high_freq)
#         return enhanced_feature
#
#
# class FeatureFusionBlock(nn.Module):
#     """
#     特征融合模块
#     将光照和反射特征进行融合。
#     """
#
#     def __init__(self, channels):
#         super(FeatureFusionBlock, self).__init__()
#         self.conv = nn.Conv2d(1 + channels, channels, kernel_size=1)
#         self.relu = nn.ReLU()
#
#     def forward(self, illumination, reflectance):
#         fused_feature = torch.cat([illumination, reflectance], dim=1)
#         fused_feature = self.relu(self.conv(fused_feature))
#         return fused_feature
#
#
# class SCINet01(nn.Module):
#     """
#     引入Retinex特征增强的SCI模块
#     """
#
#     def __init__(self, channels=64, layers=3):
#         super(SCINet01, self).__init__()
#         self.in_conv = nn.Sequential(
#             nn.Conv2d(3, channels, kernel_size=3, padding=1),
#             nn.ReLU()
#         )
#         self.illumination_branch = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
#         self.reflectance_branch = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.illumination_aware_enhancement = IlluminationAwareEnhancement(channels)
#         self.reflection_enhancement = ReflectionEnhancement(channels)
#         self.feature_fusion = FeatureFusionBlock(channels)  # 特征融合模块
#         self.blocks = nn.ModuleList([self.conv_block(channels) for _ in range(layers)])
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(channels, 3, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )
#
#     def conv_block(self, channels):
#         return nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         fea = self.in_conv(x)
#         illumination = self.illumination_branch(fea)
#         reflectance = self.reflectance_branch(fea)
#
#         # 光照感知增强
#         fea = self.illumination_aware_enhancement(fea, illumination)
#
#         # 反射增强
#         fea = self.reflection_enhancement(fea, reflectance)
#
#         # 特征融合
#         fea = self.feature_fusion(illumination, reflectance)
#
#         for block in self.blocks:
#             fea = fea + block(fea)
#
#         enhanced_image = self.out_conv(fea)
#         enhanced_image = torch.clamp(enhanced_image, 0, 1)
#         return enhanced_image
#
#
# # 测试代码
# if __name__ == "__main__":
#     import torch
#
#     # 创建一个随机输入图像
#     image_size = (1, 3, 256, 256)
#     input_image = torch.rand(*image_size)
#
#     # 初始化 SCI 网络
#     model = SCINet01(channels=64, layers=3)
#
#     # 前向传播
#     enhanced_image = model(input_image)
#
#     # 输出增强图像的形状
#     print("Enhanced Image Shape:", enhanced_image.shape)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# __all__ = ['SCINet01']
#
#
# class IlluminationAwareEnhancement(nn.Module):
#     """
#     光照感知特征增强模块
#     根据光照分量的梯度信息动态调整特征权重。
#     """
#
#     def __init__(self, channels):
#         super(IlluminationAwareEnhancement, self).__init__()
#         self.conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)  # 输入通道数改为 1
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, feature, illumination):
#         grad_illumination = self.compute_gradient(illumination)
#         weight = self.sigmoid(self.conv(grad_illumination))
#         enhanced_feature = feature * weight
#         return enhanced_feature
#
#     @staticmethod
#     def compute_gradient(x):
#         grad_x = F.pad(x[:, :, :, :-1] - x[:, :, :, 1:], (0, 1, 0, 0))
#         grad_y = F.pad(x[:, :, :-1, :] - x[:, :, 1:, :], (0, 0, 0, 1))
#         return torch.abs(grad_x) + torch.abs(grad_y)
#
#
# class ReflectionEnhancement(nn.Module):
#     """
#     反射特征增强模块
#     通过反射分量的高频信息增强特征细节。
#     """
#
#     def __init__(self, channels):
#         super(ReflectionEnhancement, self).__init__()
#         self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#
#     def forward(self, feature, reflectance):
#         high_freq = self.conv(reflectance)
#         enhanced_feature = feature + self.relu(high_freq)
#         return enhanced_feature
#
#
# class FeatureEnhancementBlock(nn.Module):
#     def __init__(self, channels):
#         super(FeatureEnhancementBlock, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
#         self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4)
#         self.attention = nn.Conv2d(channels * 3, channels * 3, kernel_size=1)  # 注意力权重的通道数为 3 * channels
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         conv1_out = self.relu(self.conv1(x))
#         conv2_out = self.relu(self.conv2(x))
#         conv3_out = self.relu(self.conv3(x))
#
#         fused_features = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)
#
#         attention_weights = self.attention(fused_features)
#         attention_weights = torch.split(attention_weights, self.conv1.out_channels, dim=1)  # 拆分为三个部分
#
#         enhanced_features = attention_weights[0] * conv1_out + attention_weights[1] * conv2_out + attention_weights[
#             2] * conv3_out
#
#         return enhanced_features
#
#
# class SCINet01(nn.Module):
#     """
#     引入Retinex特征增强和FEB模块的SCI模块
#     """
#
#     def __init__(self, channels=64, layers=3):
#         super(SCINet01, self).__init__()
#         self.in_conv = nn.Sequential(
#             nn.Conv2d(3, channels, kernel_size=3, padding=1),
#             nn.ReLU()
#         )
#         self.illumination_branch = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
#         self.reflectance_branch = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.illumination_aware_enhancement = IlluminationAwareEnhancement(channels)
#         self.reflection_enhancement = ReflectionEnhancement(channels)
#         self.feb = FeatureEnhancementBlock(channels)  # 嵌入FEB模块
#         self.blocks = nn.ModuleList([self.conv_block(channels) for _ in range(layers)])
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(channels, 3, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )
#
#     def conv_block(self, channels):
#         return nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         fea = self.in_conv(x)
#         illumination = self.illumination_branch(fea)
#         reflectance = self.reflectance_branch(fea)
#         fea = self.illumination_aware_enhancement(fea, illumination)
#         fea = self.reflection_enhancement(fea, reflectance)
#
#         # 嵌入FEB模块进行特征增强
#         fea = self.feb(fea)
#
#         for block in self.blocks:
#             fea = fea + block(fea)
#
#         enhanced_image = self.out_conv(fea)
#         enhanced_image = illumination * enhanced_image + reflectance
#         enhanced_image = torch.clamp(enhanced_image, 0, 1)
#         return enhanced_image
#
#     def forward(self, x):
#         fea = self.in_conv(x)
#         illumination = self.illumination_branch(fea)  # shape: [B, 1, H, W]
#         reflectance = self.reflectance_branch(fea)  # shape: [B, 64, H, W]
#
#         fea = self.illumination_aware_enhancement(fea, illumination)
#         fea = self.reflection_enhancement(fea, reflectance)
#
#         fea = self.feb(fea)
#
#         for block in self.blocks:
#             fea = fea + block(fea)
#
#         enhanced_image = self.out_conv(fea)  # shape: [B, 3, H, W]
#
#         # 扩展 illumination 和 reflectance 的通道数
#         illumination_expanded = illumination.repeat(1, 3, 1, 1)  # shape: [B, 3, H, W]
#         reflectance_reduced = reflectance[:, :3, :, :]  # 取前 3 个通道，shape: [B, 3, H, W]
#
#         enhanced_image = illumination_expanded * enhanced_image + reflectance_reduced
#         enhanced_image = torch.clamp(enhanced_image, 0, 1)
#         return enhanced_image
#
#
# # 测试代码
# if __name__ == "__main__":
#     import torch
#
#     # 创建一个随机输入图像
#     image_size = (1, 3, 256, 256)
#     input_image = torch.rand(*image_size)
#
#     # 初始化 SCI 网络
#     model = SCINet01(channels=64, layers=3)
#
#     # 前向传播
#     enhanced_image = model(input_image)
#
#     # 输出增强图像的形状
#     print("Enhanced Image Shape:", enhanced_image.shape)


## 1313
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# __all__ = ['SCINet01']
#
#
# class IlluminationAwareEnhancement(nn.Module):
#     """
#     光照感知特征增强模块
#     根据光照分量的梯度信息动态调整特征权重。
#     """
#
#     def __init__(self, channels):
#         super(IlluminationAwareEnhancement, self).__init__()
#         self.conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)  # 输入通道数改为 1
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, feature, illumination):
#         grad_illumination = self.compute_gradient(illumination)
#         weight = self.sigmoid(self.conv(grad_illumination))
#         enhanced_feature = feature * weight
#         return enhanced_feature
#
#     @staticmethod
#     def compute_gradient(x):
#         grad_x = F.pad(x[:, :, :, :-1] - x[:, :, :, 1:], (0, 1, 0, 0))
#         grad_y = F.pad(x[:, :, :-1, :] - x[:, :, 1:, :], (0, 0, 0, 1))
#         return torch.abs(grad_x) + torch.abs(grad_y)
#
#
# class ReflectionEnhancement(nn.Module):
#     """
#     反射特征增强模块
#     通过反射分量的高频信息增强特征细节。
#     """
#
#     def __init__(self, channels):
#         super(ReflectionEnhancement, self).__init__()
#         self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#
#     def forward(self, feature, reflectance):
#         high_freq = self.conv(reflectance)
#         enhanced_feature = feature + self.relu(high_freq)
#         return enhanced_feature
#
#
# class DynamicWeightAdjustment(nn.Module):
#     """
#     动态权重调整模块
#     动态调整不同模块的权重。
#     """
#
#     def __init__(self, num_weights):
#         super(DynamicWeightAdjustment, self).__init__()
#         self.weights = nn.Parameter(torch.ones(num_weights) / num_weights)  # 初始化为均匀权重
#
#     def forward(self, losses):
#         normalized_weights = F.softmax(self.weights, dim=0)
#         return (normalized_weights * losses).sum()
#
#
# class SCINet01(nn.Module):
#     """
#     引入Retinex特征增强的SCI模块
#     """
#
#     def __init__(self, channels=3, layers=3):
#         super(SCINet01, self).__init__()
#         self.in_conv = nn.Sequential(
#             nn.Conv2d(3, channels, kernel_size=3, padding=1),
#             nn.ReLU()
#         )
#         self.illumination_branch = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
#         self.reflectance_branch = nn.Conv2d(channels, 3, kernel_size=3, padding=1)
#         self.illumination_aware_enhancement = IlluminationAwareEnhancement(channels)
#         self.reflection_enhancement = ReflectionEnhancement(channels)
#         self.blocks = nn.ModuleList([self.conv_block(channels) for _ in range(layers)])
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(channels, 3, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )
#         self.dynamic_weight = DynamicWeightAdjustment(layers)  # 动态权重调整模块
#
#     def conv_block(self, channels):
#         return nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         fea = self.in_conv(x)
#         illumination = self.illumination_branch(fea)
#         reflectance = self.reflectance_branch(fea)
#         fea = self.illumination_aware_enhancement(fea, illumination)
#         fea = self.reflection_enhancement(fea, reflectance)
#
#         losses = []
#         for block in self.blocks:
#             fea = fea + block(fea)
#             losses.append(fea.mean())  # 假设每个块的输出可以作为损失的一部分
#
#         # 动态调整权重
#         weighted_loss = self.dynamic_weight(torch.stack(losses))
#         self.dynamic_loss = weighted_loss.item()  # 用于监控
#
#         enhanced_image = self.out_conv(fea)
#         enhanced_image = illumination * enhanced_image + reflectance
#         enhanced_image = torch.clamp(enhanced_image, 0, 1)
#         return enhanced_image


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# __all__ = ['SCINet01']
# class IlluminationAwareEnhancement(nn.Module):
#     """
#     光照感知特征增强模块
#     根据光照分量的梯度信息动态调整特征权重。
#     """
#     def __init__(self, channels):
#         super(IlluminationAwareEnhancement, self).__init__()
#         self.conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)  # 输入通道数改为 1
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, feature, illumination):
#         grad_illumination = self.compute_gradient(illumination)
#         weight = self.sigmoid(self.conv(grad_illumination))
#         enhanced_feature = feature * weight
#         return enhanced_feature
#
#     @staticmethod
#     def compute_gradient(x):
#         grad_x = F.pad(x[:, :, :, :-1] - x[:, :, :, 1:], (0, 1, 0, 0))
#         grad_y = F.pad(x[:, :, :-1, :] - x[:, :, 1:, :], (0, 0, 0, 1))
#         return torch.abs(grad_x) + torch.abs(grad_y)
#
#
# class ReflectionEnhancement(nn.Module):
#     """
#     反射特征增强模块
#     通过反射分量的高频信息增强特征细节。
#     """
#     def __init__(self, channels):
#         super(ReflectionEnhancement, self).__init__()
#         self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#
#     def forward(self, feature, reflectance):
#         high_freq = self.conv(reflectance)
#         enhanced_feature = feature + self.relu(high_freq)
#         return enhanced_feature
#
# class SCINet01(nn.Module):
#     def __init__(self, channels=3, layers=3):
#         super(SCINet01, self).__init__()
#         self.in_conv = nn.Sequential(
#             nn.Conv2d(3, channels, kernel_size=3, padding=1),
#             nn.ReLU()
#         )
#         self.illumination_branch = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
#         self.reflectance_branch = nn.Conv2d(channels, 3, kernel_size=3, padding=1)
#         self.illumination_aware_enhancement = IlluminationAwareEnhancement(channels)
#         self.reflection_enhancement = ReflectionEnhancement(channels)
#         self.blocks = nn.ModuleList([self.conv_block(channels) for _ in range(layers)])
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(channels, 3, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )
#
#     def conv_block(self, channels):
#         return nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         fea = self.in_conv(x)
#         illumination = self.illumination_branch(fea)
#         reflectance = self.reflectance_branch(fea)
#         fea = self.illumination_aware_enhancement(fea, illumination)
#         fea = self.reflection_enhancement(fea, reflectance)
#
#         # 跳跃连接
#         skip_connections = []
#         for block in self.blocks:
#             skip_connections.append(fea)  # 保存当前阶段的特征
#             fea = block(fea)
#
#         # 将跳跃连接的特征与当前特征融合
#         for skip in skip_connections:
#             fea = fea + skip  # 使用加法融合特征
#
#         enhanced_image = self.out_conv(fea)
#         enhanced_image = illumination * enhanced_image + reflectance
#         enhanced_image = torch.clamp(enhanced_image, 0, 1)
#         return enhanced_image


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# __all__ = ['SCINet01']
# class IlluminationAwareEnhancement(nn.Module):
#     """
#     光照感知特征增强模块
#     根据光照分量的梯度信息动态调整特征权重。
#     """
#     def __init__(self, channels):
#         super(IlluminationAwareEnhancement, self).__init__()
#         self.conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)  # 输入通道数改为 1
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, feature, illumination):
#         grad_illumination = self.compute_gradient(illumination)
#         weight = self.sigmoid(self.conv(grad_illumination))
#         enhanced_feature = feature * weight
#         return enhanced_feature
#
#     @staticmethod
#     def compute_gradient(x):
#         grad_x = F.pad(x[:, :, :, :-1] - x[:, :, :, 1:], (0, 1, 0, 0))
#         grad_y = F.pad(x[:, :, :-1, :] - x[:, :, 1:, :], (0, 0, 0, 1))
#         return torch.abs(grad_x) + torch.abs(grad_y)
#
#
# class ReflectionEnhancement(nn.Module):
#     """
#     反射特征增强模块
#     通过反射分量的高频信息增强特征细节。
#     """
#     def __init__(self, channels):
#         super(ReflectionEnhancement, self).__init__()
#         self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#
#     def forward(self, feature, reflectance):
#         high_freq = self.conv(reflectance)
#         enhanced_feature = feature + self.relu(high_freq)
#         return enhanced_feature
#
#
# class SCINet01(nn.Module):
#     """
#     引入Retinex特征增强的SCI模块
#     """
#     def __init__(self, channels=3, layers=3):
#         super(SCINet01, self).__init__()
#         self.in_conv = nn.Sequential(
#             nn.Conv2d(3, channels, kernel_size=3, padding=1),
#             nn.ReLU()
#         )
#         self.illumination_branch = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
#         self.reflectance_branch = nn.Conv2d(channels, 3, kernel_size=3, padding=1)
#         self.illumination_aware_enhancement = IlluminationAwareEnhancement(channels)
#         self.reflection_enhancement = ReflectionEnhancement(channels)
#         self.blocks = nn.ModuleList([self.conv_block(channels) for _ in range(layers)])
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(channels, 3, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )
#
#     def conv_block(self, channels):
#         return nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         fea = self.in_conv(x)
#         illumination = self.illumination_branch(fea)
#         reflectance = self.reflectance_branch(fea)
#         fea = self.illumination_aware_enhancement(fea, illumination)
#         fea = self.reflection_enhancement(fea, reflectance)
#         for block in self.blocks:
#             fea = fea + block(fea)
#         enhanced_image = self.out_conv(fea)
#         enhanced_image = illumination * enhanced_image + reflectance
#         enhanced_image = torch.clamp(enhanced_image, 0, 1)
#         return enhanced_image