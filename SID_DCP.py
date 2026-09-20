from __future__ import print_function
import matplotlib.pyplot as plt
# %matplotlib inline

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
from models import *

import torch
import torch.optim
import torch.nn.functional as F

# from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.sr_utils import *
from utils.dehaze import *


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
# dtype = torch.FloatTensor

imsize = -1
PLOT = True
## Dehazing
fname = 'hazy input/1449.jpg'
img_pil, img_np = get_image(fname, imsize)
img_hazed_np = img_np

# plt.imshow(img_np.transpose(1, 2, 0))  # 可能需要转置以适应 Matplotlib 的通道顺序
# plt.axis('off')  # 隐藏坐标轴
# plt.show()
## Traditional DCP method test

I = img_np.transpose(1, 2, 0)
dark = DarkChannel(I, sz=15)
A = AtmLight(I, dark)
te = TransmissionEstimate(I, A, sz=15)
t = TransmissionRefine(I, te, r=60)
J = Recover(I, t, A, 0.1)

## Traditional WLS method test

# I = img_np.transpose(1, 2, 0)
# dark = DarkChannel(I, sz=7)
# A = AtmLight(I, dark)
# t = TransmissionEstimate_WLS(I, A, sz=15, lambda_=0.8, alpha=7)  ## light fog sz=7 lambda=0.35 alpha=4
# ## heavy fog sz=15 lambda=0.8 alpha=7
# J = Recover(I, t, A, 0.1)


J_torch = np_to_torch(J)
t_torch = np_to_torch(t)
A_torch = torch.tensor(A, dtype=torch.float32)
## ASM
# gene_hazed_image_torch = ASM_torch(J_torch, t_torch, A_torch)
# gene_hazed_image = gene_hazed_image_torch.squeeze(0).cpu().numpy()
# gene_hazed_image = ASM(J, t, A)
#
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(t)
# axes[0].set_title("Original image")
# axes[1].imshow(J)
# axes[1].set_title("Haze-free image")
# plt.show()

## Setup
INPUT = 'noise'  # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net'  # 'net,input'

reg_noise_std = 1. / 30.  # skip用 1. / 30.   Unet和Resnet用0.0
LR = 0.01   ## skip用0.01，Unet和Resnet用0.001
tv_weight = 0.0

OPTIMIZER = 'adam'  # 'LBFGS'
show_every = 100
check_every = 100
exp_weight = 0.99  ## 用于对out的平滑，越接近1则平滑的越厉害，如果噪声没有那么大可以选择0.95，或者0.9

num_iter = 3000
input_depth = 32
figsize = 5

## Skip network
net = get_net(input_depth, 'skip', pad,
              skip_n33d=128,
              skip_n33u=128,
              skip_n11=4,
              num_scales=5,
              upsample_mode='bilinear').type(dtype)
## Optional U-Net generator (provided by models.Unet_new through `from models import *`)
# net = UNet_new(num_input_channels=input_depth, num_output_channels=3,
#                feature_scale=8, more_layers=0, concat_x=False,
#                upsample_mode='bilinear', pad=1,
#                norm_layer=torch.nn.InstanceNorm2d, need_sigmoid=True,
#                need_bias=True).type(dtype)
## Resnet network
# net = ResNet(input_depth, img_np.shape[0], 8, 32, need_sigmoid=True, act_fun='LeakyReLU').type(dtype)

net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()  ## for skip network
# net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype).detach() ## for Unet and Resnet network
# Compute number of parameters
s = sum([np.prod(list(p.size())) for p in net.parameters()])
print('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)
l1_loss = torch.nn.L1Loss().type(dtype)

img_hazed_torch = np_to_torch(img_hazed_np).type(dtype)  # 到这里准备任务结束了

## Optimize
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
out_avg = None
last_net = None
psrn_haze_last = 0

i = 0


def closure():
    global i, out_avg, psrn_haze_last, last_net, net_input

    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    out = net(net_input)

    # out = F.interpolate(out, size=(300, 300), mode='bilinear', align_corners=False)  ## Unet 才用到，其他的模式下记得注释掉

    # Smoothing  实现指数加权平均移动（EMA），用于平滑输出out，以减少噪声或者波动对于最终结果的影响
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

    # out_np_forASM = torch_to_np(out).transpose(1, 2, 0)
    # gene_hazed_image = ASM(out_np_forASM, t, A).type(dtype)
    # gene_hazed_image_torch = np_to_torch(gene_hazed_image.transpose(2, 0, 1))

    gene_hazed_image_torch = ASM_torch(out.permute(0, 2, 3, 1), t_torch.type(dtype), A_torch.type(dtype))
    gene_hazed_image_torch = gene_hazed_image_torch.permute(0, 3, 1, 2)

    total_loss = mse(gene_hazed_image_torch, img_hazed_torch)
    # total_loss = l1_loss(gene_hazed_image_torch, img_hazed_torch)
    ## 正则化
    if tv_weight > 0:
        total_loss += tv_weight * tv_loss(out)

    total_loss.backward()

    psrn_haze = compare_psnr(img_hazed_np, gene_hazed_image_torch.detach().cpu().numpy()[0])
    psrn_gt = compare_psnr(img_np, gene_hazed_image_torch.detach().cpu().numpy()[0])
    psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0])

    # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
    # print('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (
    # i, total_loss.item(), psrn_haze, psrn_gt, psrn_gt_sm), '\r', end='')
    print('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (
        i, total_loss.item(), psrn_haze, psrn_gt, psrn_gt_sm))

    if PLOT and i % show_every == 0:
        out_np = torch_to_np(out)
        plot_image_grid([np.clip(out_np, 0, 1),
                         np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1)

    # Backtracking  用于监控模型性能的稳定性，并在模型性能突然下降时采取回滚操作（即恢复到之前的模型参数），以避免模型陷入不良状态
    if i % check_every:
        if psrn_haze - psrn_haze_last < -5:
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())
            ## 通过循环，将 last_net 中保存的上一个检查点的模型参数复制回当前模型的参数中，进行回滚操作。

            return total_loss * 0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psrn_haze_last = psrn_haze

    i += 1

    return total_loss

p = get_params(OPT_OVER, net, net_input)  # 注意OPT_OVER模式为net,input的时候可以把net_input这个张量也作为优化参数加入进去
optimize(OPTIMIZER, p, closure, LR, num_iter)
out_np = torch_to_np(net(net_input))
# out_np = torch_to_np(F.interpolate(net(net_input), size=(300, 300), mode='bilinear', align_corners=False)) ## Unet 才用到，其他的模式下记得注释掉
q = plot_image_grid_final([np.clip(out_np, 0, 1), img_np], factor=13);




# sigma = 25
# sigma_ = sigma/255.
## laod images
# deJPEG
# fname = 'data/denoising/snail.jpg'

## denoising
# fname = 'data/denoising/F16_GT.png'
#
# if fname == 'data/denoising/snail.jpg':
#     img_noisy_pil = crop_image(get_image(fname, imsize)[0], d=32)
#     img_noisy_np = pil_to_np(img_noisy_pil)
#
#     # As we don't have ground truth
#     img_pil = img_noisy_pil
#     img_np = img_noisy_np
#
#     if PLOT:
#         plot_image_grid([img_np], 4, 5);
#
# elif fname == 'data/denoising/F16_GT.png':
#     # Add synthetic noise
#     img_pil = crop_image(get_image(fname, imsize)[0], d=32)
#     img_np = pil_to_np(img_pil)
#
#     img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
#
#     if PLOT:
#         plot_image_grid([img_np, img_noisy_np], 4, 6);
# else:
#     assert False
# if fname == 'data/denoising/snail.jpg':
#     num_iter = 2400
#     input_depth = 3
#     figsize = 5
#
#     net = skip(
#         input_depth, 3,
#         num_channels_down=[8, 16, 32, 64, 128],
#         num_channels_up=[8, 16, 32, 64, 128],
#         num_channels_skip=[0, 0, 0, 4, 4],
#         upsample_mode='bilinear',
#         need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
#
#     net = net.type(dtype)
#
# elif fname == 'data/denoising/F16_GT.png':
#     num_iter = 3000
#     input_depth = 32
#     figsize = 4
#
#     net = get_net(input_depth, 'skip', pad,
#                   skip_n33d=128,
#                   skip_n33u=128,
#                   skip_n11=4,
#                   num_scales=5,
#                   upsample_mode='bilinear').type(dtype)
#
# else:
#     assert False
