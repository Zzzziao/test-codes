import math
import torch
import torch.optim
import torch.nn.functional as F
import cv2
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sl
import matplotlib.pyplot as plt
import cupy as cp
import scipy.sparse as sparse
import scipy.sparse.linalg as sl

def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def TransmissionEstimate_WLS(im,A,sz, lambda_ = 0.35, alpha = 1.2):
    omega = 0.95
    im3 = np.empty(im.shape,im.dtype);

    ## 这个写法是对按通道滤波的
    # for ind in range(0,3):
    #     im3[:,:,ind] = im[:,:,ind]/A[0,ind]
    # dark_est = DarkChannel(im3,sz);
    # dark_est_wls = wls_filter(dark_est, lambda_, alpha)
    # transmission_wls = 1 - omega * dark_est_wls

    ## 这个写法是对初步估计的Trans map进行滤波的
    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    transmission_wls = wls_filter(transmission, lambda_, alpha)

    return transmission_wls


def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et,r = 60):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

def ASM(J,t,A,tx = 0.1):

    matrix_one = np.ones(J.shape,J.dtype)
    generated_hazed_image = np.empty(J.shape, J.dtype);
    t = cv2.max(t, tx)

    for ind in range(0,3):
        generated_hazed_image[:,:,ind] = J[:,:,ind] * t + (matrix_one[:,:,ind] - t) * A[0,ind]
    return generated_hazed_image

def ASM_torch(J, t, A, tx=0.1):
    """
        使用 PyTorch 实现的 ASM 函数，通过处理形状使计算正确进行。

        Args:
            J: 输入图像的 PyTorch 张量，形状为 [1, 300, 300, 3]，值范围为 [0, 1]。
            t: 单通道透射率图像的 PyTorch 张量，形状为 [1, 300, 300]。
            A: 大气光的 PyTorch 张量，形状为 [3, 1, 1]，每个值分别对应 RGB 通道。
            tx: 最小透射率，标量。

        Returns:
            generated_hazed_image: 生成的雾霾图像，PyTorch 张量，形状为 [1, 300, 300, 3]。
        """
    # 确保 t 的最小值为 tx
    t = torch.max(t, torch.tensor(tx, dtype=t.dtype, device=t.device))

    # 扩展 t 的维度使其与 J 的形状兼容，变为 [1, 300, 300, 1]
    t = t.unsqueeze(-1)

    # 初始化生成的雾霾图像张量，形状为 [1, 300, 300, 3]
    generated_hazed_image_torch = torch.empty_like(J)

    # 计算生成的雾霾图像
    generated_hazed_image_torch = J * t + (1 - t) * A

    return generated_hazed_image_torch


def process_difference_operator(difference_operator, lambda_, alpha, epsilon):
    difference_operator = -lambda_ / (epsilon + (np.absolute(difference_operator)**alpha))
    return difference_operator


def wls_filter(L, lambda_=0.35, alpha=1.2, epsilon=1e-4):
    # Get log-luminance
    L_log = np.log(L.astype(np.float64) + 1e-10)

    # Compute the forward and backward differences of the luminance channel
    dx_forward = L_log - cv2.copyMakeBorder(L_log[:,1:], top=0, bottom=0, left=0, right=1,
                                        borderType=cv2.BORDER_REPLICATE)
    dx_backward = L_log - cv2.copyMakeBorder(L_log[:,:-1], top=0, bottom=0, left=1, right=0,
                                        borderType=cv2.BORDER_REPLICATE)
    dy_forward = L_log - cv2.copyMakeBorder(L_log[1:,:], top=0, bottom=1, left=0, right=0,
                                        borderType=cv2.BORDER_REPLICATE)
    dy_backward = L_log - cv2.copyMakeBorder(L_log[:-1,:], top=1, bottom=0, left=0, right=0,
                                        borderType=cv2.BORDER_REPLICATE)

    # Weight each derivative
    dx_forward_weighted = process_difference_operator(dx_forward, lambda_, alpha, epsilon)
    dx_forward_weighted[:,-1] = 0

    dx_backward_weighted = process_difference_operator(dx_backward, lambda_, alpha, epsilon)
    dx_backward_weighted[:,0] = 0

    dy_forward_weighted = process_difference_operator(dy_forward, lambda_, alpha, epsilon)
    dy_forward_weighted[-1,:] = 0

    dy_backward_weighted = process_difference_operator(dy_backward, lambda_, alpha, epsilon)
    dy_backward_weighted[0,:] = 0

    central_element = np.ones_like(dx_forward)-(dx_forward_weighted + dx_backward_weighted +
                                   dy_forward_weighted + dy_backward_weighted)

    # Form sparse matrix
    N = L.size
    C = L.shape[1]

    row = np.zeros(5*N, dtype=np.int32)
    col = np.zeros_like(row, dtype=np.int32)
    data = np.zeros_like(row, dtype=np.float64)

    # Central element
    row[:N] = np.arange(N)
    col[:N] = row[:N]
    data[:N] = central_element.ravel()

    # dx_forward
    row[N:2*N] = np.arange(N)
    col[N:2*N] = row[N:2*N] + 1
    data[N:2*N] = dx_forward_weighted.ravel()

    # dx_backward
    row[2*N:3*N] = np.arange(N)
    col[2*N:3*N] = row[2*N:3*N] - 1
    data[2*N:3*N] = dx_backward_weighted.ravel()

    #dy_forward
    row[3*N:4*N] = np.arange(N)
    col[3*N:4*N] = row[3*N:4*N] + C
    data[3*N:4*N] = dy_forward_weighted.ravel()

    #dy_backward
    row[4*N:5*N] = np.arange(N)
    col[4*N:5*N] = row[4*N:5*N] - C
    data[4*N:5*N] = dy_backward_weighted.ravel()

    # Prevent out-of-bounds indices. Overlapping elements sum together, so setting all
    # out-of-bounds values to zero and repositioning them to (0,0) will have no effect
    # on other values in the sparse matrix.
    data[col >= N] = 0
    data[col < 0] = 0
    row[col >= N] = 0
    row[col < 0] = 0
    col[col >= N] = 0
    col[col < 0] = 0

    A = sparse.coo_matrix((data, (row, col))).tocsr()
    b = L.ravel()

    x, info = sl.cg(A=A, b=b)

    x = x.reshape(L.shape)

    return x

def box_filter(x, r):
    kernel_size = torch.round(r).float() * 2 + 1  # 保持浮点数
    kernel_size_int = int(kernel_size.item())
    kernel = torch.ones(1, 1, kernel_size_int, kernel_size_int, dtype=x.dtype, device=x.device) / (kernel_size_int * kernel_size_int)
    padding = kernel_size_int // 2
    return F.conv2d(x, kernel, padding=padding)

def Guidedfilter_torch(im, p, r, eps):
    mean_I = box_filter(im, r)
    mean_p = box_filter(p, r)
    mean_Ip = box_filter(im * p, r)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = box_filter(im * im, r)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)

    q = mean_a * im + mean_b
    return q

def TransmissionRefine_torch(im, et, r_torch):
    # 将 RGB 图像转换为灰度图像
    gray = 0.2989 * im[:, 0, :, :] + 0.5870 * im[:, 1, :, :] + 0.1140 * im[:, 2, :, :]
    gray = gray.float() / 255  # 归一化灰度图像

    eps = 0.0001
    t = Guidedfilter_torch(gray.unsqueeze(1), et.unsqueeze(1), r_torch, eps)  # 保留浮点形式的 r_torch
    return t.squeeze(1)

def TransmissionRefine_torch1(im,et, r_torch):

    r_int = torch.round(r_torch).int().item()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray) / 255;
    eps = 0.0001;
    t = Guidedfilter(gray, et, r_int, eps);

    return t;


def Recover_torch(im, t, A, tx=0.1):
    # im shape: (1, 3, 300, 300)
    # t shape: (1, 300, 300)
    # A shape: (1, 3)

    # Ensure t is at least tx
    t = torch.max(t, torch.tensor(tx, dtype=t.dtype, device=t.device))  # shape: (1, 300, 300)

    # Add channel dimension to t so it can be broadcasted
    t = t.unsqueeze(1)  # shape: (1, 1, 300, 300)

    # Broadcast A to match the shape of im
    A = A.view(1, 3, 1, 1)  # shape: (1, 3, 1, 1)

    # Perform the recovery operation
    res = (im - A) / t + A  # shape: (1, 3, 300, 300)

    return res

### torch版本的WLS DarkChannel ASM
def ASM_WLS_torch(J, A , I_torch, sz, lambda_, alpha, tx=0.1):
    t_torch = transmission_estimate_wls(I_torch, A, sz, lambda_, alpha)
    # 确保 t 的最小值为 tx
    t = torch.max(t_torch, torch.tensor(tx, dtype=t_torch.dtype, device=t_torch.device))

    # 扩展 t 的维度使其与 J 的形状兼容，变为 [1, 600, 600, 1]
    t = t.unsqueeze(-1)

    # 初始化生成的雾霾图像张量，形状为 [1, 600, 600, 3]
    generated_hazed_image_torch = torch.empty_like(J)

    # 计算生成的雾霾图像
    generated_hazed_image_torch = J * t + (1 - t) * A

    return generated_hazed_image_torch


def dark_channel_torch(im, sz):
    r, g, b = im[0, 0, :, :], im[0, 1, :, :], im[0, 2, :, :]
    dc = torch.min(torch.min(r, g), b)
    kernel = torch.ones((1, 1, sz, sz), device=im.device)
    dark = F.max_pool2d(dc.unsqueeze(0), kernel_size=(sz, sz), stride=1, padding=sz // 2)
    return dark.squeeze()

def process_difference_operator_torch(diff, lambda_, alpha, epsilon):
    result = lambda_ / ((torch.abs(diff) ** alpha) + epsilon)
    if torch.isnan(result).any():
        print("NaN detected in process_difference_operator_torch")
    return result

def wls_filter_torch(L, lambda_=0.35, alpha=1.2, epsilon=1e-4):
    # 增加数值稳定性
    L_log = torch.log(L + 1e-8)
    if torch.isnan(L_log).any():
        print("NaN detected in L_log")

    # 手动计算差分
    dx_forward = L_log[:, 1:] - L_log[:, :-1]
    dx_forward = torch.cat([dx_forward, dx_forward[:, -1:]], dim=1)

    dx_backward = L_log[:, :-1] - L_log[:, 1:]
    dx_backward = torch.cat([dx_backward[:, :1], dx_backward], dim=1)

    dy_forward = L_log[1:, :] - L_log[:-1, :]
    dy_forward = torch.cat([dy_forward, dy_forward[-1:, :]], dim=0)

    dy_backward = L_log[:-1, :] - L_log[1:, :]
    dy_backward = torch.cat([dy_backward[:1, :], dy_backward], dim=0)

    dx_forward_weighted = process_difference_operator_torch(dx_forward, lambda_, alpha, epsilon)
    dx_backward_weighted = process_difference_operator_torch(dx_backward, lambda_, alpha, epsilon)
    dy_forward_weighted = process_difference_operator_torch(dy_forward, lambda_, alpha, epsilon)
    dy_backward_weighted = process_difference_operator_torch(dy_backward, lambda_, alpha, epsilon)

    central_element = 1 - (dx_forward_weighted + dx_backward_weighted + dy_forward_weighted + dy_backward_weighted)

    # 检查是否有NaN值
    if torch.isnan(central_element).any():
        print("NaN detected in central_element")
        return None

    H, W = L.shape

    # 使用 cupy 代替 numpy 来加速运算
    L_flat = cp.array(L.view(-1).cpu().detach().numpy())
    central_flat = cp.array(central_element.view(-1).cpu().detach().numpy())

    row = np.repeat(np.arange(H * W, dtype=np.int32), 5)
    col = row.copy()
    data = np.zeros(row.shape, dtype=np.float64)

    data[:H * W] = central_flat.get()

    dx_fw_flat = cp.array(dx_forward_weighted.view(-1).cpu().detach().numpy())
    col[H * W:2 * H * W] += 1
    data[H * W:2 * H * W] = dx_fw_flat.get()

    dx_bw_flat = cp.array(dx_backward_weighted.view(-1).cpu().detach().numpy())
    col[2 * H * W:3 * H * W] -= 1
    data[2 * H * W:3 * H * W] = dx_bw_flat.get()

    dy_fw_flat = cp.array(dy_forward_weighted.view(-1).cpu().detach().numpy())
    col[3 * H * W:4 * H * W] += W
    data[3 * H * W:4 * H * W] = dy_fw_flat.get()

    dy_bw_flat = cp.array(dy_backward_weighted.view(-1).cpu().detach().numpy())
    col[4 * H * W:5 * H * W] -= W
    data[4 * H * W:5 * H * W] = dy_bw_flat.get()

    # Clip col to be within the valid range
    col = np.clip(col, 0, H * W - 1)

    # 创建稀疏矩阵 A
    A = sparse.coo_matrix((data, (row, col)), shape=(H * W, H * W)).tocsr()

    # 将 L_flat 转换为 numpy 数组
    L_flat = cp.asnumpy(L_flat)

    # 使用GPU加速的共轭梯度求解
    x, info = sl.cg(A, L_flat, M=None, tol=1e-10, maxiter=1000)

    # 检查共轭梯度求解器的收敛性
    if info != 0:
        print(f"CG did not converge. Info: {info}")
        return None

    # 将结果从 cupy 转回 PyTorch
    x = torch.tensor(cp.asnumpy(x), dtype=L.dtype, device=L.device).view(H, W)

    # 检查输出是否有NaN
    if torch.isnan(x).any():
        print("NaN detected in output")
        return None

    return x


def transmission_estimate_wls(im, A, sz, lambda_=0.35, alpha=1.2):
    omega = 0.95
    im3 = im / A.unsqueeze(-1).unsqueeze(-1)

    transmission = 1 - omega * dark_channel_torch(im3, sz)
    transmission_wls = wls_filter_torch(transmission, lambda_, alpha)

    return transmission_wls
