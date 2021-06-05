# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
from scipy import signal

# 构建LoG卷积核
def createLoGKernel(sigma, kernelsize):
    # LoG算子的高和宽，且两者均为奇数
    H,W = kernelsize
    r,c = np.mgrid[0:H:1,0:W:1]
    r = r - (H-1)/2
    c = c - (W-1)/2
    # 方差：更新主要在这里，计算了新的标准差!
    sigma2 = pow(sigma, 2.0)
    # LoG核
    norm2 = np.power(r,2.0) + np.power(c,2.0)
    # LoGKernel = 1.0/sigma2*(norm2/sigma2 -2)*np.exp(-norm2/(2*sigma2))
    LoGKernel = (norm2/sigma2 -2)*np.exp(-norm2/(2*sigma2))
    return LoGKernel

# LoG算子
def LoG(image, sigma, kernelsize, _boundary = 'symm'):
    # 构建LoG卷积核
    loGKernel = createLoGKernel(sigma, kernelsize)
    # 图像与LoG卷积核卷积
    img_conv_log = signal.convolve2d(image, loGKernel, 'same', boundary = _boundary)
    return img_conv_log

# 主函数
if __name__ == "__main__":
    image = cv2.imread('s1.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', image)

    # 卷积结果："卷积核大小"以及"标准差"人为指定
    # 注：这里卷积核不能太小!
    kernelsize = (127, 127)
    sigma = 5
    conv_log = LoG(image, sigma, kernelsize, 'symm')
    # 单阈值划分：
    edge_binary = np.copy(conv_log)
    edge_binary[ edge_binary>0 ] = 255
    edge_binary[ edge_binary<=0 ] = 0
    edge_binary = edge_binary.astype(np.uint8)
    cv2.imshow("Edge_LoG_New", edge_binary)
    cv2.imwrite('Edge_LoG_New.jpg', edge_binary)

    # 图像一直显示直到关闭图像窗口，程序运行结束
    cv2.waitKey(0)
    cv2.destroyAllWindows()