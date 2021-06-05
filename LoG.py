# -*- coding: utf-8 -*-
import sys
import numpy as np
import math
import cv2
from scipy import signal

# 构建LoG卷积核
def createLoGkernel(sigma, kernelsize):
    # LoG卷积核的高H和宽W，要求两者均为奇数!
    (winH, winW) = kernelsize
    logkernel = np.zeros(kernelsize, np.float32)
    # LoG卷积核的中心点：
    centerH = (winH-1)/2
    centerW = (winW-1)/2
    # 核内元素的计算：sigmaSquare是计算中必备的一个系数
    sigmaSquare = pow(sigma, 2.0)
    for h in range(winH):  # r
        for w in range(winW):  # c
            # norm2 = x^2 + y^2 = h^2 + w^2 = 分子项
            norm2 = pow(h-centerH, 2.0) + pow(w-centerW, 2.0)
            logkernel[h][w] = 1.0/sigmaSquare*(norm2/sigmaSquare - 2)*math.exp(-norm2/(2*sigmaSquare))
    return logkernel

# LoG算子
def LoG(image, sigma, kernelsize, _boundary = 'symm', _fillValue = 0):
    # 构建LoG卷积核：
    logkernel = createLoGkernel(sigma, kernelsize)
    # 计算卷积结果：
    conv_log = signal.convolve2d(image, logkernel, 'same', boundary = _boundary)
    return [logkernel, conv_log]

# 主函数
if __name__ == "__main__":
    image = cv2.imread('doge2.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', image)

    # 卷积结果：卷积核大小人为指定
    kernelsize = (11,11)
    logkernel, conv_log = LoG(image, 2, kernelsize)
    print('LoG卷积核(3位小数):\n', np.around(logkernel,3))
    # 单阈值划分：
    edge = conv_log.copy()
    edge[ edge>0 ] = 255
    edge[ edge<=0 ] = 0
    edge = edge.astype(np.uint8)
    cv2.imshow('LoG_Edge_' + str(kernelsize[0]) + 'sizes', edge)
    cv2.imwrite('Log_Edge_' + str(kernelsize[0]) + 'sizes.jpg', edge)

    # 图像一直显示直到关闭图像窗口，程序运行结束
    cv2.waitKey(0)
    cv2.destroyAllWindows()