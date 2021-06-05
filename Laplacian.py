# -*- coding: utf-8 -*-
import sys
import numpy as np
from scipy import signal
import cv2
import math
# from gaussBlur import gaussBlur # 高斯平滑，和边缘提取无关，它用来提取前去噪的

def laplacian(I, _boundary = 'fill', _fillvalue = 0):
    # 拉普拉斯卷积核：多种形式(每次只有1个卷积核)，但核内所有值的和为0
    laplacianKernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], np.float32)
    # laplacianKernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], np.float32)
    # laplacianKernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], np.float32)
    # laplacianKernel = np.array([[2,-1,2],[-1,-4,-1],[2,-1,2]], np.float32)
    # laplacianKernel = np.array([[0,2,0],[2,-8,2],[0,2,0]], np.float32)
    # laplacianKernel = np.array([[2,0,2],[0,-8,0],[2,0,2]], np.float32)
    conv_laplacian = signal.convolve2d(I, laplacianKernel, mode = 'same', boundary = _boundary, fillvalue = _fillvalue)
    return conv_laplacian

# 主函数
if __name__ == "__main__":
    image = cv2.imread('doge2.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', image)

    # 卷积结果：
    conv_laplacian = np.around( np.abs(laplacian(image)) )
    conv_laplacian[ conv_laplacian>255 ] = 255
    conv_laplacian = conv_laplacian.astype(np.uint8)
    cv2.imshow('Laplacian_Edge', conv_laplacian)
    cv2.imwrite('Laplacian_Edge.jpg', conv_laplacian)

    # 图像一直显示直到关闭图像窗口，程序运行结束
    cv2.waitKey(0)
    cv2.destroyAllWindows()
