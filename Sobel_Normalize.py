# -*- coding: utf-8 -*-
import numpy as np
import sys
import math
import cv2
from scipy import signal

# 非归一化的高斯平滑算子：nxn阶Sobel卷积核中的平滑算子，由(n-1)阶二项式展开系数组成
def pascalSmooth(n):
    pascalSmooth = np.zeros([1,n],np.float32)
    for i in range(n):
        pascalSmooth[0][i] = math.factorial(n -1)/(math.factorial(i)*math.factorial(n-1-i))
    return pascalSmooth

# 差分算子：nxn阶Sobel卷积核中的差分算子，由(n-2)阶二项式展开系数两侧补0再后向差分得到
def pascalDiff(n):
    pascalDiff = np.zeros([1,n],np.float32)
    pascalSmooth_previous = pascalSmooth(n-1)
    for i in range(n):
        if i ==0:
            #恒等于 1
            pascalDiff[0][i] = pascalSmooth_previous[0][i]
        elif i == n-1:
            #恒等于 -1
            pascalDiff[0][i] = -pascalSmooth_previous[0][i-1]
        else:
            pascalDiff[0][i] = pascalSmooth_previous[0][i] - pascalSmooth_previous[0][i-1]
    return pascalDiff

# 通过平滑系数和差分系数的卷积运算计算卷积核
def getSobelKernel(winSize):
     pascalSmoothKernel = pascalSmooth(winSize)
     pascalDiffKernel = pascalDiff(winSize)
     # 水平方向上的卷积核
     sobelKernel_x = signal.convolve2d(pascalSmoothKernel.transpose(), pascalDiffKernel, mode='full')
     # 垂直方向上的卷积核
     sobelKernel_y = signal.convolve2d(pascalSmoothKernel, pascalDiffKernel.transpose(), mode='full')
     return (sobelKernel_x, sobelKernel_y)

# Sobel算子: 卷积核大小是传入参数
def sobel(image, winSize):
    rows, cols = image.shape
    pascalSmoothKernel = pascalSmooth(winSize)
    pascalDiffKernel = pascalDiff(winSize)
    # --- 与水平方向的卷积核卷积 ----
    image_sobel_x = np.zeros(image.shape,np.float32)
    # 垂直方向上的平滑
    image_sobel_x = signal.convolve2d(image,pascalSmoothKernel.transpose(),mode='same')
    # 水平方向上的差分
    image_sobel_x = signal.convolve2d(image_sobel_x,pascalDiffKernel,mode='same')
    # --- 与垂直方向上的卷积核卷积 --- 
    image_sobel_y = np.zeros(image.shape,np.float32)
    # 水平方向上的平滑
    image_sobel_y = signal.convolve2d(image,pascalSmoothKernel,mode='same')
    # 垂直方向上的差分
    image_sobel_y = signal.convolve2d(image_sobel_y,pascalDiffKernel.transpose(),mode='same')
    return (image_sobel_x,image_sobel_y)

# 主函数
if __name__ =="__main__":
    image = cv2.imread('doge2.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', image)

    # 卷积
    image_sobel_x, image_sobel_y = sobel(image,7)
    # 平方和开方的方式
    edge = np.sqrt(np.power(image_sobel_x,2.0) + np.power(image_sobel_y,2.0))
    # 边缘强度的灰度级显示 —— 新处理方式：归一化
    edge = edge/np.max(edge)
    edge = np.power(edge,0.8)
    edge *= 255
    edge = edge.astype(np.uint8)
    cv2.imshow("sobel edge",edge)
    cv2.imwrite("sobel.jpg",edge)

    # 图像一直显示直到关闭图像窗口，程序运行结束
    cv2.waitKey(0)
    cv2.destroyAllWindows()