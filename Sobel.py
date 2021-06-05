# -*- coding: utf-8 -*-
import sys
import math
import cv2
import numpy as np
from scipy import signal

# 非归一化的高斯平滑算子：nxn阶Sobel卷积核中的平滑算子，由(n-1)阶二项式展开系数组成
def pascalSmooth(n):
    pascalSmooth = np.zeros([1,n], np.float32)
    # (n-1)阶二项式展开的系数：
    for i in range(n):
        pascalSmooth[0][i] = math.factorial(n-1)/(math.factorial(i)*math.factorial(n-1-i))
    return pascalSmooth

# 差分算子：nxn阶Sobel卷积核中的差分算子，由(n-2)阶二项式展开系数两侧补0再后向差分得到
def pascalDiff(n):
    pascalDiff = np.zeros([1,n], np.float32)
    # 注意：pascalSmooth中自带-1阶，故只需再-1阶即可！
    pascalSmooth_previous = pascalSmooth(n-1)
    for i in range(n):
        # 根据补0后的后向差分规律：左端一定是pascalSmooth(n-1)计算出的第一个系数
        if i == 0:
            pascalDiff[0][i] = pascalSmooth_previous[0][i]
        # 根据补0后的后向差分规律：右端一定是pascalSmooth(n-1)计算出的最后一个系数的相反数
        elif i == (n-1):
            pascalDiff[0][i] = -pascalSmooth_previous[0][i-1]
        # 中间的都是后向差分：右-左
        else:
            pascalDiff[0][i] = pascalSmooth_previous[0][i] - pascalSmooth_previous[0][i-1]
    return pascalDiff

# Sobel卷积核查看函数：只用来查看自定义阶的卷积核中的元素构成
# 注：真正的Sobel算子还是用的分步计算，见下面sobel函数
def getSobelKernel(kernelsize):
     pascalSmoothKernel = pascalSmooth(kernelsize) # 得到平滑算子系数
     pascalDiffKernel = pascalDiff(kernelsize)     # 得到差分算子系数
     # 水平方向上的卷积核：两个算子合并 —— 注意转置
     sobelKernel_x = signal.convolve2d(pascalSmoothKernel.transpose(), pascalDiffKernel, mode='full')
     # 垂直方向上的卷积核：两个算子合并 —— 注意转置
     sobelKernel_y = signal.convolve2d(pascalSmoothKernel, pascalDiffKernel.transpose(), mode='full')
     return [sobelKernel_x,sobelKernel_y]

# Sobel算子函数：卷积核的可分离性，可用分步卷积(两次小卷积计算)代替一次性卷积计算
def sobel(I, kernelsize):
    rows,cols = I.shape
    # 获取高斯平滑算子 + 差分算子：
    pascalSmoothKernel = pascalSmooth(kernelsize)
    pascalDiffKernel = pascalDiff(kernelsize)
    # Sobel_X向卷积：
    # (1) 水平向小卷积：非归一化的高斯平滑卷积核
    conv_sobel_x = signal.convolve2d(I, pascalSmoothKernel.transpose(), mode = 'same')
    # (2) 垂向小卷积：差分卷积核
    conv_sobel_x = signal.convolve2d(conv_sobel_x, pascalDiffKernel, mode = 'same')

    # Sobel_Y向卷积：
    # (1) 垂向小卷积：差分卷积核
    conv_sobel_y = signal.convolve2d(I, pascalSmoothKernel, mode = 'same')
    # (2) 水平向小卷积：非归一化的高斯平滑卷积核
    conv_sobel_y = signal.convolve2d(conv_sobel_y, pascalDiffKernel.transpose(), mode = 'same')
    return [conv_sobel_x,conv_sobel_y]

# 主函数
if __name__ == "__main__":
    # 获得指定阶数的卷积核：
    kernelsize = int( input('卷积核阶数：') )
    sobelkernel_x, sobelkernel_y = getSobelKernel(kernelsize)
    print(sobelkernel_x)
    print(sobelkernel_y)

    # 图像读取与显示：参数1是文件路径，路径不要有中文；当前图片在同一文件夹下
    image = cv2.imread('doge2.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', image)

    conv_sobel_x, conv_sobel_y = sobel(image, kernelsize)
    # Sobel_X向卷积：取绝对值 + 单阈值划分
    edge_x = np.abs(conv_sobel_x)
    edge_x[ edge_x>255 ] = 255
    edge_x = edge_x.astype(np.uint8)
    # Sobel_Y向卷积：取绝对值 + 单阈值划分
    edge_y = np.abs(conv_sobel_y)
    edge_y[ edge_y>255 ]=255
    edge_y = edge_y.astype(np.uint8)
    # 结果显示与保存：
    cv2.imshow('Sobel_Edge_X_' + str(kernelsize) + 'size', edge_x)
    cv2.imshow('Sobel_Edge_Y_' + str(kernelsize) + 'size', edge_y)
    cv2.imwrite('Sobel_Edge_X_' + str(kernelsize) + 'size' + '.jpg', edge_x)
    cv2.imwrite('Sobel_Edge_Y_' + str(kernelsize) + 'size' + '.jpg', edge_y)

    # 总边缘强度计算：每个像素点两个卷积结果的平方和的开方(取整)
    edge = np.round( np.sqrt(np.power(conv_sobel_x,2) + np.power(conv_sobel_y,2)) )
    # 单阈值划分：
    edge[ edge>255 ] = 255
    edge = edge.astype(np.uint8)
    cv2.imshow('Sobel_Edge_' + str(kernelsize) + 'size', edge)
    cv2.imwrite('Sobel_Edge_' + str(kernelsize) + 'size' + '.jpg', edge)

    # 图像一直显示直到关闭图像窗口，程序运行结束
    cv2.waitKey(0)
    cv2.destroyAllWindows()