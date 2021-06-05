# -*- coding: utf-8 -*-

import sys
# from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy import signal
import cv2

# roberts算子函数：本例卷积全用full类
def roberts(I, _boundary = 'fill', _fillvalue = 0):
    # 图像的高和宽：
    H1, W1 = I.shape[0:2]
    # 卷积核的尺寸：
    H2, W2 = [2,2]
    # 45°卷积核及锚点的位置：
    R1 = np.array([[1,0],[0,-1]], np.float32)
    kr1,kc1 = [0,0]
    # 计算45°卷积核的full卷积：
    IconR1 = signal.convolve2d(I, R1, mode = 'full', boundary = _boundary, fillvalue = _fillvalue)
    IconR1 = IconR1[H2-kr1-1:H1+H2-kr1-1, W2-kc1-1:W1+W2-kc1-1]
    # 135°卷积核及锚点的位置：
    R2 = np.array([[0,1],[-1,0]], np.float32)
    kr2, kc2 = [0,1]
    # 计算135°卷积核的full卷积：
    IconR2 = signal.convolve2d(I, R2, mode = 'full', boundary = _boundary, fillvalue = _fillvalue)
    IconR2 = IconR2[H2-kr2-1:H1+H2-kr2-1, W2-kc2-1:W1+W2-kc2-1]
    # 结果返回：
    return [IconR1,IconR2]

# 主函数：
if __name__ =="__main__":
    # 图像读取与显示：参数1是文件路径，路径不要有中文；当前图片在同一文件夹下
    image = cv2.imread('doge2.jpg',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', image)
    cv2.imwrite('Hui.jpg', image)

    # 两个卷积核的卷积结果：
    IconR1,IconR2 = roberts(image,'symm')
    # 45°方向上的边缘灰度变化率：显示与保存
    IconR1 = np.abs(IconR1)
    edge_45 = IconR1.astype(np.uint8)
    cv2.imshow('Robert_Edge_45', edge_45)
    cv2.imwrite('Robert_Edge_45.jpg', edge_45)
    # 135°方向上的边缘灰度变化率：显示与保存
    IconR2 = np.abs(IconR2)
    edge_135 = IconR2.astype(np.uint8)
    cv2.imshow('Robert_Edge_135', edge_135)
    cv2.imwrite('Robert_Edge_135.jpg', edge_135)

    # 总边缘强度计算：每个像素点两个卷积结果的平方和的开方(取整)
    edge = np.round( np.sqrt(np.power(IconR1,2) + np.power(IconR2,2)) )
    # 单阈值划分：
    edge[ edge>255 ] = 255
    edge = edge.astype(np.uint8)
    # 总边缘强度显示与保存：
    cv2.imshow('Robert_Edge', edge)
    cv2.imwrite('Robert_Edge.jpg', edge)

    # 图像一直显示直到关闭图像窗口，程序运行结束
    cv2.waitKey(0)
    cv2.destroyAllWindows()