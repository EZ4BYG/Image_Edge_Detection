# -*- coding: utf-8 -*-
import sys
import numpy as np
from scipy import signal
import cv2

# Prewtt算子函数：卷积核的可分离性，可用分步卷积(两次小卷积计算)代替一次性卷积计算
def prewitt(I, _boundary = 'symm'):
    # Prewitt_X向卷积：
    # (1) 垂向小卷积：平滑卷积核
    ones_y = np.array([[1],[1],[1]], np.float32)
    conv_prewitt_x = signal.convolve2d(I, ones_y, mode = 'same', boundary = _boundary)
    # (2) 水平向小卷积：差分卷积核
    diff_x = np.array([[1,0,-1]], np.float32)
    conv_prewitt_x = signal.convolve2d(conv_prewitt_x, diff_x, mode='same', boundary = _boundary)

    # Prewitt_Y向卷积：
    # (1) 水平向小卷积：平滑卷积核
    ones_x = np.array([[1,1,1]], np.float32)
    conv_prewitt_y = signal.convolve2d(I, ones_x, mode = 'same', boundary = _boundary)
    # (2) 垂向小卷积：差分卷积核
    diff_y = np.array([[1],[0],[-1]], np.float32)
    conv_prewitt_y = signal.convolve2d(conv_prewitt_y, diff_y, mode = 'same', boundary = _boundary)

    # 返回两个方向卷积核的卷积后结果：
    return [conv_prewitt_x, conv_prewitt_y]

# 主函数
if __name__ =="__main__":
    # 图像读取与显示：参数1是文件路径，路径不要有中文；当前图片在同一文件夹下
    image = cv2.imread('doge2.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', image)

    # 两个卷积核的卷积结果：卷积结果有负值(差分卷积核导致)，故要取绝对值！
    conv_prewitt_x, conv_prewitt_y = np.abs( prewitt(image) )
    # x和y向的边缘强度的灰度级显示：注意卷积后像素点有数值>255
    edge_x = conv_prewitt_x.copy()
    edge_y = conv_prewitt_y.copy()
    # 单阈值划分：
    edge_x[ edge_x>255 ] = 255
    edge_y[ edge_y>255 ] = 255
    edge_x = edge_x.astype(np.uint8)
    edge_y = edge_y.astype(np.uint8)
    # 结果显示与保存：
    cv2.imshow('Prewitt_Edge_X', edge_x)
    cv2.imshow('Prewitt_Edge_Y', edge_y)
    cv2.imwrite("Prewitt_Edge_X.jpg", edge_x)
    cv2.imwrite("Prewitt_Edge_Y.jpg", edge_y)

    # 总边缘强度计算：每个像素点两个卷积结果的平方和的开方(取整)
    edge = np.round( np.sqrt(np.power(conv_prewitt_x,2) + np.power(conv_prewitt_y,2)) )
    # 单阈值划分：
    edge[ edge>255 ] = 255
    edge = edge.astype(np.uint8)
    cv2.imshow('Prewitt_Edge', edge)
    cv2.imwrite('Prewitt_Edge.jpg', edge)

    # 图像一直显示直到关闭图像窗口，程序运行结束
    cv2.waitKey(0)
    cv2.destroyAllWindows()