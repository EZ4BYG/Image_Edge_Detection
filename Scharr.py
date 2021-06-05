# -*- coding: utf-8 -*-
import sys
import numpy as np
from scipy import signal
import cv2

def scharr(I, _boundary = 'symm'):
    # Scharr_X向卷积：不可分离，直接计算
    # scharr45 = np.array([[0,3,10],[-3,0,3],[-10,-3,0]], np.float32) # 也可用45°卷积核
    scharr_x = np.array([[3,0,-3],[10,0,-10],[3,0,-3]], np.float32)
    conv_scharr_x = signal.convolve2d(I, scharr_x, mode = 'same', boundary = 'symm')
    # Scharr_Y向卷积：不可分离，直接计算
    # scharr135 = np.array([[10,3,0],[3,0,-3],[0,-3,10]], np.float32) # 也可用135°卷积核
    scharr_y = np.array([[3,10,3],[0,0,0],[-3,-10,-3]], np.float32)
    conv_scharr_y = signal.convolve2d(I,scharr_y,mode='same',boundary='symm')
    # 结果返回：
    return [conv_scharr_x, conv_scharr_y]

# 主函数
if __name__ == "__main__":
    # 图像读取与显示：参数1是文件路径，路径不要有中文；当前图片在同一文件夹下
    image = cv2.imread('doge2.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', image)

    # 两个卷积核的卷积结果：卷积结果有负值(差分卷积核导致)，故要取绝对值！
    conv_scharr_x, conv_scharr_y = np.abs( scharr(image) )
    # x和y向的边缘强度的灰度级显示：注意卷积后像素点有数值>255
    edge_x = conv_scharr_x.copy()
    edge_y = conv_scharr_y.copy()
    # 单阈值划分：
    edge_x[ edge_x>255 ] = 255
    edge_y[ edge_y>255 ] = 255
    edge_x = edge_x.astype(np.uint8)
    edge_y = edge_y.astype(np.uint8)
    # 结果显示与保存：
    cv2.imshow('Scharr_Edge_X', edge_x)
    cv2.imshow('Scharr_Edge_Y', edge_y)
    cv2.imwrite("Scharr_Edge_X.jpg", edge_x)
    cv2.imwrite("Scharr_Edge_Y.jpg", edge_y)

    # 总边缘强度计算：每个像素点两个卷积结果的平方和的开方(取整)
    edge = np.around( np.sqrt(np.power(conv_scharr_x,2)+np.power(conv_scharr_y,2)) )
    # 单阈值划分：
    edge[ edge>255 ] = 255
    edge = edge.astype(np.uint8)
    cv2.imshow('Scharr_Edge', edge)
    cv2.imwrite('Scharr_Edge.jpg', edge)

    # 图像一直显示直到关闭图像窗口，程序运行结束
    cv2.waitKey(0)
    cv2.destroyAllWindows()