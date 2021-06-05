# -*- coding: utf-8 -*-
import sys
import numpy as np
from scipy import signal
import cv2

def krisch(I, _boundary = 'fill', _fillvalue = 0):
    # 存储8个方向的边缘强度：记录8个卷积核卷积后的结果(有负数，记得取绝对值)
    list_edge = []

    # 卷积核1：
    k1 = np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])
    conv_kirsch_k1 = signal.convolve2d(I, k1, mode = 'same', boundary = _boundary, fillvalue = _fillvalue)
    list_edge.append( np.abs(conv_kirsch_k1) )

    # 卷积核2：
    k2 = np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]])
    conv_kirsch_k2 = signal.convolve2d(I, k2, mode = 'same', boundary = _boundary, fillvalue = _fillvalue)
    list_edge.append( np.abs(conv_kirsch_k2) )

    # 卷积核3：
    k3 = np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]])
    conv_kirsch_k3 = signal.convolve2d(I, k3, mode = 'same', boundary = _boundary, fillvalue = _fillvalue)
    list_edge.append( np.abs(conv_kirsch_k3) )

    # 卷积核4：
    k4 = np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]])
    conv_kirsch_k4 = signal.convolve2d(I, k4, mode = 'same', boundary = _boundary, fillvalue = _fillvalue)
    list_edge.append( np.abs(conv_kirsch_k4) )

    # 卷积核5：
    k5 = np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]])
    conv_kirsch_k5 = signal.convolve2d(I, k5, mode = 'same', boundary = _boundary, fillvalue = _fillvalue)
    list_edge.append( np.abs(conv_kirsch_k5) )

    # 卷积核6：
    k6 = np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]])
    conv_kirsch_k6 = signal.convolve2d(I, k6, mode = 'same', boundary = _boundary, fillvalue = _fillvalue)
    list_edge.append( np.abs(conv_kirsch_k6) )

    # 卷积核7：
    k7 = np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]])
    conv_kirsch_k7 = signal.convolve2d(I, k7, mode = 'same', boundary = _boundary, fillvalue = _fillvalue)
    list_edge.append( np.abs(conv_kirsch_k7) )

    # 卷积核8：
    k8 = np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])
    conv_kirsch_k8 = signal.convolve2d(I, k8, mode = 'same', boundary = _boundary, fillvalue = _fillvalue)
    list_edge.append( np.abs(conv_kirsch_k8) )

    # 总边缘强度：取8个卷积核卷积结果对应像素位置的最大值
    edge = list_edge[0]
    for i in range( len(list_edge) ):
        # 一个一个对比，选出当下对比的二者中，所有像素点位置的最大值：
        # 第一项：edge*( edge>=list_edge[i] ) —— 选出edge中像素点值更大的位置，其他置为0
        # 第二项：list_edge[i]*( edge<list_edge[i] ) —— 选出list_edge[i]中像素点值更大的位置，其他为0
        # 两项相加，完全互相弥补0值位置的缺陷，即挑出了两者中所有像素点位置的最大值！
        edge = edge*( edge>=list_edge[i] ) + list_edge[i]*( edge<list_edge[i] )

    # 返回所有结果：
    return [list_edge, edge]

# 主函数
if __name__ == "__main__":
    # 图像读取与显示：参数1是文件路径，路径不要有中文；当前图片在同一文件夹下
    image = cv2.imread('doge2.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', image)

    # 卷积结果：
    list_edge, edge = krisch(image)
    # 单阈值划分：
    edge[ edge>255 ] = 255
    edge = edge.astype(np.uint8)
    for i in range( len(list_edge) ):
        list_edge[i][ list_edge[i]>255 ] = 255
        list_edge[i] = list_edge[i].astype(np.uint8)

    # 图像显示与保存：
    cv2.imshow('Kirsch_Edge', edge)
    cv2.imwrite('Kirsch_Edge.jpg', edge)
    for i in range( len(list_edge) ):
        cv2.imshow('Kirsch_Edge_K' + str(i+1), list_edge[i])
        cv2.imwrite('Kirsch_Edge_K' + str(i+1) + '.jpg', list_edge[i])

    # 图像一直显示直到关闭图像窗口，程序运行结束
    cv2.waitKey(0)
    cv2.destroyAllWindows()