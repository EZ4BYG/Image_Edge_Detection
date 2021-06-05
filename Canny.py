# -*- coding: utf-8 -*-
import numpy as np
import math
import cv2
# 注：Sobel.py文件必须在同一路径下，这里直接调用它利用Sobel算子的函数
import Sobel
# 注：增加嵌套层数上限(默认是1000)，防止stack overflow问题出现
import sys
sys.setrecursionlimit(20000)

# 非极大值抑制(粗糙版)：梯度方向线只跨2个邻域像素
# 传入参数：dx和dy是Sobel两个卷积核卷积后的结果：
def non_maximum_suppression_default(dx,dy):
    # 边缘强度：
    edgeMag = np.around( np.sqrt(np.power(dx,2) + np.power(dy,2)) )
    # 记录图片宽和高：
    rows, cols = edgeMag.shape
    # 梯度方向：
    gradientDirection = np.zeros(edgeMag.shape)

    # 边缘强度非极大值抑制：下面都对edgeMag_nonMaxSup处理！
    # 注意：边缘一圈都是0值，不做处理，故循环从1开始！
    edgeMag_nonMaxSup = np.zeros(edgeMag.shape)
    for r in range(1,rows-1):
        for c in range(1,cols-1):

            # angle的范围：[0,180]&[-180,0]
            angle = math.atan2(dy[r][c],dx[r][c])/math.pi*180
            gradientDirection[r][c] = angle
            # (1) 梯度方向线横跨左/右，angle的范围：[0,22.5)&(-22.5,0) & (157.5,180]&[-180,-157.5)
            # 若观测点比领域两点都大，则保持原值，否则置0
            if( abs(angle)<22.5 or abs(angle)>157.5 ):
                if( edgeMag[r][c]>edgeMag[r][c-1] and edgeMag[r][c]>edgeMag[r][c+1] ):
                    edgeMag_nonMaxSup[r][c] = edgeMag[r][c]

            # (2) 梯度方向线横跨左上/右下，angle的范围：[22.5,67.5)&[-157.5,-112.5)
            if( (angle>=22.5 and angle<67.5) or (-angle > 112.5 and -angle <= 157.5) ):
                if( edgeMag[r][c]>edgeMag[r-1][c-1] and edgeMag[r][c]>edgeMag[r+1][c+1] ):
                     edgeMag_nonMaxSup[r][c] = edgeMag[r][c]

            # (3) 梯度方向线横跨上/下，angle的范围：[67.5,112.5]&[-112.5,-67.5]
            if( (angle>=67.5 and angle<=112.5) or (angle>=-112.5 and angle<=-67.5) ):
                if( edgeMag[r][c]>edgeMag[r-1][c] and edgeMag[r][c]>edgeMag[r+1][c]):
                    edgeMag_nonMaxSup[r][c] = edgeMag[r][c]

            # (4) 梯度方向线横跨右上/左下，angle的范围：
            if((angle>112.5 and angle<=157.5) or(-angle>=22.5 and -angle< 67.5 )):
                if(edgeMag[r][c]>edgeMag[r-1][c+1] and edgeMag[r][c] > edgeMag[r+1][c-1]):
                    edgeMag_nonMaxSup[r][c] = edgeMag[r][c]
    # 返回结果：
    return edgeMag_nonMaxSup

# 非极大值抑制(精细插值版)：梯度方向线可跨4个邻域像素
def non_maximum_suppression_Inter(dx,dy):
    # 边缘强度：
    edgeMag = np.around( np.sqrt(np.power(dx,2.0) + np.power(dy,2.0)) )
    # 记录图片宽和高：
    rows, cols = edgeMag.shape
    # 梯度方向：
    gradientDirection = np.zeros(edgeMag.shape)

    # 边缘强度非极大值抑制：下面都对edgeMag_nonMaxSup处理！
    # 注意：边缘一圈都是0值，不做处理，故循环从1开始！
    edgeMag_nonMaxSup = np.zeros(edgeMag.shape)
    for r in range(1,rows-1):
        for c in range(1,cols-1):
            # 因为插值会用到dx/dy或dy/dx的比值，故要排除0值的情况：
            if dy[r][c] == 0 or dx[r][c] == 0:
                continue
            # angle的范围：[0,180]&[-180,0]
            angle = math.atan2(dy[r][c],dx[r][c])/math.pi*180
            gradientDirection[r][c] = angle
            # (1) 梯度方向线横跨左上/上/右下/下，angle范围：(45.90]&(-135,-90]；左上与上插值，右下与下插值
            if (angle>45 and angle<=90) or (angle>-135 and angle <=-90):
                ratio = abs(dx[r][c]/dy[r][c])
                # 插值：合并为两个新的"邻域"
                leftTop_top = ratio*edgeMag[r-1][c-1] + (1-ratio)*edgeMag[r-1][c]
                rightBottom_bottom = (1-ratio)*edgeMag[r+1][c] + ratio*edgeMag[r+1][c+1]
                if (edgeMag[r][c] > leftTop_top) and (edgeMag[r][c] > rightBottom_bottom):
                    edgeMag_nonMaxSup[r][c] = edgeMag[r][c]

            # (2) 梯度方向线横跨左下/下/右上/上，angle范围：(90,135]&(-90,-45]
            if (angle>90 and angle<=135) or (angle>-90 and angle<=-45):
                ratio = abs(dx[r][c]/dy[r][c])
                # 插值：合并为两个新的"邻域"
                rightTop_top = ratio*edgeMag[r-1][c+1] + (1-ratio)*edgeMag[r-1][c]
                leftBottom_bottom = ratio*edgeMag[r+1][c-1] + (1-ratio)*edgeMag[r+1][c]
                if (edgeMag[r][c] > rightTop_top) and (edgeMag[r][c] > leftBottom_bottom):
                    edgeMag_nonMaxSup[r][c] = edgeMag[r][c]

            # (3) 梯度方向线横跨左上/左/右下/右，angle范围：[0,45]&[-180,-135]
            if (angle>=0 and angle <=45) or (angle>-180 and angle<=-135):
                ratio = abs(dy[r][c]/dx[r][c])
                # 插值：合并为两个新的"邻域"
                rightBottom_right = ratio*edgeMag[r+1][c+1]+(1-ratio)*edgeMag[r][c+1]
                leftTop_left = ratio*edgeMag[r-1][c-1]+(1-ratio)*edgeMag[r][c-1]
                if (edgeMag[r][c] > rightBottom_right) and (edgeMag[r][c] > leftTop_left):
                    edgeMag_nonMaxSup[r][c] = edgeMag[r][c]

            # (4) 梯度方向线横跨左下/左/右上/右，angle范围：(135,180]&(-45,0)
            if(angle>135 and angle<=180) or (angle>-45 and angle<0):
                ratio = abs(dy[r][c]/dx[r][c])
                # 插值：合并为两个新的"邻域"
                rightTop_right = ratio*edgeMag[r-1][c+1]+(1-ratio)*edgeMag[r][c+1]
                leftBottom_left = ratio*edgeMag[r+1][c-1]+(1-ratio)*edgeMag[r][c-1]
                if (edgeMag[r][c] > rightTop_right) and (edgeMag[r][c] > leftBottom_left):
                    edgeMag_nonMaxSup[r][c] = edgeMag[r][c]
    # 返回结果：
    return edgeMag_nonMaxSup

# 判断一个点的坐标是否在图像范围内
# 参数：(r,c)为像素的坐标点，(rows,cols)为图像总尺寸
def checkInRange(r, c, rows, cols):
    if r>=0 and r<rows and c>=0 and c<cols:
        return True
    else:
        return False

# 边缘的延长函数：延长"滞后阈值"中确定性(>upperThresh)边缘点
# 传入的参数(r,c)是确定性边缘点的坐标
# trace函数是判断以(r,c)为中心的3x3范围内是否有处于">lowerThresh"的点，并将其延长成边缘点!
def trace(edgeMag_nonMaxSup, edge, lowerThresh, r, c, rows, cols):
    # 大于阈值为确定边缘点：
    # 注：先判断是否为0 —— 因为edge初始化全是0，如果这里是0，说明该位置还未进行阈值判断
    if edge[r][c] == 0:
        # 确定性边缘的赋值：因为传入的(r,c)是确定性边缘点，故直接赋值为255
        edge[r][c] = 255
        # 边缘延长：选择以(r,c)为中心的3x3窗口，中心点(r,c)与周围一圈的8个点的关系均可认为是在"一条线"上，进而延长边缘
        for i in range(-1,2):
            for j in range(-1,2):
                if checkInRange(r+i, c+j, rows, cols) and (edgeMag_nonMaxSup[r+i][c+j] >= lowerThresh):
                    trace(edgeMag_nonMaxSup, edge, lowerThresh, r+i, c+j, rows, cols)

# 双阈值(高低)滞后阈值处理：获得最终的Canny边缘
def hysteresisThreshold(edge_nonMaxSup, lowerThresh, upperThresh):
    # 图片宽和高：
    rows, cols = edge_nonMaxSup.shape
    # edge是被"阈值处理"的关键变量，先都初始化为0值：
    edge = np.zeros(edge_nonMaxSup.shape, np.uint8)
    for r in range(1,rows-1):
        for c in range(1,cols-1):
            # 大于高阈值，直接认定为确定边缘点，并以该点为起始点延长边缘
            # 更进一步：确定性边缘点要尽可能延长其边界，观测窗口为以该边缘点为中心的3x3邻域像素点
            # 注意：对确定性边缘点的赋值在trace函数中进行
            if edge_nonMaxSup[r][c] >= upperThresh:
                trace(edge_nonMaxSup, edge, lowerThresh, r, c, rows, cols)
            # 小于低阈值，直接设置为0
            if edge_nonMaxSup[r][c] < lowerThresh:
                edge[r][c] = 0
    return edge

# 主函数
if __name__ == "__main__":
    image = cv2.imread('doge2.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', image)

    # Canny边缘检测：
    # (1) 利用Sobel算子获得两个方向的卷积结果：Sobel阶数可以人为指定，最好是奇数!
    kernelsize = 3
    conv_sobel_x, conv_sobel_y = Sobel.sobel(image, kernelsize)

    # (2) 边缘强度非极大值抑制：
    # 选择1：non_maximum_suppression_default
    edgeMag_nonMaxSup_default = non_maximum_suppression_default(conv_sobel_x, conv_sobel_y)
    edgeMag_nonMaxSup_default[ edgeMag_nonMaxSup_default>255 ] = 255
    edgeMag_nonMaxSup_default = edgeMag_nonMaxSup_default.astype(np.uint8)
    cv2.imshow('Canny_edgeMag_nonMaxSup_Default_' + str(kernelsize) + 'sizesSobel', edgeMag_nonMaxSup_default)
    cv2.imwrite('Canny_edgeMag_nonMaxSup_Default.jpg', edgeMag_nonMaxSup_default)
    # 选择2：non_maximum_suppression_Inter
    edgeMag_nonMaxSup_Inter = non_maximum_suppression_Inter(conv_sobel_x, conv_sobel_y)
    edgeMag_nonMaxSup_Inter[ edgeMag_nonMaxSup_Inter>255 ] = 255
    edgeMag_nonMaxSup_Inter = edgeMag_nonMaxSup_Inter.astype(np.uint8)
    cv2.imshow('Canny_edgeMag_nonMaxSup_Inter_' + str(kernelsize) + 'sizesSobel', edgeMag_nonMaxSup_Inter)
    cv2.imwrite('Canny_edgeMag_nonMaxSup_Inter.jpg', edgeMag_nonMaxSup_Inter)

    # (3) 双阈值滞后阈值处理：lowerThresh和upperThresh人为指定(超参数)
    lowerThresh = 40
    upperThresh = 180
    edge_default = hysteresisThreshold(edgeMag_nonMaxSup_default, lowerThresh, upperThresh)
    edge_inter = hysteresisThreshold(edgeMag_nonMaxSup_Inter, lowerThresh, upperThresh)
    cv2.imshow('Canny_Edge_Default_' + str(kernelsize) + 'sizesSobel', edge_default)
    cv2.imshow('Canny_Edge_Inter_' + str(kernelsize) + 'sizesSobel', edge_inter)
    cv2.imwrite('Canny_Edge_Default.jpg', edge_default)
    cv2.imwrite('Canny_Edge_Inter.jpg', edge_inter)

    # 图像一直显示直到关闭图像窗口，程序运行结束
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        