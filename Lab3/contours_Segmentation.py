# -*- coding: utf-8 -*-
"""
Created on 11 30 15:49 2021

@author: douhaoYaz@DGUT ACE Lab
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def savefig(plt, filename, width=1024, height=768, dpi=300):
    fig = plt.gcf()
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    fig.set_size_inches(width / dpi, height /dpi)
    fig.text(0.5, 0.5, '201841510108 符浩扬', fontsize=10, rotation=0, color='red', ha='right', va='bottom', alpha=0.7)
    fig.savefig(filename, format='png', transparent=True, dpi=300, pad_inches=0)

def showImages(images, titles, savePath):
    """ 绘图函数

    :param images:绘制的子图
    :param titles: 子图的标题
    :param savePath: 保存路径
    :return: None
    """

    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i], fontsize=8)
        plt.xticks([]), plt.yticks([])

    savefig(plt, savePath, 3.3 * images[0].shape[1], 2.2 * images[0].shape[0])
    plt.show()


if __name__ == '__main__':
    # imread()不能读取路径包含中文的图片，此处先用numpy的fromfile()函数读取，再用OpenCV的imdecode()从内存的buffer中读取图片
    coins = cv.imdecode(
        np.fromfile(r'F:\Work and Learn\FIGHT\大三\数字图像处理\Lab\Lab3\Lab3_Image\coins.jpg', dtype=np.uint8), -1)

    # 利用cv2.pyrMeanShiftFiltering()在色彩层面的平滑滤波，它可以中和色彩分布相近的颜色，平滑色彩细节，侵蚀掉面积较小的颜色区域
    shifted = cv.pyrMeanShiftFiltering(coins, 21, 51)

    # 转为灰度图像
    gray = cv.cvtColor(shifted, cv.COLOR_BGR2GRAY)

    # Otsu阈值分割
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # 寻找轮廓
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    print("[INFO] {} unique contours found".format(len(cnts)))

    # 将轮廓在原图上标记出来
    labeled_coins = coins.copy()
    for (i, c) in enumerate(cnts):
        ((x, y), _) = cv.minEnclosingCircle(c)
        cv.putText(labeled_coins, "#{}".format(i + 1), (int(x) - 10, int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv.drawContours(labeled_coins, [c], -1, (0, 255, 0), 2)

    # 创建子图序列和标题序列
    images = [coins, shifted, gray, thresh, labeled_coins]
    titles = ['Original', 'MeanShiftFiltering', 'Gray', 'OTSU', 'Labeled']

    # 显示
    showImages(images, titles, "contours.png")
