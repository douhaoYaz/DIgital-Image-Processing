# -*- coding: utf-8 -*-
"""
Created on 11 30 17:06 2021

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

def showImages(images, titles, rows, cols, savePath):
    """ 绘图函数

    :param images: 绘制的子图
    :param titles: 子图的标题
    :param rows: 画布中子图的行数
    :param cols: 画布中子图的列数
    :param savePath: 保存路径
    :return: None
    """

    for i in range(len(images)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i], fontsize=8)
        plt.xticks([]), plt.yticks([])

    savefig(plt, savePath, cols * 5 * images[0].shape[1], rows * 5 * images[0].shape[0])
    plt.show()


if __name__ == '__main__':
    # 任务a
    # imread()不能读取路径包含中文的图片，此处先用numpy的fromfile()函数读取，再用OpenCV的imdecode()从内存的buffer中读取图片
    shapes = cv.imdecode(
        np.fromfile(r'F:\Work and Learn\FIGHT\大三\数字图像处理\Lab\Lab3\Lab3_Image\shapes.jpg', dtype=np.uint8), -1)

    # 转为灰度图像
    gray = cv.cvtColor(shapes, cv.COLOR_BGR2GRAY)

    # 利用Canny算子进行边缘检测
    edges = cv.Canny(gray, 50, 150)

    # 检测圆的位置
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 50, param2=30)

    # 画出圆
    drawing = np.zeros(shapes.shape[:], dtype=np.uint8)
    for i in circles[0, :]:
        # 画出圆周
        cv.circle(drawing, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
        # 画出圆心
        cv.circle(drawing, (int(i[0]), int(i[1])), 2, (255, 0, 0), 3)


    # 构造子图序列和标题序列
    images = [shapes, edges, drawing]
    titles = ['Original', 'Edges', 'Statistic Hough']

    # 显示
    showImages(images, titles, 1, 3, 'hough1.png')


    # 任务b
    # imread()不能读取路径包含中文的图片，此处先用numpy的fromfile()函数读取，再用OpenCV的imdecode()从内存的buffer中读取图片
    stairs = cv.imdecode(
        np.fromfile(r'F:\Work and Learn\FIGHT\大三\数字图像处理\Lab\Lab3\Lab3_Image\stairs.jpg', dtype=np.uint8), -1)

    # 转为灰度图像
    gray = cv.cvtColor(stairs, cv.COLOR_BGR2GRAY)

    # 利用Canny算子进行边缘检测
    edges = cv.Canny(gray, 50, 150)

    # 检测直线位置
    lines = cv.HoughLines(edges, 1, np.pi / 180, 200)

    # 画出直线
    standard = np.zeros(stairs.shape[:], np.uint8)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv.line(standard, (x1, y1), (x2, y2), (0, 0, 255))

    # 统计概率霍夫线变换
    statistics = stairs.copy()
    lines = cv.HoughLinesP(edges, 0.8, np.pi / 100, 90, minLineLength=50, maxLineGap=10)

    # 将检测的线画出来
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(statistics, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv.LINE_AA)

    # 构造子图序列和标题序列
    images = [stairs, edges, standard, statistics]
    titles = ['Original', 'Edges', 'Standard Hough', 'Statistic Hough']

    showImages(images, titles, 2, 2, 'hough2.png')

