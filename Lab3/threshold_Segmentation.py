# -*- coding: utf-8 -*-
"""
Created on 11 30 9:42 2021

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

def showImages(nameWindow, img, title_img, img1, title_img1, img2, title_img2, img3, title_img3, img4, title_img4, img5, title_img5, savePath):
    """ 显示处理后的图像

    :param nameWindow: 窗口名称
    :param img: 原始图像
    :param title_img: 图像标题
    :param img1: 修改后图像1
    :param title_img1: 图像标题
    :param img2: 修改后图像2
    :param title_img2: 图像标题
    :param img3: 修改后图像3
    :param title_img3: 图像标题
    :param img4: 修改后图像4
    :param title_img4: 图像标题
    :param img5: 修改后图像5
    :param title_img5: 图像标题
    :param savePath: 保存路径
    :return: None
    """

    # 创建画布figure
    fig = plt.figure(nameWindow)
    # 创建子图1
    subplot1 = fig.add_subplot(2, 3, 1)
    subplot1.set_title(title_img)
    subplot1.imshow(img, cmap='gray')
    plt.axis('off')

    # 创建子图2
    subplot1 = fig.add_subplot(2, 3, 2)
    subplot1.set_title(title_img1)
    subplot1.imshow(img1, cmap='gray')
    plt.axis('off')

    # 创建子图3
    subplot1 = fig.add_subplot(2, 3, 3)
    subplot1.set_title(title_img2)
    subplot1.imshow(img2, cmap='gray')
    plt.axis('off')

    # 创建子图4
    subplot1 = fig.add_subplot(2, 3, 4)
    subplot1.set_title(title_img3)
    subplot1.imshow(img3, cmap='gray')
    plt.axis('off')

    # 创建子图5
    subplot1 = fig.add_subplot(2, 3, 5)
    subplot1.set_title(title_img4)
    subplot1.imshow(img4, cmap='gray')
    plt.axis('off')

    # 创建子图6
    subplot1 = fig.add_subplot(2, 3, 6)
    subplot1.set_title(title_img5)
    subplot1.imshow(img5, cmap='gray')
    plt.axis('off')

    # 调整子图间距
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    #保存
    savefig(plt, savePath, 3.3 * img.shape[1], 1.1 * img.shape[0])

    # 显示图像
    plt.show()

if __name__ == '__main__':
    # imread()不能读取路径包含中文的图片，此处先用numpy的fromfile()函数读取，再用OpenCV的imdecode()从内存的buffer中读取图片
    gradient = cv.imdecode(np.fromfile(r'F:\Work and Learn\FIGHT\大三\数字图像处理\Lab\Lab3\Lab3_Image\gradient.jpg', dtype=np.uint8), -1)
    # 转成灰度图
    gradient = cv.cvtColor(gradient, cv.COLOR_BGR2GRAY)

    # 进行5个不同类型的阈值化
    ret1, gradient_bin = cv.threshold(gradient, 127, 255, cv.THRESH_BINARY)
    ret2, gradient_bin_inv = cv.threshold(gradient, 127, 255, cv.THRESH_BINARY_INV)
    ret3, gradient_trunc = cv.threshold(gradient, 127, 255, cv.THRESH_TRUNC)
    ret4, gradient_toz =  cv.threshold(gradient, 127, 255, cv.THRESH_TOZERO)
    ret5, gradient_toz_inv = cv.threshold(gradient, 127, 255, cv.THRESH_TOZERO_INV)

    showImages("", gradient, "Original", gradient_bin, "BINARY", gradient_bin_inv, "BINARY_INV", gradient_trunc, "TRUNC", gradient_toz, "TOZERO", gradient_toz_inv, "TOZERO_INV", "gradient.png")

    # cv.imshow("gradient", gradient_toz_inv)
    # cv.waitKey(0)
