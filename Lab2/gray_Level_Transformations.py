# -*- coding: utf-8 -*-
"""
Created on 11 03 15:36 2021

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

def transform(value, a, b):
    res = a * int(value) + b
    if res < 0 :
        res = np.uint8(0)
        return res
    elif res > 255 :
        res = np.uint8(255)
        return res
    else:
        return res

def showImages(nameWindow, img, title_img, img1, title_img1, img2, title_img2, img3, title_img3, img4, title_img4, img5, title_img5, savePath):
    """ 显示灰度线性变换后的图像

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


def log(c, img):
    output = c * np.log(1.0 + img)
    output = np.uint8(output + 0.5)
    return output



# imread()不能读取路径包含中文的图片，此处先用numpy的fromfile()函数读取，再用OpenCV的imdecode()从内存的buffer中读取图片
img = cv.imdecode(np.fromfile(r'F:\Work and Learn\FIGHT\大三\数字图像处理\Lab\Lab2\lena.png', dtype=np.uint8), -1)
# 转成灰度图
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 获取图像高度和宽度
height = img.shape[0]
width = img.shape[1]
# 创建5幅图像进行计算变换
img1 = np.zeros((height, width), np.uint8)
img2 = np.zeros((height, width), np.uint8)
img3 = np.zeros((height, width), np.uint8)
img4 = np.zeros((height, width), np.uint8)
img5 = np.zeros((height, width), np.uint8)
# 进行灰度线性变换
for i in range(height):
    for j in range(width):
        img1[i,j] = transform(img[i,j], 1, 30)
        img2[i,j] = transform(img[i,j], 1.5, 0)
        img3[i,j] = transform(img[i,j], 0.2, 0)
        img4[i,j] = transform(img[i,j], -1, 255)
        img5[i,j] = transform(img[i,j], 1.5, 10)

showImages(nameWindow=None, img=img, title_img='original', img1=img1, title_img1='a=1, b=30', img2=img2, title_img2='a=1.5, b=0', img3=img3, title_img3='a=0.2, b=0', img4=img4, title_img4='a=-1, b=255', img5=img5, title_img5='a=1.5, b=10', savePath='task2_1.png')



# imread()不能读取路径包含中文的图片，此处先用numpy的fromfile()函数读取，再用OpenCV的imdecode()从内存的buffer中读取图片
img_airport = cv.imdecode(np.fromfile(r'F:\Work and Learn\FIGHT\大三\数字图像处理\Lab\Lab2\airport.png', dtype=np.uint8), -1)
img_airport = cv.cvtColor(img_airport, cv.COLOR_BGR2GRAY)
rows = img_airport.shape[0]
cols = img_airport.shape[1]
img_gamma = np.zeros((rows, cols), np.uint8)
for i in range(rows):
    for j in range(cols):
        img_gamma[i,j] = 5 * pow(img_airport[i,j], 0.8)

fig = plt.figure()

plt.subplot(1, 2, 1)
plt.title('original')
plt.imshow(img_airport, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('gamma transformation')
plt.imshow(img_gamma, cmap='gray')
plt.axis('off')

fig.subplots_adjust(wspace=0.5, hspace=0.5)
savefig(plt, "task2_2.png", 2.2 * img_airport.shape[1], 1.1 * img_airport.shape[0])
plt.show()



# imread()不能读取路径包含中文的图片，此处先用numpy的fromfile()函数读取，再用OpenCV的imdecode()从内存的buffer中读取图片
img_street = cv.imdecode(np.fromfile(r'F:\Work and Learn\FIGHT\大三\数字图像处理\Lab\Lab2\street.png', dtype=np.uint8), -1)
img_street = cv.cvtColor(img_street, cv.COLOR_BGR2GRAY)
img_log = log(42, img_street)

fig = plt.figure()

plt.subplot(1, 2, 1)
plt.title('original')
plt.imshow(img_street, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Log transformation')
plt.imshow(img_log, cmap='gray')
plt.axis('off')

fig.subplots_adjust(wspace=0.5, hspace=0.5)
savefig(plt, "task2_3.png", 2.2 * img_street.shape[1], 1.1 * img_street.shape[0])
plt.show()
