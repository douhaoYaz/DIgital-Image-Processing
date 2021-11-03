# -*- coding: utf-8 -*-
"""
Created on 11 03 17:45 2021

@author: douhaoYaz@DGUT ACE Lab
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
# from Lab2.gray_Level_Transformations import showImages

def savefig(plt, filename, width=1024, height=768, dpi=300):
    fig = plt.gcf()
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    fig.set_size_inches(width / dpi, height /dpi)
    fig.text(0.5, 0.5, '201841510108 符浩扬', fontsize=10, rotation=0, color='red', ha='right', va='bottom', alpha=0.7)
    fig.savefig(filename, format='png', transparent=True, dpi=300, pad_inches=0)

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



# imread()不能读取路径包含中文的图片，此处先用numpy的fromfile()函数读取，再用OpenCV的imdecode()从内存的buffer中读取图片
img = cv.imdecode(np.fromfile(r'F:\Work and Learn\FIGHT\大三\数字图像处理\Lab\Lab2\lena.png', dtype=np.uint8), -1)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 添加高斯噪声
img_gaussian = random_noise(img, mode='gaussian')
img_gaussian = np.array(255*img_gaussian, dtype = 'uint8')

# 添加盐噪声
img_salt = random_noise(img, mode='salt', amount=0.1)
img_salt = np.array(255*img_salt, dtype = 'uint8')

# 添加椒噪声
img_pepper = random_noise(img, mode='pepper', amount=0.1)
img_pepper = np.array(255*img_pepper, dtype = 'uint8')

# 添加椒盐噪声
img_sp = random_noise(img, mode='s&p', amount=0.1)
img_sp = np.array(255*img_sp, dtype = 'uint8')

# 添加speckle噪声
img_speckle = random_noise(img, mode='s&p', amount=0.1)
img_speckle = np.array(255*img_speckle, dtype = 'uint8')

# TODO showImage函数是gray_Level_Transformations.py里定义的函数，调用该函数不知道为何会输出那个py文件里结果
showImages(None, img, 'original', img_gaussian, 'Gaussian Noise', img_salt, 'Salt Noise', img_pepper, 'Pepper Noise', img_sp, 'SP Noise', img_speckle, 'Speckle Noise', 'task3_1.png')

# 高斯噪声经过滤波后
img_gaussian_blur = cv.blur(img_gaussian, (3,3))
img_gaussian_box = cv.boxFilter(img_gaussian, -1, (7,7), normalize=1)
img_gaussian_gaus = cv.GaussianBlur(img_gaussian, (7,7), 7)
img_gaussian_med = cv.medianBlur(img_gaussian, 3)

showImages(None, img, 'original', img_gaussian, 'gaussian noise', img_gaussian_blur, 'average filter', img_gaussian_box, 'box filter', img_gaussian_gaus, 'gaussian filter', img_gaussian_med, 'media filter', 'task3_2.png')

# 盐噪声经过滤波后
img_salt_blur = cv.blur(img_salt, (3,3))
img_salt_box = cv.boxFilter(img_salt, -1, (7,7), normalize=1)
img_salt_gaus = cv.GaussianBlur(img_salt, (7,7), 7)
img_salt_med = cv.medianBlur(img_salt, 3)

showImages(None, img, 'original', img_salt, 'salt noise', img_salt_blur, 'average filter', img_salt_box, 'box filter', img_salt_gaus, 'gaussian filter', img_salt_med, 'media filter', 'task3_3.png')

# 椒噪声经过滤波后
img_pepper_blur = cv.blur(img_pepper, (3,3))
img_pepper_box = cv.boxFilter(img_pepper, -1, (7,7), normalize=1)
img_pepper_gaus = cv.GaussianBlur(img_pepper, (7,7), 7)
img_pepper_med = cv.medianBlur(img_pepper, 3)

showImages(None, img, 'original', img_pepper, 'pepper noise', img_pepper_blur, 'average filter', img_pepper_box, 'box filter', img_pepper_gaus, 'gaussian filter', img_pepper_med, 'media filter', 'task3_4.png')

# 椒盐噪声
img_sp_blur = cv.blur(img_sp, (3,3))
img_sp_box = cv.boxFilter(img_sp, -1, (7,7), normalize=1)
img_sp_gaus = cv.GaussianBlur(img_sp, (7,7), 7)
img_sp_med = cv.medianBlur(img_sp, 3)

showImages(None, img, 'original', img_sp, 'salt&pepper noise', img_sp_blur, 'average filter', img_sp_box, 'box filter', img_sp_gaus, 'gaussian filter', img_sp_med, 'media filter', 'task3_5.png')

# speckle噪声
img_speckle_blur = cv.blur(img_speckle, (3,3))
img_speckle_box = cv.boxFilter(img_speckle, -1, (7,7), normalize=1)
img_speckle_gaus = cv.GaussianBlur(img_speckle, (7,7), 7)
img_speckle_med = cv.medianBlur(img_speckle, 3)

showImages(None, img, 'original', img_speckle, 'speckle noise', img_speckle_blur, 'average filter', img_speckle_box, 'box filter', img_speckle_gaus, 'gaussian filter', img_speckle_med, 'media filter', 'task3_6.png')


