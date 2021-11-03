# -*- coding: utf-8 -*-
"""
Created on Tue Nov 2 09:26:21 2021

@author: douhaoYaz@东莞理工ACE实验室
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




# imread()不能读取路径包含中文的图片，此处先用numpy的fromfile()函数读取，再用OpenCV的imdecode()从内存的buffer中读取图片
img = cv.imdecode(np.fromfile(r'F:\Work and Learn\FIGHT\大三\数字图像处理\Lab\Lab2\island.png', dtype=np.uint8), -1)
# 分别获取图像RGB通道
img_r = img[:, :, 2]
img_g = img[:, :, 1]
img_b = img[:, :, 0]

# # 使用numpy统计直方图
# hist, bins = np.histogram(img_r.ravel(), 256, [0, 256])

# TODO 使用OpenCV统计直方图

# # 使用OpenCV统计直方图并使用Matplotlib绘制直方图
# color = ('b', 'g', 'r')
# x = np.arange(256)
# for i, col in enumerate(color):
#     histr = cv.calcHist([img], [i], None, [256], [0, 256])
#     plt.plot(histr, color=col)
#     plt.fill_between(x, histr[:,0], color=col)
#     plt.xlim([0, 256])
# plt.show()

# 使用Matplotlib的hist函数画出来
row = 1
col = 2
fig = plt.figure(figsize=(col * img.shape[1] / 300, row * 1.1 * img.shape[0] / 300), dpi=300)

plt.subplot(row, col, 1)
plt.title('original')
plt.imshow(img[:, :, ::-1])
plt.axis('off')

plt.subplot(row, col, 2)
plt.title('Histogram_RGB')
plt.hist(img_b.ravel(), 256, [0, 256], facecolor='b', edgecolor='b', histtype='stepfilled', alpha=0.7, stacked=True)
plt.hist(img_g.ravel(), 256, [0, 256], facecolor='g', edgecolor='g', histtype='stepfilled', alpha=0.7, stacked=True)
plt.hist(img_r.ravel(), 256, [0, 256], facecolor='r', edgecolor='r', histtype='stepfilled', alpha=0.7, stacked=True)
# plt.imshow()
plt.axis('off')

# 保存
savefig(plt, "task1_1.png", col * img.shape[1], row * 1.1 * img.shape[0])
plt.show()



# 转换为灰度图像，计算其直方图和
row = 1
col = 2
fig = plt.figure(figsize=(col * img.shape[1] / 300, row * 1.1 * img.shape[0] / 300), dpi=300)

plt.subplot(row, col, 1)
plt.title('Origin')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.imshow(img_gray, cmap='gray')
plt.axis('off')

plt.subplot(row, col, 2)
plt.title('Histogram')
hist1, bins = np.histogram(img_gray.ravel(), 256, [0, 256])
cdf = hist1.cumsum()
cdf_normalised = cdf * float(hist1.max() / cdf.max())
plt.hist(img_gray.ravel(), 256, [0, 256], facecolor='black')
plt.plot(cdf_normalised, color='blue')
plt.axis('off')
# 保存
savefig(plt, "task1_2.png", col * img.shape[1], row * 1.1 * img.shape[0])
plt.show()



# 直方图均衡化
row = 1
col = 2
fig = plt.figure(figsize=(col * img.shape[1] / 300, row * 1.1 * img.shape[0] / 300), dpi=300)

plt.subplot(row, col, 1)
plt.title('Equalized')
img_equ = cv.equalizeHist(img_gray)
plt.imshow(img_equ, cmap='gray')
plt.axis('off')

plt.subplot(row, col, 2)
plt.title('Histogram')
hist_equ, bins = np.histogram(img_equ.ravel(), 256, [0, 256])
cdf = hist_equ.cumsum()
cdf_normalised = cdf * float(hist_equ.max() / cdf.max())
plt.hist(img_equ.flatten(), 256, [0, 256], facecolor='black')
plt.plot(cdf_normalised, color='blue')
plt.axis('off')
# 保存
savefig(plt, "task1_3.png", col * img.shape[1], row * 1.1 * img.shape[0])
plt.show()



# CLAHE自适应直方图均衡化
row = 1
col = 2
fig = plt.figure(figsize=(col * img.shape[1] / 300, row * 1.1 * img.shape[0] / 300), dpi=300)

plt.subplot(row, col, 1)
plt.title('Adaptive Equalized')
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img_gray)
plt.imshow(img_clahe, cmap='gray')
plt.axis('off')

plt.subplot(row, col, 2)
plt.title('Histogram')
hist_clahe, bins = np.histogram(img_clahe.ravel(), 256, [0, 256])
cdf = hist_clahe.cumsum()
cdf_normalised = cdf * float(hist_clahe.max() / cdf.max())
plt.hist(img_clahe.flatten(), 256, [0, 256], facecolor='black')
plt.plot(cdf_normalised, color='blue')
plt.axis('off')
# 保存
savefig(plt, "task1_4.png", col * img.shape[1], row * 1.1 * img.shape[0])
plt.show()



# 将原始的灰度图片，自动均衡化后的图像和采用自适应均衡化后的图像显示在一起比较
fig = plt.figure('img')
subplot1 = fig.add_subplot(1, 3, 1)
subplot1.imshow(img_gray, cmap='gray')
plt.axis('off')
subplot2 = fig.add_subplot(1, 3, 2)
subplot2.imshow(img_equ, cmap='gray')
plt.axis('off')
subplot3 = fig.add_subplot(1, 3, 3)
subplot3.imshow(img_clahe, cmap='gray')
plt.axis('off')
# 调整子图间距
fig.subplots_adjust(wspace=0, hspace=0)
# 保存
savefig(plt, "task1_5.png", 3 * img.shape[1], 1 * 1.1 * img.shape[0])
plt.show()
