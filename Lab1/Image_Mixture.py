"""
Created on Sat Oct 23 2021

@author: douhaoYaz@东莞理工ACE实验室
"""

import cv2 as cv
import numpy as np
import copy

# 读入lena.png
# imread()不能读取路径包含中文的图片，此处先用numpy的fromfile()函数读取，再用OpenCV的imdecode()从内存的buffer中读取图片
lena = cv.imdecode(np.fromfile(r'F:\Work and Learn\FIGHT\大三\数字图像处理\Lab\lena.png', dtype=np.uint8), -1)
# 读入logo.png
logo = cv.imdecode(np.fromfile(r'F:\Work and Learn\FIGHT\大三\数字图像处理\Lab\logo.png', dtype=np.uint8), -1)
# 通过logo.png的shape来决定lena的ROI大小
rows, columns, channels = logo.shape
lena_ROI = lena[:rows, :columns, :]

# 将两幅图像混合
alpha = 0.3
beta = (1.0 - alpha)
mixture = cv.addWeighted(lena_ROI, alpha, logo, beta, 0.0)

# 将混合后的图像覆盖到lena的副本lena_copy上，注意！一定要使用deepcopy，否则lena、lena_ROI
# 以及这里的lena_copy都是对同一片内存空间进行操作的，会使本来不想被混合的原图像lena和lena_ROI受到混合的效果
lena_1 = copy.deepcopy(lena)
lena_1[:rows, :columns, :] = mixture

# 将logo转换成灰度图
logo_gray = cv.cvtColor(logo, cv.COLOR_BGR2GRAY)
# 将logo的灰度图阈值化
ret, mask = cv.threshold(logo_gray, 10, 255, cv.THRESH_BINARY)
# 将掩模mask进行按位非运算
mask_inverse = cv.bitwise_not(mask)
# 通过按位与运算得到lena_ROI的背景部分
lena_bg = cv.bitwise_and(lena_ROI, lena_ROI, mask=mask_inverse)
# 通过按位与运算得到logo的前景部分
logo_fg = cv.bitwise_and(logo, logo, mask=mask)
# 将lena_ROI的背景部分和logo的前景部分叠加起来就能得到想要的图像ROI区域的效果
mixture_ROI = cv.add(lena_bg, logo_fg)
# 将处理后的ROI图像替换掉原始的lena图像，此处也对原始lena图像进行一个深拷贝
lena_2 = copy.deepcopy(lena)
lena_2[:rows, :columns, :] = mixture_ROI

# 用addWeighted来代替add来改变效果
mixture_ROI_new = cv.addWeighted(lena_ROI, 0.7, mixture_ROI, 0.3, 0.0)
# 将处理后的ROI图像替换掉原始的lena图像，此处也对原始lena图像进行一个深拷贝
lena_3 = copy.deepcopy(lena)
lena_3[:rows, :columns, :] = mixture_ROI_new

cv.imshow("lena_3", lena_3)
cv.waitKey(0)
