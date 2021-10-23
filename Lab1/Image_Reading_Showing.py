"""
Created on Thu Oct 21 2021

@author: douhaoYaz@东莞理工ACE实验室
"""

import cv2 as cv
import numpy as np
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt

# img = cv2.imread(r'F:\Work and Learn\FIGHT\大三\fruits.png')
# imread()不能读取路径包含中文的图片，此处先用numpy的fromfile()函数读取，再用OpenCV的imdecode()从内存的buffer中读取图片
img = cv.imdecode(np.fromfile(r'F:\Work and Learn\FIGHT\大三\数字图像处理\Lab\fruits.png', dtype=np.uint8), -1)

# 此处参考OpenCV官方文档，读取完图像后进行判空
if img is None:
    sys.exit("Could not read the image.")

# 打印图片的尺寸信息
print("图片的尺寸为 " + str(img.shape))

# 显示图片
cv.imshow("201841510108 FuHaoyang",img)

# returns the code of the pressed key
key = cv.waitKey(0)
# 需要使用ord函数返回字母“s”的ASCII编码再进行比较
if key == ord("s"):
    cv.imwrite("fruits.jpg", img)
    print("Successfully save image")
else:
    # 退出当前显示窗口
    cv.destroyAllWindows()
    # 用matplotlib窗口显示fruits.png
    # img_plt = Image.open(os.path.join(r'F:\Work and Learn\FIGHT\大三\数字图像处理\Lab\fruits.png'))
    # 将cv读入的BGR通道顺序图像转换成matplotlib默认的RGB通道顺序图像
    img_plt = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img_plt)
    plt.axis('off')
    plt.show()
    print("Successfully show image by matplotlib")

cv.destroyAllWindows()


