"""
Created on Thu Oct 21 2021

@author: douhaoYaz@东莞理工ACE实验室
"""

import numpy as np
import cv2 as cv

# 创建纯黑背景的彩色图片
img = np.zeros((720, 600, 3), np.uint8)

# 红色部分
cv.ellipse(img, (300, 140), (140, 140), 120, 0, 300, (0, 0, 255), cv.FILLED)
cv.circle(img, (300, 140), 55, (0, 0, 0), cv.FILLED)
# 绿色部分
cv.ellipse(img, (140, 415), (140, 140), 0, 0, 300, (0, 255, 0), cv.FILLED)
cv.circle(img, (140, 415), 55, (0, 0, 0), cv.FILLED)
# 蓝色部分
cv.ellipse(img, (460, 415), (140, 140), -60, 0, 300, (255, 0, 0), cv.FILLED)
cv.circle(img, (460, 415), 55, (0, 0, 0), cv.FILLED)

# 添加OpenCV文本
cv.putText(img, "OpenCV", (75, 675), cv.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 3)
# # 添加学号姓名
# cv.putText(img, "Created by 201841510108 Fuhaoyang", (300, 710), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
# # 加矩形框
# cv.rectangle(img, (298, 698), (585, 715), (0, 255, 0))

# 利用OpenCV的鼠标事件，实现鼠标在窗口的任意位置双击时，显示学号姓名和矩形框
def drawInfo(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        # 添加学号姓名
        cv.putText(img, "Created by 201841510108 Fuhaoyang", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255),1)
        # 加矩形框
        cv.rectangle(img, (x-2, y-12), (x+285, y+5), (0, 255, 0))

cv.namedWindow("OpenCV")
cv.setMouseCallback("OpenCV", drawInfo)

while(1):
    cv.imshow("OpenCV", img)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()

# # 显示图像
# cv.imshow("OpenCV Logo", img)
# cv.waitKey(0)

# 缩放图片到1/2，保存为logo.png
height, width = img.shape[:2]
height, width = int(0.5 * height), int(0.5 * width)
# numpy的格式是(height, width)，png的格式是(width, height)
img_save = cv.resize(img, (width,height))
cv.imshow("logo.png", img_save)
cv.waitKey(0)
cv.imwrite("logo.png", img_save)
