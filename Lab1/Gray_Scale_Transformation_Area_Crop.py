import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def showMultiImages(name_img, img_blue, img_green, img_red, img_gray):
    """利用matplotlib的add_subplot函数把图像的三个通道图像和灰度图像同时在一个窗口显示

    :param name_img: 窗口名称
    :param img_blue: 蓝色通道图像
    :param img_green: 绿色通道图像
    :param img_red: 红色通道图像
    :param img_gray: 灰色通道图像
    :return: None
    """

    # 创建画布figure
    fig = plt.figure(name_img)
    # 创建子图1
    subplot1 = fig.add_subplot(2, 2, 1)
    subplot1.set_title('blue')
    subplot1.imshow(img_blue, cmap='gray')
    plt.axis('off')

    # 创建子图2
    subplot2 = fig.add_subplot(2, 2, 2)
    subplot2.set_title('green')
    subplot2.imshow(img_green, cmap='gray')
    plt.axis('off')

    # 创建子图3
    subplot3 = fig.add_subplot(2, 2, 3)
    subplot3.set_title('red')
    subplot3.imshow(img_red, cmap='gray')
    plt.axis('off')

    # 创建子图4
    subplot4 = fig.add_subplot(2, 2, 4)
    subplot4.set_title('gray')
    subplot4.imshow(img_gray, cmap='gray')
    plt.axis('off')

    # 调整子图间距
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    # 显示图像
    plt.show()


if __name__ == '__main__':
    # 读入lena.png
    # imread()不能读取路径包含中文的图片，此处先用numpy的fromfile()函数读取，再用OpenCV的imdecode()从内存的buffer中读取图片
    img = cv.imdecode(np.fromfile(r'F:\Work and Learn\FIGHT\大三\数字图像处理\Lab\lena.png', dtype=np.uint8), -1)

    # 获取图像的BGR通道图像
    img_blue = img[:, :, 0]
    img_green = img[:, :, 1]
    img_red = img[:, :, 2]
    # 将三通道图像转换为灰度图
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 利用matplotlib的add_subplot函数把上述4幅图像同时在一个窗口显示
    showMultiImages('lena', img_blue, img_green, img_red, img_gray)


    # 截取脸部
    x1 = 192
    y1 = 193
    x2 = 366
    y2 = 374
    img_face = img[y1:y2, x1:x2]
    cv.imwrite('face.png', img_face)
    cv.imshow('lena face', img_face)
    cv.waitKey(0)

    # 截取帽子部分
    x1 = 108
    y1 = 47
    x2 = 408
    y2 = 201
    img_hat = img[y1:y2, x1:x2]
    # 获取红色通道
    img_hat_red = img_hat[:, :, 2]
    cv.imshow('red hat', img_hat_red)
    cv.waitKey(0)

    # 将帽子部分蓝色通道清除（置为0）
    img_hat[:, :, 0] = 0
    # 获取帽子部分BGR通道图像
    hat_blue = img_hat[:, :, 0]
    hat_green = img_hat[:, :, 1]
    hat_red = img_hat[:, :, 2]
    # 将三通道图像转换为灰度图
    hat_gray = cv.cvtColor(img_hat, cv.COLOR_BGR2GRAY)
    # 在一个窗口分别显示其R，G，B通道图片和进行灰度变换后的图片
    showMultiImages('hat', hat_blue, hat_green, hat_red, hat_gray)

