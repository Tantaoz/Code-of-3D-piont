import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

"""生成混沌序列"""


def Logistic(N, Xor, a, t0):
    X0 = Xor
    Lx, RcLx = [], []
    # 混沌序列Lx生成
    for s in range(0, t0 + N):
        X0 = a * X0*(1 - X0)
        Lx.append(X0)
    cLx = Lx[t0:]  # 舍去固定次数产生的迭代值
    # 向下取整
    for j in cLx:
        xi = int(N * j) % N
        RcLx.append(xi)  # 取整后的混沌序列
    # #  随机序列处理
    return RcLx


"""水印图像处理"""


def Watermark_deal(img_deal):
    h, w = img_deal.shape
    for i in range(0, h):
        for j in range(0, w):
            if img_deal[i][j] < 100:
                img_deal[i][j] = 0
            else:
                img_deal[i][j] = 255
    return img_deal


"""加密"""


def Encrypt(List_watermark, RcLx):
    Li_water = List_watermark
    for i in range(0, len(RcLx)):
        temp = Li_water[i]
        Li_water[i] = Li_water[RcLx[i]]
        Li_water[RcLx[i]] = temp
    return Li_water


"""解密"""

def Dencrypt(En_watermark, RcLx, N):
    Re_water=En_watermark
    r = 1
    for j in reversed(RcLx):
        temp=Re_water[j]
        Re_water[j]=Re_water[N - r]
        Re_water[N - r] = temp
        r += 1
    return Re_water


if __name__ == '__main__':
    img1 = cv2.imread(r'E:\digital watermarking\point cloud\watermarking image\GIS64.bmp', 0)  # 读取水印图像
    img2 = cv2.imread(r'E:\digital watermarking\point cloud\feature points\ex-wm\Tent_luan.jpg', 0)  # 读取水印图像
    deal_img1 = Watermark_deal(img1)
    deal_img2 = Watermark_deal(img2)
    List_watermark1 = deal_img1.flatten()  # 降维
    List_watermark2 = deal_img2.flatten()  # 降维
    N, Xor, a, t0 = 4096, 0.35, 3.65, 100
    RcL = Logistic(N, Xor, a, t0)
    En_watermark = Encrypt(List_watermark1, RcL)
    De_watermark = Dencrypt(List_watermark2, RcL, N)
    Array_Z = np.array(En_watermark).reshape(64, 64)  # 加密后
    Array = np.array(De_watermark).reshape(64, 64)  # 解密后
    # 显示置乱水印
    plt.subplot(221)
    plt.imshow(Array_Z, 'gray')
    plt.title("jiami")
    cv2.imwrite("E:\digital watermarking\point cloud/feature points\ex-wm\Tent_luan.jpg", Array_Z)

    plt.subplot(222)
    plt.imshow(Array, 'gray')
    plt.title("jiemi")
    plt.show()
