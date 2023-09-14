import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
import cv2

# 二值化水印（0，1）
def Erzhi_watermark(arnold_img):
    for i in range(0,arnold_img.shape[0]):
        for j in range(0,arnold_img.shape[1]):
            if arnold_img[i][j]>=100:
                arnold_img[i][j]=1
            else:
                arnold_img[i][j]=0
    return arnold_img

def zero_watermarking(madist):
    block=[list() for i in range(4096)]
    w=4096 * [0]
    for i in madist:
        index=int(i*10000%4096)   #构建索引
        a=int(i*100%2)
        if a==1:
            block[index].append(1)   #投票原则
            w[index]+=1
        else:
            block[index].append(-1)
            w[index]-=1

    feature_matrix=[]   #转换成特征矩阵
    for j in w:
        if j >0:
            feature_matrix.append(1)
        else:
            feature_matrix.append(0)
    print(feature_matrix)

    watermark=np.array(feature_matrix).reshape(64,64)
    return watermark

# 构造零水印
def Zero_Watermark(watermark1,watermark2):
    zero_img=np.zeros((watermark1.shape[0],watermark1.shape[1]))
    for i in range(0,watermark1.shape[0]):
        for j in range(0,watermark1.shape[1]):
            if watermark1[i][j]==watermark2[i][j]:
                zero_img[i][j]=255
            else:
                zero_img[i][j]=0
    return zero_img
#计算NC值
def value_nc(ima1,ima2):
    # s_wm = np.sqrt(sum(ima1** 2))
    # s_wm1 = np.sqrt(sum(ima2 ** 2))
    # cji = sum(ima1*ima2)
    # nc = cji / (s_wm * s_wm1)
    count=0
    for i,j in zip(ima1,ima2 ):
        if i==j:
            count+=1
        else:
            count+=0
    nc=count/len(ima1)
    return nc
if __name__=="__main__":

    # -------------------------读取点云-----------------------------
    pcd = o3d.io.read_point_cloud("E:\digital watermarking\point cloud/feature points/result/geomatric attack/a_rotate_x300.pcd")
    print(pcd)
    # -----------------------计算马氏距离------------------------
    madist = pcd.compute_mahalanobis_distance()
    madist = np.array(madist)
    print(madist)

    old_xulie=cv2.imread(r"E:\digital watermarking\point cloud\feature points\ex-wm\madist_a.bmp",0)
    erzhi_old=Erzhi_watermark(old_xulie)
    feature_matrix=zero_watermarking(madist)   #特征矩阵生成
    fan_zero=Zero_Watermark(feature_matrix,erzhi_old)  #异或求原始水印
    fn_w="E:\digital watermarking\point cloud/feature points\ex-wm\ex_a_zerowm.bmp"

    cv2.imwrite(fn_w,fan_zero)
    plt.subplot(221)
    plt.imshow(old_xulie, 'gray')
    plt.title("old_xulie")

    plt.subplot(222)
    plt.imshow(feature_matrix, 'gray')
    plt.title("new_xulie")

    plt.subplot(223)
    plt.imshow(fan_zero, 'gray')
    plt.title("fan_zero")


    plt.show()
    im1 = 'E:\digital watermarking\point cloud\watermarking image/GIS64.bmp'

    img1 = cv2.imread(im1)
    im1=img1[:,:,0]

    img2 = cv2.imread(fn_w)
    im2=img2[:,:,0]

    im1[im1>0] =1              #水印图像二值化
    wm1= list(im1.flatten())

    im2[im2 > 0] = 1
    wm2 = list(im2.flatten())

    get_nc = value_nc(wm1,wm2)
    print('图像NC值：',get_nc)




# """提取零水印"""
# # -------------------------读取点云-----------------------------
# pcd1 = o3d.io.read_point_cloud("bunny+noise.pcd")
#
# # -----------------------计算马氏距离------------------------
# madist1 = pcd1.compute_mahalanobis_distance()
# madist1 = np.array(madist1)
# print(madist1)
# zero_wm=cv2.imread(r"E:\digital watermarking\point cloud\feature points\madist_emb_zerowm.bmp",0) #读取零水印图像
# zero_wm[zero_wm>0] =1              #水印图像二值化
# zero_wm= list(zero_wm.flatten())
# ex_zero_img=zero_watermarking(madist1,zero_wm)
# cv2.imwrite(r"E:\digital watermarking\point cloud\feature points\ex_zerowm.bmp",ex_zero_img)
