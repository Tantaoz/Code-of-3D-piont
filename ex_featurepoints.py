import open3d as o3d
import numpy as np
import time
import cv2
"""点云数据的读取"""
def Read_pcd(fn_r):

    print("->正在加载点云... ")
    pcd = o3d.io.read_point_cloud(fn_r)
    xyz=np.asarray(pcd.points)
    num=len(xyz)   #点的个数
    colors = np.asarray(pcd.colors)*255
    rgb=[]
    for i in range(len(colors)):
        R,G,B=colors[i]
        value = (int(R) << 16 | int(G) << 8 | int(B))
        rgb.append(value)
    xyz=np.c_[xyz,rgb]
    return pcd,xyz,num


def iss_feature(pcd):
    # --------------------ISS关键点提取的相关参数---------------------
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd,
                                                            salient_radius=0.2,
                                                            non_max_radius=0.2,
                                                            gamma_21=0.8,
                                                            gamma_32=0.5)
    toc = 1000 * (time.time() - tic)
    print("ISS Computation took {:.0f} [ms]".format(toc))
    keypoints_xyz=np.asarray(keypoints.points)   #读取特征点的坐标
    colors=np.asarray(keypoints.colors)*255  #特征点的颜色信息
    feature=np.c_[keypoints_xyz,colors]   #将颜色信息与坐标合并成二维数组

    keypoints_uniques = np.unique(feature,axis=0)   #去除重复点的坐标
    keypoints_coor=keypoints_uniques[:,0:3]   #特征点的坐标
    keypoints_rgb=keypoints_uniques[:,3:6]    #特征点的颜色

    rgb=[]   #
    for i in range(len(keypoints_rgb)):
        R,G,B=keypoints_rgb[i]
        value = (int(R) << 16 | int(G) << 8 | int(B))
        rgb.append(value)
    return keypoints_coor,rgb

def ex_WM(rgb,keypoints_xyz):
    wm=4096 * [0]
    block=[list() for i in range(4096)]
    List=keypoints_xyz[:,0]+keypoints_xyz[:,1]
    for i,j in zip(rgb,List):
        Index=int(j*100 % 4096)
        a=i%2
        wm[Index]+=a
        block[Index].append(a)
    Wm=[]
    for e in range(len(wm)):
        count=len(block[e])
        if wm[e]>count/2:
            Wm.append(1)
        else:
            Wm.append(0)
    print(Wm)
    return Wm
def wm_image(fn_w,wm):
    for q in range(0, len(wm)):
        if wm[q] > 0:
            wm[q] = 255
        else:
            wm[q] = 0
    W=np.array(wm)
    zli =W.reshape(64,64)
    tran02 = []
    for i in range(64):
        for j in range(64):
           if zli[i][j] == 255:
             xs = 255
           else:
             xs = 0
           tran02.append(xs)
    tran02=np.array(tran02)
    W=tran02.reshape(64,64)
    cv2.imwrite(fn_w,W)
#计算NC值
def value_nc(ima1,ima2):
    count=0
    for i,j in zip(ima1,ima2 ):
        if i==j:
            count+=1
        else:
            count+=0
    nc=count/len(ima1)
    return nc

if __name__ == '__main__':
    time_start = time.time()  # 记录开始时间
    fn_r="emb_data_c.pcd"
    pcd,xyz,num=Read_pcd(fn_r)  #原始点云数据的读取
    tic = time.time()
    keypoints_xyz,rgb=iss_feature(pcd)   #iss特征点提取
    wm=ex_WM(rgb,keypoints_xyz)
    fn_w="watermarking_image.bmp"
    wm_image(fn_w,wm)
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



