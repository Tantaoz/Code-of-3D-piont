import open3d as o3d
import numpy as np
import time
import cv2
"""点云数据的读取"""
def Read_pcd(fn_r):

    print("->正在加载点云... ")
    pcd = o3d.io.read_point_cloud(fn_r)
    # [center, covariance] = pcd.compute_mean_and_covariance()
    # center=np.array(center).astype(np.int)
    # print("点云的中心为：\n", center)
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

# 这个函数对算法没什么影响，只是突出表现一下特征点
def keypoints_to_spheres(keypoints):
        spheres = o3d.geometry.TriangleMesh()
        for keypoint in keypoints.points:
         sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
         sphere.translate(keypoint)
         spheres += sphere
         spheres.paint_uniform_color([1.0, 0.0, 0.0])
        return spheres

"""点云数据的生成"""

HEADER = '''\
# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgba
SIZE 4 4 4 4
TYPE F F F U
COUNT 1 1 1 1
WIDTH {}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {}
DATA ascii
'''

def write_pcd(points, save_pcd_path):
    n = len(points)

    lines = []
    for i in points:
        x=i[0]
        y=i[1]
        z=i[2]
        rgb=i[3]
        # x, y, z = points[i]

        lines.append('{:.16f} {:.16f} {:.16f} {} '.format( \
            x, y, z,rgb))
    with open(save_pcd_path, 'w') as f:
        f.write(HEADER.format(n, n))
        f.write('\n'.join(lines))


def emb_rgb(rgb,wm,keypoints_coor):
    List=keypoints_coor[:,0]+keypoints_coor[:,1]
    new_rgb=[]
    for i,j in zip (rgb,List):
        Index=int(j*100 % 4096)
        if i %2==wm[Index]:
            i==i
        elif i %2!=wm[Index] and i %10<5:
            i=i+1
        elif i %2!=wm[Index] and i %10>5:
            i=i-1
        new_rgb.append(i)

    return  new_rgb

def iss_feature(xyz,pcd):
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

    keypoint_indices = []   #特征点索引
    for keypoint in keypoints_coor:
        index = np.where((np.asarray(pcd.points) == keypoint).all(axis=1))[0]
        if len(index) > 0:
            keypoint_indices.append(index[0])
    print(len(keypoint_indices))
    print(len(set(keypoint_indices)))
    # from collections import Counter   #引入Counter
    # b = dict(Counter(keypoint_indices))
    # c=[key for key,value in b.items()if value > 1]  #只展示重复元素
    #
    # print ({key:value for key,value in b.items()if value > 1})  #展现重复元素和重复次数
    rgb=[]   #
    for i in range(len(keypoints_rgb)):
        R,G,B=keypoints_rgb[i]
        value = (int(R) << 16 | int(G) << 8 | int(B))
        rgb.append(value)
    # keyxyz=np.c_[keypoints_xyz,keypoints_rgb]
    # print(len(keyxyz))
    xyz_surplus=np.delete(xyz,keypoint_indices,axis=0) #删除特征点，剩余的为未分类的点
    print(len(xyz_surplus))
    return keypoints_coor,rgb,xyz_surplus
def zero_watermarking(madist,wm):
    block=[list() for i in range(4096)]
    w=4096 * [0]
    for i in madist:
        index=int(i*10000%4096)   #构建索引
        a=int(i%2)
        if a==1:
            block[index].append(1)   #投票原则
            w[index]+=1
        else:
            block[index].append(-1)
            w[index]-=1

    feature_matrix=[]   #转换成特征矩阵
    for j in w:
        if j ==1:
            feature_matrix.append(1)
        else:
            feature_matrix.append(0)
    print(feature_matrix)
    zero_img=[]
    for e,v in zip(feature_matrix,wm):
        if e==v:
            zero_img.append(255)
        else:
            zero_img.append(0)
    zero_img=np.asarray(zero_img)
    zero_img=zero_img.reshape((64,64))
    print(zero_img)
    cv2.imwrite(r"E:\digital watermarking\point cloud\feature points\madist_emb_zerowm.bmp",zero_img)



    return

if __name__ == '__main__':
    time_start = time.time()  # 记录开始时间
    wm=cv2.imread(r"E:\digital watermarking\point cloud\watermarking image\wm64.bmp",0) #读取原始水印图像
    wm[wm>0] =1              #水印图像二值化
    wm= list(wm.flatten())
    fn_r="data_d.pcd"
    pcd,xyz,num=Read_pcd(fn_r)  #原始点云数据的读取
    # """计算马氏距离，构建零水印"""
    # madist = pcd.compute_mahalanobis_distance()   #马氏距离计算
    # madist = np.array(madist)
    # print(madist)
    # zero_watermarking(madist,wm)


    """特征提取，颜色信息嵌入水印"""

    tic = time.time()
    keypoints_coor,rgb,xyz_surplus=iss_feature(xyz,pcd)
    new_rgb=emb_rgb(rgb,wm,keypoints_coor)   #在特征点的rgb信息上嵌入水印
    new_keypoints=np.c_[keypoints_coor,new_rgb]
    new_xyz=np.concatenate((xyz_surplus,new_keypoints),axis=0)
    fn_w="emb_data_d.pcd"
    write_pcd(new_xyz,fn_w,)        #写出嵌入水印后的点云数据

# pcd.paint_uniform_color([0.0, 1.0, 0.0])
# o3d.visualization.draw_geometries([keypoints_to_spheres(keypoints), pcd],width=800,height=800)

