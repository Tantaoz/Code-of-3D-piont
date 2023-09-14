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
        lines.append('{:.16f} {:.16f} {:.16f} {} '.format( \
            x, y, z,rgb))
    with open(save_pcd_path, 'w') as f:
        f.write(HEADER.format(n, n))
        f.write('\n'.join(lines))

def emb_rgb(wm,xyz,index):
    for i in range(len(index)):
        a=xyz[index[i]]
        coordinate=a[0:3]
        # rgb=xyz[index[i],[-1]]
        # q=rgb%2
        List=coordinate[0]+coordinate[1]
        Index=int(List*100 % 4096)
        if xyz[index[i],[-1]]  %2==wm[Index]:
            xyz[index[i],[-1]] ==xyz[index[i],[-1]]
        elif xyz[index[i],[-1]]  %2!=wm[Index] and xyz[index[i],[-1]]  %10<5:
            xyz[index[i],[-1]] =xyz[index[i],[-1]] +1
        elif xyz[index[i],[-1]]  %2!=wm[Index] and xyz[index[i],[-1]]  %10>5:
            xyz[index[i],[-1]] =xyz[index[i],[-1]] -1

    return  xyz

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

    return keypoint_indices



if __name__ == '__main__':
    time_start = time.time()  # 记录开始时间
    wm=cv2.imread(r"E:\digital watermarking\point cloud\watermarking image\GIS64.bmp",0) #读取原始水印图像
    wm[wm>0] =1              #水印图像二值化
    wm= list(wm.flatten())
    fn_r="data_c.pcd"
    pcd,xyz,num=Read_pcd(fn_r)  #原始点云数据的读取

    """特征提取，颜色信息嵌入水印"""

    tic = time.time()
    index=iss_feature(xyz,pcd)
    new_xyz=emb_rgb(wm,xyz,index)   #在特征点的rgb信息上嵌入水印
    fn_w="emb_data_c.pcd"
    write_pcd(new_xyz,fn_w,)        #写出嵌入水印后的点云数据

# pcd.paint_uniform_color([0.0, 1.0, 0.0])
# o3d.visualization.draw_geometries([keypoints_to_spheres(keypoints), pcd],width=800,height=800)

