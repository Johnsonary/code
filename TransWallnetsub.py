import pandas as pd
import torch,copy
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neighbors import KDTree
from glob import glob
import h5py
import open3d as o3d
import os
import json
from tqdm import tqdm
from time import time
import random
import matplotlib.pyplot as plt
import laspy
import pickle
from thop import profile   # 计算模型参数量和FLOPs
from datetime import datetime
current_time = datetime.now()
month = current_time.strftime('%m')
day = current_time.strftime('%d')
hour = current_time.strftime('%H')
minute = current_time.strftime('%M')
dirname =str(month+day+hour+minute)

'''
hyperparam
'''
num_class = 5
classes = ['wall', 'column', 'window', 'bulge', 'cross']
class2label = {cls: i for i,cls in enumerate(classes)}
seg_label_to_cat = {}
seg_classes = class2label
labelweights = [0 for _ in range(num_class)]
total_I = [0 for _ in range(num_class)]
total_U = [0 for _ in range(num_class)]
for i,cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat
savenum = 0
pkldir = "/home/ubuntu/PointNeXt/TW/08201918_TW"
# pkl = "ep160_miou0.695821.pkl"
pkl = "ep24_miou0.628210.pkl"
pklpath = os.path.join(pkldir,pkl)
savepath = "./TW/%s_TW/" % (dirname)
def draw(points, labels=None):
    global pkl
    global savepath
    global savenum

    color_map = {
        0: np.array([102,171,250])/255, # wall
        1: np.array([82,227,117])/255, # column
        2: np.array([255,125,75])/255, # window
        3: np.array([197,82,227])/255, # bulge
        4: np.array([251,236,90])/255, # cross
    }
    points = np.asarray(points)
    pt = o3d.geometry.PointCloud()
    pt.points = o3d.utility.Vector3dVector(points)
    if labels is not None:
        label = np.asarray(labels)
        # colors = plt.get_cmap("tab20")(label / 5)
        colors = np.array([color_map[i] for i in label])
        # colors = plt.get_cmap("tab20")(label)
        pt.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pt], window_name='open3d', width=500, height=500)
    o3d.io.write_point_cloud(os.path.join(pkldir,'%s_'%savenum+pkl.split('_')[1][:9])+'.ply',pt, write_ascii=True)
    print(os.path.join(pkldir,'%s_'%savenum+pkl.split('_')[1][:9])+'.ply','saving succeeded')
    savenum+=1

class S3DISDataset(Dataset):
    '''
    def __init__(self, root="./data/S3DIS", num_point=40960, sub_grid_size=0.04, split="train",
                 test_area=5, transform=None, num_layers=4, subsample_ratio=[4, 4, 4, 2], k=16):

    return:
        x, neighbor_index, subsample_index, upsample_index, feature, label
     #  x: [num_point, 3]   neighbor_index: [num_point, k]   subsample_index: [num_point//stride, k]  upsample_index: [num_point, 1]
     #  feature: [num_point, 4]   label: [num_point,]
    '''

    def __init__(self, root="/home/ubuntu/PointNeXt/S3DIS", num_point=40960, sub_grid_size=0.04, split="train",
                 test_area=4, transform=None, num_layers=4, subsample_ratio=[4, 4, 4, 2], k=16):
        '''
        sub_grid_size: 以sub_grid_size m³为一个网格，在内随机选择一个点，代表该网格
        num_layers: 深度神经网络Encoder层的数目
        subsample_ratio: 下采样点数比例
        k: k近邻
        '''
        self.root = root
        self.num_point = num_point  # 所需采样的点数
        self.sub_grid_size = sub_grid_size
        self.split = split
        self.transform = transform  # 数据增强
        self.num_layers = num_layers  # Encoder层的数目
        self.subsample_ratio = subsample_ratio
        self.k = k  # k近邻

        self.original_path = root + "/" + "original"
        self.tree_path = root + "/" + "sub_grid_sample"

        self.test_proj = []
        self.test_proj_label = []

        self.trees = []
        self.colors = []
        self.labels = []

        self.possibility = []
        self.min_possibility = []
        '''
        test
        '''
        self.check = []
        self.num = 0
        self.query_index = []

        original_files = [os.path.basename(file) for file in glob(
            self.original_path + "/*.h5")]  # ['Area_1_conferenceRoom_1.h5', 'Area_1_conferenceRoom_2.h5', ..., 'Area_6_pantry_1.h5']
        if split == "train":
            original_files = [file for file in original_files if int(file.split("_")[1]) != test_area]
        else:
            original_files = [file for file in original_files if int(file.split("_")[1]) == test_area]

        for original_file in original_files:
            filename = original_file.split(".")[0]  # 'Area_1_conferenceRoom_1'

            # Read sub-sampled point cloud data
            f = h5py.File(self.tree_path + "/" + filename + ".h5", "r")
            sub_points = np.array(f["data"])  # [S, 6]  XYZRGB
            sub_labels = np.array(f["label"])  # [S,]    L
            f.close()

            # Read search tree data
            with open(self.tree_path + "/" + filename + "_KDTree.pkl", "rb") as f:
                search_tree = pickle.load(f)

            self.trees.append(search_tree)  # 由sub-sample数据生成的search_tree，search_tree.data获取sub-sample数据
            self.colors.append(sub_points[:, 3:6])  # [S, 3]
            self.labels.append(sub_labels)  # [S,]

            if split != "train":  # Test
                with open(self.tree_path + "/" + filename + "_proj.pkl", "rb") as f:
                    proj_index, proj_labels = pickle.load(f)  # [N,]  [N,]
                self.test_proj.append(proj_index)
                self.test_proj_label.append(proj_labels)

            '''
            test
            '''
        for color in self.colors:
            possi = np.random.rand(color.shape[0]) * 1e-3  # [S,]  range:[0, 0.001)
            check = np.zeros(color.shape[0])
            self.check.append(check)
            self.possibility.append(possi)  # 对每个点云文件都存储 [S,] 随机数
            self.min_possibility.append(min(possi))  # 对每个点云文件都存储 1 个随机数
            self.query_index.append(np.array([]))

    def trees(self):
        return np.array(self.trees)
    def labels(self):
        return self.labels

    def __getitem__(self, index):

        # 根据 min_possibility 选取点云
        pc_index = np.argmin(self.min_possibility)
        # 根据 pc_index 在possibility内 选取 一个点
        p_index = np.argmin(self.possibility[pc_index])

        # 根据 pc_index 获取 sub-sample 的点云坐标数据
        points = np.array(self.trees[pc_index].data)  # [S, 3]
        # 根据 p_index 获取 sub-sample 的某个点坐标数据
        center_point = points[p_index, :].reshape(1, -1)  # [1, 3]

        # 给 center_point 添加噪声
        noise = np.random.normal(scale=0.35, size=center_point.shape)  # [1, 3]
        pick_point = center_point + noise  # [1, 3]

        # 检查 points 的点数是符合 num_point
        if points.shape[0] < self.num_point:  # 下采样点数 < num_point
            query_index = self.trees[pc_index].query(pick_point, k=points.shape[0])[1][0].astype(
                np.int32)  # [points.shape[0],]
            supplement_index = np.random.choice(query_index, self.num_point - points.shape[0])
            query_index = np.concatenate([query_index, supplement_index])  # [num_point,]

        else:
            query_index = self.trees[pc_index].query(pick_point, k=self.num_point)[1][0].astype(
                np.int32)  # [num_point,]

        # 打乱 query_index
        shuffle_index = np.arange(len(query_index))
        np.random.shuffle(shuffle_index)
        query_index = query_index[shuffle_index]


        '''
        test
        '''
        query_index_out = copy.deepcopy(self.query_index)
        query_index_out[pc_index] = np.append(query_index_out[pc_index],query_index)

        # 根据 query_index 获取相应的坐标数据、颜色数据以及标签值
        query_xyz = points[query_index]  # [num_point, 3]
        query_height = query_xyz[:, -1:]  # [num_point, 1]  高度特征信息
        query_xyz = query_xyz - pick_point  # [S', 3]
        query_color = self.colors[pc_index][query_index]  # [num_point, 3]
        query_label = self.labels[pc_index][query_index]  # [num_point,]

        '''
        test
        '''
        self.check[pc_index][query_index] = 1
        if (min([min(self.check[i]) for i in range(len(self.check))])!=0) and (self.num == 0):
            self.num += 1
            print('fully test')

        # 到此为止，进行了以下步骤：
        # 1、首先，随机选取了一个点云
        # 2、在该点云内，随机选取一个点
        # 3、以该点为中心检索出距离其最近的k个点，获取对应索引
        # 4、根据索引，获取坐标数据、颜色数据以及标签值，并且将索引点的坐标相对化。

        # 根据邻近点与中心点的距离，更新 possibility 数据（距离越近，增加越多。增加范围：[0, 1]）
        dists = np.sum(query_xyz ** 2, axis=-1)  # [num_point,]
        delta = (1 - dists / max(dists)) ** 2
        self.possibility[pc_index][query_index] += delta
        self.min_possibility[pc_index] = min(self.possibility[pc_index])






        query_xyz = query_xyz.astype(np.float32)
        feature = np.concatenate([query_color, query_height], axis=-1).astype(np.float32)  # [num_point, 4]
        label = query_label.astype(np.int32)

        x = []
        neighbor_index = []
        subsample_index = []
        upsample_index = []

        for i in range(self.num_layers):
            neighbor_idx = KDTree(query_xyz,metric='chebyshev').query(query_xyz, k=self.k, return_distance=False).astype(np.int32)
            # [num_point, k]

            # 对点云进行下采样 这里就是随机采样了吧?
            sub_x = query_xyz[:query_xyz.shape[0] // self.subsample_ratio[i], :]  # [num_point//stride, 3]
            sub_idx = neighbor_idx[:query_xyz.shape[0] // self.subsample_ratio[i], :]  # [num_point//stride, k]

            # 检索 query_xyz在sub_x内的最近点索引
            up_index = KDTree(sub_x).query(query_xyz, k=1, return_distance=False).astype(np.int32)
            # [num_point, 1]

            x.append(query_xyz)
            neighbor_index.append(neighbor_idx)
            subsample_index.append(sub_idx)
            upsample_index.append(up_index)

            query_xyz = sub_x  # [num_point//stride, 3]

        # print(self.num)
        return x, neighbor_index, subsample_index, upsample_index, feature, label, query_index_out
        #  x: [num_point, 3]   neighbor_index: [num_point, k]   subsample_index: [num_point//stride, k]  upsample_index: [num_point, 1]
        #  feature: [num_point, 4]   label: [num_point,]

    def __len__(self):
        iter = np.sum([len(self.trees[i].data) for i in range(len(self.trees))]) // self.num_point
        iter = iter *4*4 #降采样比例
        if self.split == "train":
            # return len(self.trees) * 24  # 源代码设置为1000，因内存不够只能设置这么大
            return iter
        else:
            # return len(self.trees) * 10
            return iter



#Network Architecture
def square_distance(src, dst):
    '''
        src：central_coords
        dst：coords

        计算两点之间的欧氏距离
        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
             = sum(src**2,dim=-1) + sum(dst**2,dim=-1) - 2*src^T*dst

        输入：
            src：[B, S, C]
            dst：[B, N, C]

        返回：两组点间，两两之间的距离
            dist：[B, S, N]
    '''
    B, S, _ = src.shape
    _, N, _ = dst.shape

    dists = -2 * (src @ dst.transpose(2, 1))  # [B, S, C] @ [B, C, N] ==> [B, S, N]
    dists = dists + torch.sum(src ** 2, dim=-1).view(B, S,
                                                     1)  # torch.sum(src ** 2, dim=-1).shape：[B, S]   dist.shape：[B, S, N]
    dists = dists + torch.sum(dst ** 2, dim=-1).view(B, 1,
                                                     N)  # torch.sum(dst ** 2, dim=-1)：[B, N]    dist.shape：[B, S, N]

    return dists
def chebyshev_distance(src, dst):
    src_expanded = src.unsqueeze(2)
    dst_expanded = dst.unsqueeze(2)
    diff = torch.abs(src_expanded - dst_expanded)
    dists = torch.max(diff, dim=-1)[0]
    return dists
def index_points(coords, index):
    '''
    获取索引点的坐标

    coords：[B, N, 3]
    index：[B, S]    也可能为 [B, S, K]

    返回；
        new_coords    [B, S, 3]    也可能为 [B, S, K, 3]
    '''
    device = coords.device

    B = coords.shape[0]

    index = index.long()

    view_shape = list(index.shape)  # [B, S]
    view_shape[1:] = [1] * (len(view_shape) - 1)  # 变为 [B, 1]     后续需要使用  .view().repeat()
    repeat_shape = list(index.shape)  # [B, S]
    repeat_shape[0] = 1  # [1, S]

    batch_index = torch.arange(0, B, dtype=torch.long).view(view_shape).repeat(repeat_shape).to(device)  # [B, S]

    new_coords = coords[batch_index, index]

    return new_coords
def knn(k, xyz, central_xyz):
    '''
    K nearest neighborhood

    在所有点中，现在要搜索出距离中心点最近的k个点

    输入：
        k：局部区域最大的采样数量
        xyz：所有的点，[B, N, C]
        central_xyz：每个neighborhood中的中心点，即被采样出的点，[B, S, C]

    输出：
        group_idx：分组点的索引，[B, S, k]    包含在每个neighbor中的所有索引
    '''
    dists = square_distance(central_xyz, xyz)  # [B, S, N]

    elements, index = torch.topk(dists, k, dim=-1, largest=False)  # 返回k个最小值（elements：元素；index：元素所在位置）

    return index
class AttentivePool(nn.Module):
    '''
    def __init__(self, input_channel, output_channel):

    def forward(self, feature):   # [B, C, N, k]

    return:
        result   # [B, output_channel, N]
    '''

    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.Wq = nn.Sequential(
            nn.Conv1d(input_channel, input_channel, 1, bias=False),
            nn.BatchNorm1d(input_channel),
            nn.ReLU(inplace=True)
        )
        self.Wk = nn.Sequential(
            nn.Conv1d(input_channel, input_channel, 1, bias=False),
            nn.BatchNorm1d(input_channel),
            nn.ReLU(inplace=True)
        )
        self.Wv = nn.Sequential(
            nn.Conv1d(input_channel, input_channel, 1, bias=False),
            nn.BatchNorm1d(input_channel),
            nn.ReLU(inplace=True)
        )
        self.mlp1 = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, 1, bias=False),
            nn.Softmax(dim=-1)
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, 1, bias=False),
            nn.BatchNorm1d(output_channel, eps=1e-06, momentum=0.99),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x_info, feature, neighbor_index):  # [B, C, N, k]

        Wq = self.Wq(feature)
        Wk = self.Wk(feature)
        Wv = self.Wv(feature)
        Wk = index_points(Wk.permute(0, 2, 1), neighbor_index).permute(0, 3, 1, 2)  # [B, 3, N, k]
        # Wq = torch.mean(Wq, dim=-1)
        Ww = Wk - Wq.unsqueeze(-1)
        W = x_info * Ww
        Wv = index_points(Wv.permute(0, 2, 1), neighbor_index).permute(0, 3, 1, 2)  # [B, 3, N, k]
        feature = F.softmax(W, dim=-1) * Wv
        # score = self.mlp1(feature)  # [B, C, N, k]
        # feature = feature * score  # [B, C, N, k]

        feature = torch.sum(feature, dim=-1)  # [B, C, N]

        result = self.mlp2(feature)  # [B, output_channel, N]

        return result  # [B, output_channel, N]
class FeaturePropagation(nn.Module):
    '''
    def __init__(self, input_channel, mlp):

    def forward(self, feature, encoder_feature, interp_index):   # [B, C1, N//stride]   [B, C2, N]   [B, N, 1]

    return:
        new_points   # [B, out, N]
    '''

    def __init__(self, input_channel, mlp):
        super().__init__()
        self.mlp_conv = nn.ModuleList()
        self.mlp_bn = nn.ModuleList()
        for output_channel in mlp:
            self.mlp_conv.append(nn.ConvTranspose1d(input_channel, output_channel, 1))
            self.mlp_bn.append(nn.BatchNorm1d(output_channel))
            input_channel = output_channel

    def forward(self, feature, encoder_feature, interp_index):  # [B, C1, N//stride]   [B, C2, N]   [B, N, 1]

        interp_feature = index_points(feature.permute(0, 2, 1), interp_index).permute(0, 3, 1, 2).squeeze(
            -1)  # [B, C1, N]

        feature = torch.cat([encoder_feature, interp_feature], dim=1)  # [B, C1+C2, N]

        for conv, bn in zip(self.mlp_conv, self.mlp_bn):
            new_points = F.leaky_relu(bn(conv(feature)), 0.2, True)

        return new_points  # [B, out, N]
class LSEA(nn.Module):
    '''
    def __init__(self, input_channel, output_channel):

    def forward(self, x, neighbor_index, feature):   # [B, 3, N]  [B, N, k]  [B, C, N]

    return:
        feature   # [B, 2*output_channel, N]
    '''

    def __init__(self, input_channel, output_channel,first=False):
        super().__init__()
        qkv_bias=True
        self.first = first
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channel, output_channel // 2, 1, bias=False),
            nn.BatchNorm1d(output_channel // 2, eps=1e-06, momentum=0.99),
            nn.LeakyReLU(0.2, True)
        )

        self.lse1 = nn.Sequential(
            nn.Conv2d(10, output_channel // 2, 1, bias=False),
            nn.BatchNorm2d(output_channel // 2, eps=1e-06, momentum=0.99),
            nn.LeakyReLU(0.2, True)
        )
        self.pool0 = AttentivePool(8, 8)
        self.pool1 = AttentivePool(output_channel // 2, output_channel // 2)

        self.lsebroadx = nn.Sequential(
            nn.Conv2d(input_channel , output_channel , 1, bias=False),
            nn.BatchNorm2d(output_channel , eps=1e-06, momentum=0.99),
            nn.LeakyReLU(0.2, True)
        )



        self.lse2 = nn.Sequential(
            nn.Conv2d(output_channel // 2, output_channel // 2, 1, bias=False),
            nn.BatchNorm2d(output_channel // 2, eps=1e-06, momentum=0.99),
            nn.LeakyReLU(0.2, True)
        )
        self.pool2 = AttentivePool(output_channel//2, output_channel)

        self.mlp2 = nn.Sequential(
            nn.Conv1d(output_channel, 2 * output_channel, 1, bias=False),
            nn.BatchNorm1d(2 * output_channel, eps=1e-06, momentum=0.99),
        )
        self.res = nn.Sequential(
            nn.Conv1d(input_channel, 2 * output_channel, 1, bias=False),
            nn.BatchNorm1d(2 * output_channel, eps=1e-06, momentum=0.99),
        )

    def forward(self, x, neighbor_index, feature):  # [B, 3, N]  [B, N, k]  [B, C, N]
        B, C, N = feature.shape
        _, _, k = neighbor_index.shape

        identity = feature  # [B, C, N]

        # =================================================================================================
        feature = self.mlp1(feature)  # [B, output_channel//2, N]
        center_x = x.unsqueeze(-1).repeat(1, 1, 1, k)  # [B, 3, N, k]
        knn_x = index_points(x.permute(0, 2, 1), neighbor_index).permute(0, 3, 1, 2)  # [B, 3, N, k]

        distance = torch.sqrt(torch.sum((knn_x - center_x) ** 2, dim=1, keepdim=True))  # [B, 1, N, k]
        x_info = torch.cat((center_x, knn_x, knn_x - center_x, distance), dim=1)  # [B, 10, N, k]

        # LocalSpatialEncoding1 & AttentivePool1
        x_info = self.lse1(x_info)  # [B, output_channel//2, N, k]
        if(self.first):
            # no color feature 8->16
            # feature = self.lsebroadx(x_info)
            x_info0 = self.pool0(x_info, feature, neighbor_index)
            identity = x_info0
            # feature = torch.cat([x_info, x_info], dim=1)  # [B, output_channel, N, k]
        # else:
        #     knn_feature = index_points(feature.permute(0, 2, 1), neighbor_index).permute(0, 3, 1,
        #                                                                                  2)  # [B, output_channel//2, N, k]
        #     feature = torch.cat([x_info, knn_feature], dim=1)  # [B, output_channel, N, k]




        feature = self.pool1(x_info,feature,neighbor_index)  # [B, output_channel//2, N]
        # feature = feature + self.nl1(feature, feature)   # [B, output_channel, N]

        # LocalSpatialEncoding2 & AttentivePool2
        x_info = self.lse2(x_info)  # [B, output_channel//2, N, k]
        # knn_feature = index_points(feature.permute(0, 2, 1), neighbor_index).permute(0, 3, 1,
        #                                                                              2)  # [B, output_channel//2, N, k]
        # feature = torch.cat([x_info, knn_feature], dim=1)  # [B, output_channel, N, k]
        feature = self.pool2(x_info,feature,neighbor_index)  # [B, output_channel, N]
        # feature = feature + self.nl2(feature, feature)   # [B, output_channel, N]

        # 残差
        feature = self.mlp2(feature)  # [B, 2*output_channel, N]
        res = self.res(identity)  # [B, 2*output_channel, N]
        feature = F.leaky_relu(feature + res, 0.2, True)  # [B, 2*output_channel, N]
        # =================================================================================================

        return feature  # [B, 2*output_channel, N]
class Encoder(nn.Module):
    '''
    def __init__(self, input_channel, output_channel):

    def forward(self, x, neighbor_index, subsample_index, feature):   # [B, 3, N]  [B, N, k]  [B, S, k]  [B, C, N]

    return:
        feature, feature_firstlayer   # [B, 2*output_channel, S]
    '''

    def __init__(self, input_channel, output_channel, first = False):
        super().__init__()

        self.lsea = LSEA(input_channel, output_channel, first=first)

    def forward(self, x, neighbor_index, subsample_index, feature):  # [B, 3, N]  [B, N, k]  [B, S, k]  [B, C, N]

        feature = self.lsea(x, neighbor_index, feature)  # [B, 2*out, N]

        feature_firstlayer = feature

        # sub-sample
        feature = index_points(feature.permute(0, 2, 1), subsample_index).permute(0, 3, 1, 2)  # [B, 2*out, S, k]
        feature = torch.max(feature, dim=-1)[0]  # # [B, 2*out, S]

        return feature, feature_firstlayer  # [B, 2*output_channel, S]
class semseg_network(nn.Module):
    # def __init__(self, seg_num=13):
    def __init__(self, seg_num=num_class):
        super().__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(4, 8, 1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True)
        )

        self.encoder1 = Encoder(8, 16, first=True)
        self.encoder2 = Encoder(32, 64)
        self.encoder3 = Encoder(128, 128)
        self.encoder4 = Encoder(256, 256)
        self.encoder5 = Encoder(512, 512)

        self.mlp = nn.Sequential(
            nn.Conv1d(1024, 1024, 1, bias=False),
            nn.BatchNorm1d(1024, eps=1e-06, momentum=0.99),
            nn.ReLU(True)
        )

        self.fp5 = FeaturePropagation(1536, [512])
        self.fp4 = FeaturePropagation(768, [256])
        self.fp3 = FeaturePropagation(384, [128])
        self.fp2 = FeaturePropagation(160, [32])
        self.fp1 = FeaturePropagation(64, [32])

        self.segMLP = nn.Sequential(
            nn.Conv1d(32, 64, 1, bias=False),
            nn.BatchNorm1d(64, eps=1e-06, momentum=0.99),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(64, 32, 1, bias=False),
            nn.BatchNorm1d(32, eps=1e-06, momentum=0.99),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Conv1d(32, seg_num, 1)
        )

    def forward(self, x, neighbor_index, subsample_index, upsample_index, feature):
        feature0 = self.conv0(feature)  # [B, 8, N]

        feature1, feature0 = self.encoder1(x[0], neighbor_index[0], subsample_index[0],
                                           feature0)  # [B, 3, N//4]  [B, 32, N//4]
        feature2, _ = self.encoder2(x[1], neighbor_index[1], subsample_index[1],
                                    feature1)  # [B, 3, N//16]   [B, 128, N//16]
        feature3, _ = self.encoder3(x[2], neighbor_index[2], subsample_index[2],
                                    feature2)  # [B, 3, N//64]   [B, 256, N//64]
        feature4, _ = self.encoder4(x[3], neighbor_index[3], subsample_index[3],
                                    feature3)  # [B, 3, N//256]    [B, 512, N//256]
        feature5, _ = self.encoder5(x[4], neighbor_index[4], subsample_index[4],
                                    feature4)  # [B, 3, N//512]    [B, 1024, N//512]

        feature5 = self.mlp(feature5)  # [B, 1024, N//512]

        feature4 = self.fp5(feature5, feature4, upsample_index[4])  # [B, 512, N//256]
        feature3 = self.fp4(feature4, feature3, upsample_index[3])  # [B, 256, N//64]
        feature2 = self.fp3(feature3, feature2, upsample_index[2])  # [B, 128, N//16]
        feature1 = self.fp2(feature2, feature1, upsample_index[1])  # [B, 32, N//4]
        feature0 = self.fp1(feature1, feature0, upsample_index[0])  # [B, 32, N]

        result = self.segMLP(feature0)  # [B, seg_num, N]

        return result  # [B, seg_num, N]
# lossfunction
def cal_loss(pred, label, smoothing=True):
    '''
    计算交叉熵巡损失，若有需要，则使用标签平滑（label smoothing）
    pred: (B, 40) 或 (B×2048, 50)
    label: (B) 或 (B×2048)

    :return: 单一元素
    '''
    if smoothing:
        label = label.view(-1, 1)  # (B×2048, 1)

        eps = 0.2
        n_class = pred.shape[1]
        # torch.zeros_like(pred): (B×2048, 50)
        one_hot = torch.zeros_like(pred).scatter(1, label.long(), 1)  # (B×2048, 50)
        '''
        Cross Entropy Loss = -Σ pᵢ log(qᵢ)
        原先：预测正确记 pᵢ=1   预测错误记 pᵢ=0
        现在把 pᵢ 换成 下面2个
        预测正确时，pᵢ = 1 换成 pᵢ = 1 - eps
        预测错误时，pᵢ = 0 换成 pᵢ = eps / (n_class - 1)
        '''
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prob = F.log_softmax(pred, dim=-1)  # (B×2048, 50)

        loss = -(one_hot * log_prob).sum(dim=-1).mean()  # .sum(dim=-1): (B×2048)
    else:
        loss = F.cross_entropy(pred, label.long(), reduction="mean")

    return loss

def focal_loss(pred, target, smoothing=True):
    alpha = 0.25
    gamma = 2
    # ce = torch.nn.CrossEntropyLoss(weight=None, reduction='none')
    # logp = ce(pred, target)
    # p = torch.exp(-logp)
    # loss = alpha * (1 - p) ** gamma * logp
    # return loss.mean()
    target = target.to(torch.int64)
    pred_softmax =torch.softmax(pred,dim=1)
    target_onehot = torch.zeros_like(pred_softmax)
    target_onehot.scatter_(1,target.view(-1,1),1)
    cross_entropy = -(target_onehot*torch.log(pred_softmax+1e-8))
    focal_loss = alpha * torch.pow(1-pred_softmax,gamma)*cross_entropy
    return torch.sum(focal_loss)/pred.shape[0]
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
def caculate_sem_IOU(pred, label):
    '''
    pred: [B × 4096]
    label: [B × 4096]
    '''
    # I_all = torch.zeros(13)
    # U_all = torch.zeros(13)
    I_all = torch.zeros(num_class)
    U_all = torch.zeros(num_class)
    P_all = torch.zeros(num_class)
    R_all = torch.zeros(num_class)
    F1_all = torch.zeros(num_class)

    # for sem in range(13):
    for sem in range(num_class):
        TP = torch.sum(torch.logical_and(pred == sem, label == sem))
        FP = torch.sum(torch.logical_and(pred == sem, label != sem))
        FN = torch.sum(torch.logical_and(pred != sem, label == sem))
        I_all[sem] = I_all[sem] + TP
        U_all[sem] = U_all[sem] + TP + FP + FN
        P_all[sem] = TP / (TP + FP + 1e-6)
        R_all[sem] = TP / (TP + FN + 1e-6)
        F1_all[sem] = 2 * (P_all[sem] * R_all[sem]) / (P_all[sem] + R_all[sem] + 1e-6)
        labelweights[sem] = int(torch.sum((label==sem)))
        total_I[sem] = int(TP)
        total_U[sem] = int(TP + FP + FN)


    # # 为什么要删除指定列？
    # # 因为：
    # #     1、有些label，没有在数据集中出现过，所以需要删除
    # #     2、若不删除指定列，会导致IOU为 nan
    # '''==== delete specific columns ===='''
    # # 需要删除的列索引
    # del_index = []
    # # 转换为 array，便于删除指定列
    # I_all = np.array(I_all)
    # U_all = np.array(U_all)
    # # 获取 需要删除的 列
    # I_index = np.nonzero(I_all == 0)[0].tolist()
    # U_index = np.nonzero(U_all == 0)[0].tolist()
    # for i in I_index:
    #     if i in U_index:
    #         del_index.append(i)
    # # 删除
    # I_all = np.delete(I_all, del_index)
    # U_all = np.delete(U_all, del_index)
    # # array转换为tensor
    # I_all = torch.tensor(I_all)
    # U_all = torch.tensor(U_all)
    # '''==== delete specific columns ===='''

    return I_all / (U_all+ 1e-6), P_all, R_all, F1_all
def eval_show():
    global labelweights
    setup_seed(1)


    '''
    test
    '''
    test_loader = DataLoader(S3DISDataset(num_point=8192, split="test", transform=None, num_layers=5,
                                          subsample_ratio=[4, 4, 4, 4, 2]),
                             1, True, drop_last=True, num_workers=1, pin_memory=True)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    START_EPOCH = 0
    EPOCH = 300
    lr = 0.01

    model = semseg_network().to(device)
    optimizer = optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
    # scheduler = MultiStepLR(optimizer, [int(EPOCH*0.6), int(EPOCH*0.8)], gamma=0.1)
    scheduler = CosineAnnealingLR(optimizer, EPOCH, 1e-5)
    criterion = cal_loss
    # criterion = focal_loss

    best_test_iou = 0

    '''
    test
    '''
    # 加载断点，继续训练
    path = pklpath
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    best_test_iou = checkpoint["best_test_iou"]
    START_EPOCH = checkpoint["epoch"] + 1
    '''hyparam2save'''
    hyparam = " "
    epoch = START_EPOCH



    '''
    Val
    '''
    val_start = time()

    test_loss = 0
    count = 0
    test_pred_seg = []
    test_true_seg = []
    test_pred_iou = []
    test_true_iou = []
    '''
    test
    '''
    whole_points = []
    whole_labels = []
    origin_labels = []
    for i in range(len(test_loader.dataset.trees)):
        whole_points.append(np.array(test_loader.dataset.trees[i].data))
        origin_labels.append(np.array(test_loader.dataset.labels[i]))
        # whole_labels.append(np.array([-1]*len(whole_points[i])))
        whole_labels.append(np.zeros((len(whole_points[i]), num_class)))
    with torch.no_grad():
        model.eval()
        '''
        test
        '''
        for x, neighbor_index, subsample_index, upsample_index, feature, semseg, query_index in tqdm(test_loader,
                                                                                    total=len(test_loader)):
            B = x[0].shape[0]

            for i in range(len(x)):
                x[i] = x[i].to(device, non_blocking=True)
                x[i] = x[i].transpose(2, 1).float()  # [B, 3, N]
            feature = feature.to(device, non_blocking=True)
            feature = feature.transpose(2, 1).float()
            semseg = semseg.to(device, non_blocking=True)
            semseg = semseg.int()

            pred_semseg = model(x, neighbor_index, subsample_index, upsample_index, feature)

            pred_semseg = pred_semseg.transpose(2, 1).contiguous()  # [B, 4096, 13]

            '''
            test
            '''
            for i in range(len(query_index)):
                if len(query_index[i][0])!=0:
                    whole_labels[i][np.array(query_index[i][0],dtype=int)] += np.array(pred_semseg[0].to('cpu'))

            '''Draw part test'''
            # points = x[0].transpose(2, 1).to('cpu')[0]
            # labels = np.argmax(pred_semseg.to('cpu')[0], axis=-1)
            # draw(points,labels)

            # loss = criterion(pred_semseg.view(-1, 13), semseg.view(-1))
            loss = criterion(pred_semseg.view(-1, num_class), semseg.view(-1))

            count = count + B

            pred_semseg_class = pred_semseg.max(-1)[1]  # [B, 4096]

            test_loss = test_loss + loss.item() * B

            pred_semseg_class = pred_semseg_class.cpu()
            semseg = semseg.cpu()

            test_pred_seg.append(pred_semseg_class.view(-1))  # [B×4096]
            test_true_seg.append(semseg.view(-1))  # [B×4096]
            test_pred_iou.append(pred_semseg_class.view(-1))  # [B×4096]
            test_true_iou.append(semseg.view(-1))  # [B×4096]


        '''
        test
        '''
        val_end = time()
        labels = []
        for i in range(len(whole_labels)):
            labels.append(np.argmax(whole_labels[i], axis=-1))
            draw(whole_points[i], labels[i])


        test_pred_seg = torch.cat(test_pred_seg)
        test_true_seg = torch.cat(test_true_seg)
        test_pred_iou = torch.cat(test_pred_iou)
        test_true_iou = torch.cat(test_true_iou)

        acc = accuracy_score(test_true_seg, test_pred_seg)
        avg_acc = balanced_accuracy_score(test_true_seg, test_pred_seg)

        iou, P_all, R_all, F1_all = caculate_sem_IOU(test_pred_iou, test_true_iou)

        # os.mkdir('./TW/%s_TW' % dirname) if not os.path.exists('./TW/%s_TW' % dirname) else None

        outstr = "Test: %s, loss: %s, acc: %s, avg acc: %s, mIOU: %s, mF1: %s time-consuming: %s" % (
        str(epoch), str(test_loss / count),
        str(acc), str(avg_acc),
        str(torch.mean(iou).item()), str(torch.mean(F1_all).item()), str(val_end - val_start))
        print(outstr)
        # with open("./InterLA/test.txt", "a") as f:
        labelnum=labelweights
        labelweights = labelweights / np.sum(labelweights)
        for l in range(num_class):
            clsoutstr='test_class %s weight: %.3f, IoU: %.5f, F1: %.5f, Pre: %.5f, Rec: %.5f,' % (
                seg_label_to_cat[l] + ' ' * (num_class + 1 - len(seg_label_to_cat[l])), labelweights[l],
                iou[l], F1_all[l], P_all[l], R_all[l])
            print(clsoutstr,  end='\n')
            with open("./TW/%s/test_%s.txt" % (pkldir.split('/')[-1], pkl.split('_')[1][:9]), "a") as f:
                f.write(clsoutstr)
                f.write("\n")
        with open("./TW/%s/test_%s.txt" % (pkldir.split('/')[-1], pkl.split('_')[1][:9]), "a") as f:
            f.write(outstr)
            f.write("\n")
        # os.mkdir('./TW/%s_TW' % dirname) if not os.path.exists('./TW/%s_TW' % dirname) else None
        '''
        test
        '''
        iou, P_all, R_all, F1_all = caculate_sem_IOU(torch.tensor(np.concatenate(labels)),
                                                     torch.tensor(np.concatenate(origin_labels)))
        outstr = "avgTest: %s, loss: %s, acc: %s, avg acc: %s, mIOU: %s, mF1: %s time-consuming: %s" % (
        str(epoch), str(test_loss / count),
        str(acc), str(avg_acc),
        str(torch.mean(iou).item()), str(torch.mean(F1_all).item()), str(val_end - val_start))
        print(outstr)
        # with open("./InterLA/test.txt", "a") as f:
        labelnum=labelweights
        labelweights = labelweights / np.sum(labelweights)
        for l in range(num_class):
            clsoutstr='avgtest_class %s weight: %.3f, IoU: %.5f, F1: %.5f, Pre: %.5f, Rec: %.5f,' % (
                seg_label_to_cat[l] + ' ' * (num_class + 1 - len(seg_label_to_cat[l])), labelweights[l],
                iou[l], F1_all[l], P_all[l], R_all[l])
            print(clsoutstr,  end='\n')
            with open("./TW/%s/test_%s.txt" % (pkldir.split('/')[-1],pkl.split('_')[1][:9]), "a") as f:
                f.write(clsoutstr)
                f.write("\n")
        with open("./TW/%s/test_%s.txt" % (pkldir.split('/')[-1],pkl.split('_')[1][:9]), "a") as f:
            f.write(outstr)
            f.write("\n")


eval_show()