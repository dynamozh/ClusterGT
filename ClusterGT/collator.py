import torch
import random
from torch_geometric.loader import ClusterData


def pad_1d_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


# def pad_attn_bias_unsqueeze(x, padlen):
#     xlen = x.size(0)
#     if xlen < padlen:
#         new_x = x.new_zeros(
#             [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
#         new_x[:xlen, :xlen] = x
#         new_x[xlen:, :xlen] = 0
#         x = new_x
#     return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(2)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


class Batch():
    def __init__(self, x_inter, x_intra, y):
        super(Batch, self).__init__()
        self.x_inter, self.x_intra, self.y = x_inter, x_intra, y
        # self.attn_bias = attn_bias
        # self.ids = ids

    def to(self, device):
        self.x_inter, self.x_intra, self.y = self.x_inter.to(device), self.x_intra.to(device), self.y.to(device)
        # self.attn_bias = self.attn_bias.to(device)
        # self.ids = self.ids.to(device)
        return self

    def __len__(self):
        return self.y.size(0)


def get_partition_index(element_id, cluster_data: ClusterData) -> int:
    # Find the partition index for the given element ID
    for idx in range(len(cluster_data.partition.partptr) - 1):
        node_start = int(cluster_data.partition.partptr[idx])
        node_end = int(cluster_data.partition.partptr[idx + 1])

        if node_start <= element_id < node_end:
            return idx

    # If the element ID does not belong to any partition
    raise ValueError(f"Element ID {element_id} does not belong to any partition.")


def calculate_partition_mean_feature(partition_idx, cluster_data: ClusterData) -> torch.Tensor:
    # Get the start and end indices of nodes in the specified partition
    node_start = int(cluster_data.partition.partptr[partition_idx])
    node_end = int(cluster_data.partition.partptr[partition_idx + 1])

    # Extract node features for the specified partition
    partition_features = cluster_data.data.x[node_start:node_end]

    # Calculate the mean of node features
    mean_features = torch.mean(partition_features, dim=0)

    return mean_features


def concatenate_similar_features(node_id, cluster_data: ClusterData) -> torch.Tensor:
    # Get the partition index for the given node ID
    partition_idx = get_partition_index(node_id, cluster_data)

    # Get the start and end indices of nodes in the specified partition
    node_start = int(cluster_data.partition.partptr[partition_idx])
    node_end = int(cluster_data.partition.partptr[partition_idx + 1])

    # Extract node features for the specified partition
    partition_features = cluster_data.data.x[node_start:node_end]

    # Extract features for the given node
    node_features = cluster_data.data.x[node_id].unsqueeze(0)

    # Find the index of the given node in the partition features
    node_index_in_partition = node_id - node_start

    # Remove the duplicated node features from the partition features
    partition_features = torch.cat([
        partition_features[:node_index_in_partition],
        partition_features[node_index_in_partition + 1:]
    ], dim=0)

    # Concatenate node features for the specified partition with the given node
    concatenated_features = torch.cat([node_features, partition_features], dim=0)

    return concatenated_features


def calculate_max_partition_nodes(cluster_data: ClusterData) -> int:
    max_nodes = 0

    for partition_idx in range(len(cluster_data.partition.partptr) - 1):
        # Get the start and end indices of nodes in the specified partition
        node_start = int(cluster_data.partition.partptr[partition_idx])
        node_end = int(cluster_data.partition.partptr[partition_idx + 1])

        # Calculate the number of nodes in the partition
        num_nodes = node_end - node_start

        # Update the maximum number of nodes if needed
        max_nodes = max(max_nodes, num_nodes)

    return max_nodes


def random_dropout_rows_3d(tensor, dropout_ratio):
    # 获取tensor的维度
    num_dims, num_rows, num_cols = tensor.size()

    # 计算需要保留的行数
    num_remain_rows = (num_rows - 1) - int(dropout_ratio * (num_rows - 1))

    # 生成一个与行数相同的随机索引，除了第一行之外
    remain_indices = torch.randperm(num_rows - 1)[:num_remain_rows] + 1

    # 使用torch.cat删除对应索引的行
    tensor = torch.cat([tensor[:, 0:1, :], tensor[:, remain_indices, :]], dim=1)

    return tensor


# def collator(items, data, perturb=False):
#     all_partition_mean_feature_list = []
#     for partition_idx in range(len(data.partition.partptr) - 1):
#         # Append the mean features to the list
#         mean_feature = calculate_partition_mean_feature(partition_idx, data)
#         all_partition_mean_feature_list.append(mean_feature)
#
#     max_node_num = calculate_max_partition_nodes(data) + len(data.partition.partptr) - 1
#     y = torch.cat([data.data.y[item].unsqueeze(0) for item in items])
#     batch_feature_list = []
#     for item in items:
#         item_partition_idx = get_partition_index(item,data)
#         item_feature = concatenate_similar_features(item, data)
#         # concatenate other partition
#         for partition_idx in range(len(data.partition.partptr) - 1):
#             if partition_idx != item_partition_idx:
#                 item_feature = torch.cat([item_feature, all_partition_mean_feature_list[partition_idx].unsqueeze(0)], dim=0)
#         batch_feature_list.append(item_feature)
#
#     x = torch.cat([pad_2d_unsqueeze(batch_feature_list[i], max_node_num) for i in range(len(batch_feature_list))])
#     if perturb:
#         x += torch.FloatTensor(x.shape).uniform_(-0.1, 0.1)
#
#     return Batch(
#         # attn_bias=None,
#         x=x,
#         y=y,
#     )

# 交替/并行版
# def collator(items, data, perturb=False):
#     # all_partition_mean_feature_list = []
#     # for partition_idx in range(len(data.partition.partptr) - 1):
#     #     # Append the mean features to the list
#     #     mean_feature = calculate_partition_mean_feature(partition_idx, data)
#     #     all_partition_mean_feature_list.append(mean_feature)
#     start_time = time.time()
#     max_node_num = max(len(item[0]) for item in items)
#     y = torch.tensor([], dtype=torch.int64)
#     x_intra = torch.tensor([])
#     x_inter_id = torch.tensor([], dtype=torch.int64)
#
#     for item in items:
#         x_intra_ids = item[0]
#         x_inter_ids = item[1]
#         y = torch.cat([y, data.data.y[x_intra_ids[0]].unsqueeze(0)], dim=0)
#         x_intra = torch.cat([x_intra, pad_2d_unsqueeze(data.data.x[x_intra_ids], max_node_num)], dim=0)
#
#         # inter_feature = data.data.x[x_intra_ids[0]].unsqueeze(0)
#         # for partition_idx in x_inter_ids:
#         #     inter_feature = torch.cat([inter_feature, all_partition_mean_feature_list[partition_idx].unsqueeze(0)], dim=0)
#         x_inter_id = torch.cat([x_inter_id, torch.tensor(x_inter_ids, dtype=torch.int64).unsqueeze(0)], dim=0)
#
#     if perturb:
#         # x_inter += torch.FloatTensor(x_inter.shape).uniform_(-0.1, 0.1)
#         x_intra += torch.FloatTensor(x_intra.shape).uniform_(-0.1, 0.1)
#
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"data loader took {elapsed_time:.2f} seconds to run.")
#
#     return Batch(
#         # attn_bias=None,
#         x_inter=x_inter_id,
#         x_intra=x_intra,
#         y=y,
#     )

# 加速改造版
def collator(items, data, shuffle=False, perturb=False):
    # start_time = time.time()

    # all_partition_mean_feature_list = []
    # for partition_idx in range(len(data.partition.partptr) - 1):
    #     # Append the mean features to the list
    #     mean_feature = calculate_partition_mean_feature(partition_idx, data)
    #     all_partition_mean_feature_list.append(mean_feature)
    all_partition_mean_feature_list = data.mean_feature_list
    if shuffle:
        random.shuffle(items)

    max_node_num = max(len(item[0]) for item in items)
    y = torch.cat([data.data.y[item[0][0]].unsqueeze(0) for item in items])

    # attn_bias = torch.cat([pad_attn_bias_unsqueeze(item[2], max_node_num) for item in items])
    x_intra = torch.cat([pad_2d_unsqueeze(data.data.x[item[0]], max_node_num) for item in items])
    # x_inter_id = torch.cat([torch.tensor(item[1], dtype=torch.int64).unsqueeze(0) for item in items])

    inter_feature_list = []
    for item in items:
        inter_feature = torch.cat(
            [all_partition_mean_feature_list[partition_idx].unsqueeze(0) for partition_idx in item[1]])
        inter_feature = torch.cat([data.data.x[item[0][0]].unsqueeze(0), inter_feature], dim=0)
        inter_feature_list.append(inter_feature)
    x_inter = torch.cat([inter_feature.unsqueeze(0) for inter_feature in inter_feature_list])

    if perturb:
        x_inter += torch.FloatTensor(x_inter.shape).uniform_(-0.1, 0.1)
        x_intra += torch.FloatTensor(x_intra.shape).uniform_(-0.1, 0.1)

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"data loader took {elapsed_time:.2f} seconds to run.")

    return Batch(
        # attn_bias=attn_bias,
        x_inter=x_inter,
        x_intra=x_intra,
        y=y,
    )

# def collator(items, data, perturb=False):
#     batch_list = []
#     for item in items:
#         for x in item:
#             batch_list.append((x[0], x[1], x[2][0]))
#     if shuffle:
#         random.shuffle(batch_list)
#     attn_biases, xs, ys = zip(*batch_list)
#     max_node_num = max(i.size(0) for i in xs)
#     y = torch.cat([i.unsqueeze(0) for i in ys])
#     x = torch.cat([pad_2d_unsqueeze(feature[i], max_node_num) for i in xs])
#     ids = torch.cat([i.unsqueeze(0) for i in xs])
#     if perturb:
#         x += torch.FloatTensor(x.shape).uniform_(-0.1, 0.1)
#     attn_bias = torch.cat([i.unsqueeze(0) for i in attn_biases])
#
#     return Batch(
#         attn_bias=attn_bias,
#         x=x,
#         y=y,
#     )
