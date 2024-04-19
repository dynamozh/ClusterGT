import torch
import random
from torch_geometric.loader import ClusterData
import numpy as np
import scipy.sparse as sp
from time import perf_counter
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from torch_geometric.utils import to_undirected


# def concatenate_ids(node_id, cluster_data: ClusterData, all_partition_power_adj_list):
#     # Get the partition index for the given node ID
#     partition_idx = get_partition_index(node_id, cluster_data)
#     inter_id_list = list(range(len(cluster_data.partition.partptr) - 1))
#     inter_id_list.remove(partition_idx)
#
#     # Get the start and end indices of nodes in the specified partition
#     node_start = int(cluster_data.partition.partptr[partition_idx])
#     node_end = int(cluster_data.partition.partptr[partition_idx + 1])
#
#     # 创建一个列表从 node_start 到 node_end
#     intra_id_list = list(range(node_start, node_end))
#
#     # 移除 node_id 如果它在列表中
#     intra_id_list.remove(node_id)
#
#     # random dropout
#     # intra_id_list = random_delete_elements(intra_id_list, 0.2)
#     # inter_id_list = random_delete_elements(inter_id_list, 0.8)
#
#     # select top k
#     intra_k = int(len(intra_id_list) * 0.8)
#     inter_k = int(len(inter_id_list) * 0.2)
#
#     intra_similarity_list = []
#     inter_similarity_list = []
#     for intra_id in intra_id_list:
#         # 计算余弦相似度
#         # cosine_similarity函数期望输入是二维数组，因此需要对向量进行reshape
#         similarity = cosine_similarity(cluster_data.data.x[node_id].numpy().reshape(1, -1),
#                                        cluster_data.data.x[intra_id].numpy().reshape(1, -1))
#         # similarity是一个1x1的数组，包含了两个向量之间的余弦相似度
#         similarity_value = similarity[0, 0]
#         intra_similarity_list.append(similarity_value)
#
#     for inter_id in inter_id_list:
#         # 计算余弦相似度
#         # cosine_similarity函数期望输入是二维数组，因此需要对向量进行reshape
#         similarity = cosine_similarity(cluster_data.data.x[node_id].numpy().reshape(1, -1),
#                                        cluster_data.mean_feature_list[inter_id].numpy().reshape(1, -1))
#         # similarity是一个1x1的数组，包含了两个向量之间的余弦相似度
#         similarity_value = similarity[0, 0]
#         inter_similarity_list.append(similarity_value)
#
#     intra_id_list_selected = []
#     inter_id_list_selected = []
#     # 使用 numpy 的 argsort 函数，得到数组元素排序后的下标
#     # [-k:] 用于获取最后k个最大元素的下标
#     # [::-1] 用于将这些下标按相似度值从大到小排序
#     for index in np.argsort(intra_similarity_list)[-intra_k:][::-1]:
#         intra_id_list_selected.append(intra_id_list[index])
#
#     for index in np.argsort(inter_similarity_list)[-inter_k:][::-1]:
#         inter_id_list_selected.append(inter_id_list[index])
#
#     # 将 node_id 插入到列表的最前面
#     intra_id_list_selected.insert(0, node_id)
#
#     # create attention bias for intra (positional encoding)
#     # power_adj_list = all_partition_power_adj_list[partition_idx]
#     # # 因为算出来的子图下标都是从0开始
#     # intra_id_list_from_zero = [i - node_start for i in intra_id_list]
#     # attn_bias = torch.cat(
#     #     [torch.tensor(i[intra_id_list_from_zero, :][:, intra_id_list_from_zero].toarray(), dtype=torch.float32).unsqueeze(0) for i in
#     #      power_adj_list])
#     # attn_bias = attn_bias.permute(1, 2, 0)
#
#     return intra_id_list_selected, inter_id_list_selected


def concatenate_ids(node_id, cluster_data: ClusterData, all_partition_power_adj_list):
    # Get the partition index for the given node ID
    partition_idx = get_partition_index(node_id, cluster_data)
    inter_id_list = list(range(len(cluster_data.partition.partptr) - 1))
    inter_id_list.remove(partition_idx)

    # Get the start and end indices of nodes in the specified partition
    node_start = int(cluster_data.partition.partptr[partition_idx])
    node_end = int(cluster_data.partition.partptr[partition_idx + 1])

    # 创建一个列表从 node_start 到 node_end
    intra_id_list = list(range(node_start, node_end))

    # 移除 node_id 如果它在列表中
    intra_id_list.remove(node_id)

    # random dropout
    intra_id_list = random_delete_elements(intra_id_list, 0.2)
    inter_id_list = random_delete_elements(inter_id_list, 0.8)

    # 将 node_id 插入到列表的最前面
    intra_id_list.insert(0, node_id)

    # create attention bias for intra (positional encoding)
    # power_adj_list = all_partition_power_adj_list[partition_idx]
    # # 因为算出来的子图下标都是从0开始
    # intra_id_list_from_zero = [i - node_start for i in intra_id_list]
    # attn_bias = torch.cat(
    #     [torch.tensor(i[intra_id_list_from_zero, :][:, intra_id_list_from_zero].toarray(), dtype=torch.float32).unsqueeze(0) for i in
    #      power_adj_list])
    # attn_bias = attn_bias.permute(1, 2, 0)

    return intra_id_list, inter_id_list


def optimized_concatenate_ids(node_id, cluster_data, all_partition_power_adj_list, intra_k_ratio=0.8, inter_k_ratio=0.2):
    partition_idx = get_partition_index(node_id, cluster_data)
    inter_id_list = list(range(len(cluster_data.partition.partptr) - 1))
    inter_id_list.remove(partition_idx)

    node_start = int(cluster_data.partition.partptr[partition_idx])
    node_end = int(cluster_data.partition.partptr[partition_idx + 1])
    intra_id_list = list(range(node_start, node_end))
    intra_id_list.remove(node_id)

    # 减少不必要的数据转换
    node_id_data = cluster_data.data.x[node_id].numpy()

    def compute_similarity(other_id, is_inter):
        if is_inter:
            other_data = cluster_data.mean_feature_list[other_id].numpy()
        else:
            other_data = cluster_data.data.x[other_id].numpy()
        similarity = cosine_similarity(node_id_data.reshape(1, -1), other_data.reshape(1, -1))
        return similarity[0, 0]

    # 并行计算相似度
    with ThreadPoolExecutor() as executor:
        intra_similarity_list = list(executor.map(lambda x: compute_similarity(x, False), intra_id_list))
        inter_similarity_list = list(executor.map(lambda x: compute_similarity(x, True), inter_id_list))

    intra_k = int(len(intra_id_list) * intra_k_ratio)
    inter_k = int(len(inter_id_list) * inter_k_ratio)

    intra_id_list_selected = [intra_id_list[i] for i in np.argsort(intra_similarity_list)[-intra_k:][::-1]]
    inter_id_list_selected = [inter_id_list[i] for i in np.argsort(inter_similarity_list)[-inter_k:][::-1]]
    intra_id_list_selected.insert(0, node_id)

    return intra_id_list_selected, inter_id_list_selected


def get_partition_index(element_id, cluster_data: ClusterData) -> int:
    # Find the partition index for the given element ID
    for idx in range(len(cluster_data.partition.partptr) - 1):
        node_start = int(cluster_data.partition.partptr[idx])
        node_end = int(cluster_data.partition.partptr[idx + 1])

        if node_start <= element_id < node_end:
            return idx

    # If the element ID does not belong to any partition
    raise ValueError(f"Element ID {element_id} does not belong to any partition.")


def random_delete_elements(input_list, delete_ratio):
    # 确保 delete_ratio 在 0 到 1 之间
    delete_ratio = max(0, min(1, delete_ratio))

    # 计算要删除的元素数量
    num_elements_to_delete = int(len(input_list) * delete_ratio)

    # 随机选择要删除的元素的索引
    indices_to_delete = random.sample(range(1, len(input_list)), num_elements_to_delete)

    # 删除选定的元素
    remaining_list = [input_list[0]] + [value for index, value in enumerate(input_list[1:], start=1) if index not in indices_to_delete]

    return remaining_list


def calculate_partition_mean_feature(partition_idx, cluster_data: ClusterData) -> torch.Tensor:
    # Get the start and end indices of nodes in the specified partition
    node_start = int(cluster_data.partition.partptr[partition_idx])
    node_end = int(cluster_data.partition.partptr[partition_idx + 1])

    # Extract node features for the specified partition
    partition_features = cluster_data.data.x[node_start:node_end]

    # Calculate the mean of node features
    mean_features = torch.mean(partition_features, dim=0)

    return mean_features


def adj_normalize(mx):
    "A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2"
    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def sgc_precompute(features, adj, degree):
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = perf_counter()-t
    return features, precompute_time


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def edge_index_to_adj_t(edge_index, num_nodes):
    # Convert edge_index to undirected representation
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    # Create a sparse tensor representing the adjacency matrix
    adj_t = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]))

    # Coalesce the tensor before extracting indices
    adj_t = adj_t.coalesce()

    # Extract indices from the coalesced tensor
    indices = adj_t.indices()

    return indices


def seed_everything(seed):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random

    Args:
        seed: the integer value seed for global random state.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed