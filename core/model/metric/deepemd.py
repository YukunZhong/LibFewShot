import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


import yaml

from core.utils import accuracy, accuracy_DeepEMD
from .metric_model import MetricModel
from core.model.backbone.resnet12_emd import custom_resnet12

# 加载预训练模型的路径
DATA_DIR = './core/data/miniImageNet--ravi'
PRETRAIN_DIR = './results/DeepEMD_Pretrain-miniImageNet--ravi-resnet12emd-5-1-Jul-06-2024-08-14-06/checkpoints/model_best.pth'

class DistanceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(DistanceCrossEntropyLoss, self).__init__()

    def forward(self, distances, targets):
        log_probabilities = F.log_softmax(-distances, dim=1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        if targets.dtype != torch.long:
            targets = targets.long()
        target_log_probs = log_probabilities.gather(1, targets).squeeze(1)
        loss = -target_log_probs.mean()
        return loss


def compute_cost_matrix(U, V):
    U_flat = U.view(U.size(0), U.size(1) * U.size(2), -1)  # [batch_size, num_prototypes, feature_dim]
    V_flat = V.view(V.size(0), V.size(1), -1)  # [batch_size, num_prototypes, feature_dim]

    U_norm = F.normalize(U_flat, p=2, dim=2)
    V_norm = F.normalize(V_flat, p=2, dim=2)

    cost_matrix = 1 - torch.matmul(U_norm, V_norm.transpose(1, 2))

    return cost_matrix


class StructuredFullyConnectedLayer(nn.Module):
    def __init__(self, input_size, num_classes, num_prototypes):
        super(StructuredFullyConnectedLayer, self).__init__()
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.input_size = input_size
        self.prototypes = nn.Parameter(torch.randn(num_classes, num_prototypes, input_size))

    def compute_weights(self, x, prototypes):
        if prototypes.dim() == 2:
            prototypes = prototypes.unsqueeze(0)  # Add a batch dimension if missing
        batch_size, num_prototypes, input_size = prototypes.size()

        x_mean = torch.mean(x, dim=0, keepdim=True)  # Now x_mean should be [1, 640]
        x_mean_flat = x_mean.view(1, -1)  # Flatten to [1, feature_dim]

        weights = []
        for i in range(batch_size):
            proto_flat = prototypes[i].view(num_prototypes, -1)  # Flatten prototypes to [num_prototypes, input_size]
            dot_product = torch.matmul(proto_flat, x_mean_flat.transpose(0, 1)).squeeze()  # should result in [num_prototypes]
            weight = torch.relu(dot_product)
            weights.append(weight)
        weights = torch.stack(weights)  # Stack to get [batch_size, num_prototypes]
        weights /= (torch.sum(weights, dim=1, keepdim=True) + 1e-8)  # Add a small epsilon if division by zero is possible

        return weights

    def forward(self, x):
        results = []
        for i in range(self.num_classes):
            proto = self.prototypes[i]
            if proto.dim() == 2:
                proto = proto.unsqueeze(0).repeat(self.num_prototypes, 1, 1)  # 在最前面增加一个维度，尺寸为 num_prototypes
            weights = self.compute_weights(x, proto)
            results.append((x * weights.unsqueeze(2), proto * weights.unsqueeze(2)))
        return results


class DeepEMD(MetricModel):
    def __init__(self, args, resnet12emd, **kwargs):
        super(DeepEMD, self).__init__(init_type='normal', **kwargs)
        self.input_size = args.input_size
        self.num_classes = args.num_classes
        self.num_prototypes = args.num_prototypes
        self.use_sfc = args.use_sfc
        self.emd_iterations = args.emd_iterations

        self.fc_layer = StructuredFullyConnectedLayer(self.input_size, self.num_classes, self.num_prototypes)
        self.loss_func = DistanceCrossEntropyLoss()
        self.encoder = custom_resnet12()  # 初始化自定义的ResNet12_emd模型

        if args.pretrain == 'origin':
            load_model(self.encoder, args.pretrain_dir)

    def normalize_feature(self, x):
        if self.args.norm == 'center':
            x = x - x.mean(1).unsqueeze(1)
        return x

    def build_feature_pyramid(self, feature):
        feature_list = []
        feature_pyramid = [int(size) for size in self.args.feature_pyramid.split(',')]
        for size in feature_pyramid:
            feature_list.append(F.adaptive_avg_pool2d(feature, size).view(
                feature.shape[0], feature.shape[1], 1, -1))
        feature_list.append(feature.view(
            feature.shape[0], feature.shape[1], 1, -1))
        out = torch.cat(feature_list, dim=-1)
        return out

    def get_similarity_map(self, proto, query):
        way = proto.shape[0]
        num_query = query.shape[0]
        query = query.view(query.shape[0], query.shape[1], -1)
        proto = proto.view(proto.shape[0], proto.shape[1], -1)

        proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
        query = query.unsqueeze(1).repeat([1, way, 1, 1])
        proto = proto.permute(0, 1, 3, 2)
        query = query.permute(0, 1, 3, 2)
        feature_size = proto.shape[-2]

        if self.args.metric == 'cosine':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = F.cosine_similarity(proto, query, dim=-1)
        if self.args.metric == 'l2':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = (proto - query).pow(2).sum(-1)
            similarity_map = 1 - similarity_map

        return similarity_map

    def set_forward(self, batch):
        support_images, query_images, support_labels, query_labels = batch
        support_images = support_images.to(self.device)
        query_images = query_images.to(self.device)

        support_feat = self.encoder(support_images)
        support_feat = self.normalize_feature(support_feat)
        if self.args.feature_pyramid:
            support_feat = self.build_feature_pyramid(support_feat)
        support_results = self.fc_layer(support_feat)

        query_feat = self.encoder(query_images)
        query_feat = self.normalize_feature(query_feat)
        if self.args.feature_pyramid:
            query_feat = self.build_feature_pyramid(query_feat)
        query_results = self.fc_layer(query_feat)

        emd_distances = []
        for x_weighted, proto_weighted in support_results:
            emd_distance, _ = self.compute_emd(x_weighted, proto_weighted)
            emd_distances.append(emd_distance)
        emd_distances = torch.stack(emd_distances, dim=1)

        output = F.softmax(-emd_distances, dim=-1)
        acc = accuracy_DeepEMD(emd_distances, query_labels.view(-1))
        loss = self.loss_func(emd_distances, query_labels)

        return acc, emd_distances, output, query_labels, loss

    def compute_emd(self, x_weighted, proto_weighted):
        batch_size = x_weighted.size(0)
        num_prototypes = proto_weighted.size(1)
        cost_matrix = compute_cost_matrix(x_weighted.unsqueeze(1), proto_weighted).cpu().detach().numpy()
        emd_distances = []
        optimal_flows = []
        for i in range(batch_size):
            row_ind, col_ind = linear_sum_assignment(cost_matrix[i])
            emd_distance = cost_matrix[i, row_ind, col_ind].sum() / num_prototypes
            cost_matrix_tensor = torch.tensor(cost_matrix[i], device=x_weighted.device)
            flow_matrix = torch.zeros_like(cost_matrix_tensor)
            flow_matrix[row_ind, col_ind] = 1
            emd_distances.append(emd_distance)
            optimal_flows.append(flow_matrix)
        emd_distances = torch.tensor(emd_distances, dtype=torch.float32).to(x_weighted.device)
        optimal_flows = torch.stack([torch.tensor(f, dtype=torch.float32).to(x_weighted.device) for f in optimal_flows])
        return emd_distances, optimal_flows

    def set_forward_loss(self, batch):
        support_images, query_images, support_labels, query_labels = batch
        support_images = support_images.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)

        support_feat = self.encoder(support_images)
        support_feat = self.normalize_feature(support_feat)
        if self.args.feature_pyramid:
            support_feat = self.build_feature_pyramid(support_feat)
        support_results = self.fc_layer(support_feat)

        query_feat = self.encoder(query_images)
        query_feat = self.normalize_feature(query_feat)
        if self.args.feature_pyramid:
            query_feat = self.build_feature_pyramid(query_feat)
        query_results = self.fc_layer(query_feat)

        emd_distances = []
        for x_weighted, proto_weighted in zip(support_results, query_results):
            emd_distance, _ = self.compute_emd(x_weighted, proto_weighted)
            emd_distances.append(emd_distance)
        emd_distances = torch.stack(emd_distances, dim=1)

        output = F.softmax(-emd_distances, dim=-1)
        loss = self.loss_func(emd_distances, query_labels)
        acc = accuracy_DeepEMD(output, query_labels)

        # print(f"emd_distances shape: {emd_distances.shape}")
        # print(f"query_labels shape: {query_labels.shape}")
        # print(f"Accuracy: {acc}, Loss: {loss}")
        # print(f"loss.requires_grad: {loss.requires_grad}")

        return output, acc, loss

    def forward(self, batch):
        return self.set_forward_loss(batch)

    __call__ = forward


def load_model(model, dir):
    model_dict = model.state_dict()
    print('loading model from :', dir)
    pretrained_dict = torch.load(dir)['params']
    print('pretrained_dict.keys():', pretrained_dict.keys())

    if 'encoder' in list(pretrained_dict.keys())[0]:
        if 'module' in list(pretrained_dict.keys())[0]:
            pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
        else:
            pretrained_dict = {k: v for k, v in pretrained_dict.items()}
    else:
        pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print('load model success')
    return model


if __name__ == "__main__":
    # 读取配置文件
    with open('./config/deepemd.yaml', 'r') as f:
        config = yaml.safe_load(f)

    resnet12emd = custom_resnet12()  # Initialize  resnet12_emd model here
    model = DeepEMD(config['classifier']['kwargs']['args'], resnet12emd)
