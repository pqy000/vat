import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from utils.saliency import calculate_feature_importance
from model.model_backbone import OurConv4

def plot_feature_importance_map(feature_importance, cmap='viridis'):
    """
    绘制特征重要性图，并放大字体和调整美化细节。
    """
    plt.imshow(feature_importance.T, cmap=cmap, interpolation='bicubic', aspect='auto')
    plt.colorbar(label='Importance', pad=0.01, fraction=0.046)
    plt.xlabel('Time Steps', fontsize=18, fontname='Times New Roman')
    plt.ylabel('Features', fontsize=18, fontname='Times New Roman')
    plt.xticks(fontsize=15, fontname='Times New Roman')
    plt.yticks(fontsize=15, fontname='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title('Feature Importance Map', fontsize=22, fontname='Times New Roman', weight='bold')

def plot_feature_curve(ax, feature_curve, feature_idx, color):
    """
    在指定的轴上绘制特征曲线图，并放大字体和调整美化细节。
    """
    ax.plot(range(len(feature_curve)), feature_curve, marker='o', color=color, label=f'Feature {feature_idx}', markersize=5)
    ax.fill_between(range(len(feature_curve)), feature_curve, alpha=0.2, color=color)
    ax.set_xlabel('Time Steps', fontsize=18, fontname='Times New Roman')
    ax.set_ylabel('Feature Value', fontsize=18, fontname='Times New Roman')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=15)

def plot_combined_figure(X_train, y_train, model, sample_idx, feature_idx_1, feature_idx_2, cmap='viridis'):
    """
    绘制特征重要性图和两个不同特征索引的曲线图，分别展示在三个子图中，并美化整体效果。
    """
    # 将数据转换为 PyTorch tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    # 抽取样本
    sample = X_train_tensor[sample_idx].unsqueeze(0)
    label = y_train_tensor[sample_idx]

    # 计算特征重要性
    feature_importance = calculate_feature_importance(model, sample, label)

    # 提取两个不同特征的曲线
    feature_curve_1 = X_train_tensor[sample_idx, :, feature_idx_1].cpu().numpy()
    feature_curve_2 = X_train_tensor[sample_idx, :, feature_idx_2].cpu().numpy()

    # 创建图形对象，并指定子图排列方式
    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])

    # 上方的特征重要性图
    ax1 = plt.subplot(gs[0])
    plot_feature_importance_map(feature_importance, cmap=cmap)

    # 中间的第一个特征曲线图
    ax2 = plt.subplot(gs[1])
    plot_feature_curve(ax2, feature_curve_1, feature_idx_1, color='darkorange')

    # 下方的第二个特征曲线图
    ax3 = plt.subplot(gs[2])
    plot_feature_curve(ax3, feature_curve_2, feature_idx_2, color='dodgerblue')

    plt.tight_layout()
    plt.show()

# 示例调用
if __name__ == "__main__":
    # 加载数据
    X_train = np.load('./datasets/Heartbeat/X_train.npy')
    y_train = np.load('./datasets/Heartbeat/y_train.npy')

    model = OurConv4()
    model_path = "ckpt/exp-cls/Teacher/Heartbeat/Heartbeat/jitter_cutout_G0_time_warp/label0.4/2000/"
    combined_path = os.path.join(model_path, 'backbone_best.tar')
    model.load_state_dict(torch.load(combined_path))
    model.eval()

    # 设置样本索引和两个特征维度索引
    sample_idx = 3  # 要查看的样本索引
    feature_idx_1 = 49  # 第一个特征维度
    feature_idx_2 = 10  # 第二个特征维度

    # 绘制特征重要性图和两个特征曲线在同一张图片中
    plot_combined_figure(X_train, y_train, model, sample_idx, feature_idx_1, feature_idx_2)