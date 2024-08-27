import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from saliency import calculate_feature_importance
from model.model_backbone import OurConv4

def plot_feature_importance_map(ax, feature_importance, cmap='plasma'):
    """
    绘制特征重要性图。
    """
    ax.imshow(feature_importance.T, cmap=cmap, interpolation='bicubic', aspect='auto')
    ax.set_title('Feature Importance Map', fontsize=20, fontname='Times New Roman', weight='bold')
    ax.set_xlabel('Time Steps', fontsize=15, fontname='Times New Roman')
    ax.set_ylabel('Features', fontsize=15, fontname='Times New Roman')
    ax.tick_params(axis='both', which='major', labelsize=12, labelrotation=0)
    ax.grid(True, linestyle='--', alpha=0.5)

def plot_feature_curve(ax, feature_curve, feature_idx, color):
    """
    绘制特征曲线图，并确保阴影仅在折线下方。
    """
    sns.lineplot(x=range(len(feature_curve)), y=feature_curve, marker='o', color=color, label=f'Feature {feature_idx}', ax=ax)
    ax.fill_between(range(len(feature_curve)), feature_curve, alpha=0.2, color=color, where=(feature_curve >= 0))
    ax.set_title(f'Feature {feature_idx} Curve over Time', fontsize=20, fontname='Times New Roman', weight='bold')
    ax.set_xlabel('Time Steps', fontsize=15, fontname='Times New Roman')
    ax.set_ylabel('Feature Value', fontsize=15, fontname='Times New Roman')
    ax.tick_params(axis='both', which='major', labelsize=12, labelrotation=0)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=15)

def plot_combined_figure(X_train, y_train, model, sample_idx, feature_idx):
    """
    绘制特征重要性图和对应特征的原始曲线图，分别上下分开展示在同一张图片中。
    """
    # 将数据转换为 PyTorch tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    # 抽取样本
    sample = X_train_tensor[sample_idx].unsqueeze(0)
    label = y_train_tensor[sample_idx]

    # 计算特征重要性
    feature_importance = calculate_feature_importance(model, sample, label)

    # 提取整条特征曲线
    feature_curve = X_train_tensor[sample_idx, :, feature_idx].cpu().numpy()

    # 创建图形对象和子图布局
    fig = plt.figure(figsize=(8, 8))
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1])

    # 绘制特征重要性图（无颜色条）
    ax1 = plt.subplot(gs[0])
    plot_feature_importance_map(ax1, feature_importance, cmap='plasma')

    # 绘制特征曲线图
    ax2 = plt.subplot(gs[1])
    plot_feature_curve(ax2, feature_curve, feature_idx, color='darkorange')

    plt.tight_layout()
    plt.show()

# 示例调用
if __name__ == "__main__":
    # 加载数据
    X_train = np.load('./datasets/NATOPS/X_train.npy')
    y_train = np.load('./datasets/NATOPS/y_train.npy')

    # 加载模型并初始化
    model = OurConv4(n_class=6)  # 替换为你自己的模型定义
    model_path = "ckpt/exp-cls/Teacher/NATOPS/jitter_cutout_G0_time_warp/label0.4/2000/backbone_best.tar"  # 替换为实际路径
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 设置样本索引和特征维度索引
    sample_idx = 2
    feature_idx = 10
    # 绘制特征重要性图和特征曲线在同一张图片中
    plot_combined_figure(X_train, y_train, model, sample_idx, feature_idx)