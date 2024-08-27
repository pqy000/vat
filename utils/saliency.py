import matplotlib.pyplot as plt
import torch
import seaborn as sns

def calculate_feature_importance(model, input, target):
    model.eval()
    input.requires_grad = True
    output = model(input)
    target_output = output[0, target]
    target_output.backward()
    grad = input.grad.data
    grad_abs = torch.abs(grad)
    # grad_importance = grad_abs / grad_abs.max(dim=2, keepdim=True)[0]
    grad_importance = grad_abs / grad_abs.max()
    return grad_importance.squeeze()

def visualize_feature_importance(feature_importance, cmap='viridis'):
    feature_importance = feature_importance.cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.imshow(feature_importance.T, cmap=cmap, interpolation='bilinear', aspect='auto')
    plt.colorbar()
    plt.title('Feature Importance', fontsize=15, fontname='Times New Roman')
    plt.xlabel('Time Steps', fontsize=15, fontname='Times New Roman')
    plt.ylabel('Features', fontsize=15, fontname='Times New Roman')
    plt.xticks(fontsize=15, fontname='Times New Roman')
    plt.yticks(fontsize=15, fontname='Times New Roman')
    plt.show()


def plot_feature_importance_map(feature_importance, cmap='coolwarm'):
    """
    绘制特征重要性图。
    """
    plt.figure(figsize=(14, 6))
    plt.imshow(feature_importance.T, cmap=cmap, interpolation='bicubic', aspect='auto')
    plt.colorbar(label='Importance')
    plt.title('Feature Importance Map', fontsize=20, fontname='Times New Roman', weight='bold')
    plt.xlabel('Time Steps', fontsize=15, fontname='Times New Roman')
    plt.ylabel('Features', fontsize=15, fontname='Times New Roman')
    plt.xticks(fontsize=12, fontname='Times New Roman')
    plt.yticks(fontsize=12, fontname='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_feature_curve(feature_curve, feature_idx):
    """
    绘制特征曲线图。
    """
    plt.figure(figsize=(14, 6))
    sns.lineplot(x=range(len(feature_curve)), y=feature_curve, marker='o', color='orange', label=f'Feature {feature_idx}')
    plt.fill_between(range(len(feature_curve)), feature_curve, alpha=0.4, color='darkorange')
    plt.title(f'Feature {feature_idx} Curve over Time', fontsize=20, fontname='Times New Roman', weight='bold')
    plt.xlabel('Time Steps', fontsize=15, fontname='Times New Roman')
    plt.ylabel('Value', fontsize=15, fontname='Times New Roman')
    plt.xticks(fontsize=12, fontname='Times New Roman')
    plt.yticks(fontsize=12, fontname='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=15)
    plt.show()

# def plot_feature_importance_map(feature_importance, cmap='coolwarm'):
#     """
#     绘制特征重要性图，并在图内部添加标题。
#     """
#     plt.figure(figsize=(14, 6))
#     plt.imshow(feature_importance.T, cmap=cmap, interpolation='bicubic', aspect='auto')
#     plt.colorbar(label='Importance')
#     plt.title('Feature Importance Map', fontsize=20, fontname='Times New Roman', weight='bold', pad=20)
#     plt.xlabel('Time Steps', fontsize=15, fontname='Times New Roman')
#     plt.ylabel('Features', fontsize=15, fontname='Times New Roman')
#     plt.xticks(fontsize=12, fontname='Times New Roman')
#     plt.yticks(fontsize=12, fontname='Times New Roman')
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.text(0.95, 0.05, 'Feature Importance Map', fontsize=20, fontname='Times New Roman', weight='bold', color='white',
#              ha='right', va='bottom', transform=plt.gca().transAxes, bbox=dict(facecolor='black', alpha=0.5))
#     plt.show()
#
# def plot_feature_curve(feature_curve, feature_idx):
#     """
#     绘制特征曲线图，并在图内部添加标题。
#     """
#     plt.figure(figsize=(14, 6))
#     sns.lineplot(x=range(len(feature_curve)), y=feature_curve, marker='o', color='darkorange', label=f'Feature {feature_idx}')
#     plt.fill_between(range(len(feature_curve)), feature_curve, alpha=0.2, color='darkorange')
#     plt.title(f'Feature {feature_idx} Curve over Time', fontsize=20, fontname='Times New Roman', weight='bold', pad=20)
#     plt.xlabel('Time Steps', fontsize=15, fontname='Times New Roman')
#     plt.ylabel('Feature Value', fontsize=15, fontname='Times New Roman')
#     plt.xticks(fontsize=12, fontname='Times New Roman')
#     plt.yticks(fontsize=12, fontname='Times New Roman')
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.text(0.95, 0.05, f'Feature {feature_idx} Curve', fontsize=20, fontname='Times New Roman', weight='bold',
#              color='darkorange', ha='right', va='bottom', transform=plt.gca().transAxes, bbox=dict(facecolor='black', alpha=0.5))
#     plt.legend(fontsize=15)
#     plt.show()

def plot_feature_importance_map(feature_importance, cmap='viridis'):
    """
    绘制特征重要性图。
    """
    plt.imshow(feature_importance.T, cmap=cmap, interpolation='bicubic', aspect='auto')
    plt.colorbar(label='Importance')
    plt.xlabel('Time Steps', fontsize=15, fontname='Times New Roman')
    plt.ylabel('Features', fontsize=15, fontname='Times New Roman')
    plt.xticks(fontsize=12, fontname='Times New Roman')
    plt.yticks(fontsize=12, fontname='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title('Feature Importance Map', fontsize=20, fontname='Times New Roman', weight='bold')
