import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# 加载预训练的模型（以ResNet为例）
model = models.resnet50(pretrained=True)
model.eval()  # 设置模型为评估模式

# 定义图像的预处理步骤
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载和预处理图像
img = Image.open("your_image.jpg")  # 替换为你的图像路径
img_tensor = preprocess(img).unsqueeze(0)  # 添加一个批次维度

# 确保图像有梯度信息
img_tensor.requires_grad_()

# 前向传播
output = model(img_tensor)

# 选择一个目标类别（例如，分类最高的类别）
target_class = output.argmax(dim=1).item()

# 计算目标类别相对于输入图像的梯度
model.zero_grad()  # 清除先前的梯度
output[0, target_class].backward()

# 获取梯度并计算Saliency Map
saliency_map = img_tensor.grad.data.abs().squeeze().max(dim=0)[0]

# 将Saliency Map转换为NumPy数组以便于可视化
saliency_map = saliency_map.cpu().numpy()

# 显示Saliency Map
plt.imshow(saliency_map, cmap='hot')
plt.axis('off')
plt.show()