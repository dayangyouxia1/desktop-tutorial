import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
import csv

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义图像转换
pipeline = transforms.Compose([
    transforms.ToPILImage(),  # 转换为 PIL 图像
    transforms.Resize((28, 28)),  # 将图像大小调整为 28x28 像素
    transforms.ToTensor(),  # 将图像转换为 PyTorch 张量，并将像素值缩放到 [0, 1]
    transforms.Normalize((0.1307,), (0.3081,))  # 进行归一化处理，标准化像素值
])

from Recognition_with_data_and_attention import Digit

model = Digit().to(DEVICE)
model.load_state_dict(torch.load("mnist_cnn_with_data_and_attention.pt"))
model.eval()

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_folder, label_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        # 读取标签文件
        labels_df = pd.read_csv(label_file)
        self.labels = dict(zip(labels_df['id'].astype(str), labels_df['label']))

        # 获取所有图片 ID
        self.image_ids = list(self.labels.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_name = f"{self.image_folder}/{img_id}.png"
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)  # 读取灰度图像
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image)
            image = image.unsqueeze(0)  # 添加一个维度，变成 (1, height, width)
            image = image.float() / 255.0  # 归一化到 [0, 1]

        label = self.labels[img_id]
        return image, label, img_id

BATCH_SIZE = 16  # 批处理，每次处理的数据
# 创建自定义数据集实例
custom_dataset = CustomDataset("test_data", "/workspaces/desktop-tutorial/课件+代码/ComputerVision/MNIST/test_data/label.csv", transform=pipeline)
custom_loader = DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 保存预测结果到 CSV 文件
with open("result.csv", "w", newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["image_id", "predicted_digit"])  # 写入表头

    # 遍历数据加载器
    for images, labels, img_ids in custom_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1, keepdim=True).cpu().numpy().flatten()  # 获取预测结果

        for img_id, pred in zip(img_ids, preds):
            csv_writer.writerow([img_id, pred])  # 写入每一行的结果

print("预测结果已保存到 result.csv 文件中。")
