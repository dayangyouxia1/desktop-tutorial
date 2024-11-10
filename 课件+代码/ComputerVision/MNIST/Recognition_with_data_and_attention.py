import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import time
import cv2

start_time=time.time()

BATCH_SIZE = 32 # 批处理，每次处理的数据
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #使用CPU还是GPU
EPOCHS = 15  # 轮次，一个数据集循环运行几次

from torchvision import transforms

pipeline = transforms.Compose([
    transforms.ToPILImage(),  # Convert to PIL image
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.RandomAffine(
        degrees=10,  # Rotation up to ±10 degrees
        translate=(0.1, 0.1),  # Allow up to 10% translation
        scale=(0.9, 1.1),  # Allow slight zoom-in and zoom-out
        shear=(-5, 5)  # Allow shearing of up to ±5 degrees
    ),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),  # Apply perspective transformation with 50% probability
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Add random changes to brightness and contrast
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize using dataset mean and std
])


test_pipeline = transforms.Compose([
    transforms.ToPILImage(),  # 转换为 PIL 图像
    transforms.Resize((28, 28)),  # 调整大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化
])

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
        return image, label


# 自注意力层定义


class SelfAttention(nn.Module):
    def __init__(self, in_dim, num_heads=4, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm([in_dim, 28, 28])  # Assumes input size of 28x28; adjust if needed.

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B * N * C
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  # B * C * N
        energy = torch.bmm(proj_query, proj_key)  # B * N * N
        attention = self.dropout(torch.softmax(energy / (channels ** 0.5), dim=-1))  # Dropout and scaling

        proj_value = self.value_conv(x).view(batch_size, -1, width * height)  # B * C * N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(batch_size, channels, width, height)
        out = self.gamma * out + x  # Residual connection
        out = self.norm(out)  # Layer normalization

        return out



def weight_variable(shape):
    return nn.init.trunc_normal_(torch.empty(shape), std=0.1)

def bias_variable(shape):
    return torch.full(shape, 0.1)

class Digit(nn.Module):
    def __init__(self):
        super(Digit, self).__init__()

        # Convolutional Layers with Batch Normalization
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # 5x5 kernel
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)  # 5x5 kernel
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 3x3 kernel for deeper features
        self.bn3 = nn.BatchNorm2d(128)
        # 添加自注意力层
        self.selfattn1 = SelfAttention(32)
        self.selfattn2 = SelfAttention(256)
        

        # Fully Connected Layers
        self.fc1 = nn.Linear(3 * 3 * 128, 512)  # Adjusted size for the new structure
        self.bn4 = nn.BatchNorm2d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10)  # Output layer for 10 classes

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Initialize weights and biases
        self.init_weights()

    def init_weights(self):
        # Initialize weights with truncated normal and biases with constant 0.1
        weight_variable(self.conv1.weight.shape).copy_(self.conv1.weight)
        bias_variable(self.conv1.bias.shape).copy_(self.conv1.bias)
        weight_variable(self.conv2.weight.shape).copy_(self.conv2.weight)
        bias_variable(self.conv2.bias.shape).copy_(self.conv2.bias)
        weight_variable(self.conv3.weight.shape).copy_(self.conv3.weight)
        bias_variable(self.conv3.bias.shape).copy_(self.conv3.bias)
        weight_variable(self.fc1.weight.shape).copy_(self.fc1.weight)
        bias_variable(self.fc1.bias.shape).copy_(self.fc1.bias)
        weight_variable(self.fc2.weight.shape).copy_(self.fc2.weight)
        bias_variable(self.fc2.bias.shape).copy_(self.fc2.bias)
        weight_variable(self.fc3.weight.shape).copy_(self.fc3.weight)
        bias_variable(self.fc3.bias.shape).copy_(self.fc3.bias)

    def forward(self, x):
        # First convolutional layer with batch normalization, relu, and max pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)  # Pooling to 14x14

        # Second convolutional layer with batch normalization, relu, and max pooling
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)  # Pooling to 7x7

        # Third convolutional layer with batch normalization, relu, and max pooling
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2, 2)  # Pooling to 3x3

        # Flatten for fully connected layer
        x = x.view(-1, 3 * 3 * 128)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Output layer with softmax
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)  # Log softmax for classification


# 7 定义训练方法

def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        # 部署到device上
        data, target = data.to(device), target.to(device)
        # 梯度初始化为0
        optimizer.zero_grad()
        # 训练后的结果
        output = model(data)
        # 计算损失
        loss = F.nll_loss(output, target)
        # 反向传播
        loss.backward()
        # 参数优化
        optimizer.step()
        if batch_index % 3000 == 0:
            print("Train Epoch : {} \t Loss : {:.6f}".format(epoch, loss.item()))

# 8 定义测试方法

def ceshi(model, device, test_loader):
    # 模型验证
    model.eval()
    # 正确率
    correct = 0.0
    # 测试损失
    test_loss = 0.0
    # 不会进行计算梯度，也不会进行反向传播
    with torch.no_grad():
        for data, target in test_loader:
            # 部署到device上
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算损失
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # 找到概率值最大的下标
            pred = output.max(1, keepdim=True)[1]
            #pred = torch.max(output, dim=1)
            # pred = output.argmax(dim=1)
            # 累计正确的值
            correct += pred.eq(target.view_as(pred)).sum().item()
        # 计算平均loss
        test_loss /= len(test_loader.dataset)
        print("Test -- Average Loss : {:.4f}, Accuracy : {:.3f}\n".format(test_loss, correct/len(test_loader.dataset)*100.0))


if __name__ == '__main__':

    # 和Recognition的区别是，这里读取本地数据集
    # 创建自定义数据集实例
    train_set = CustomDataset("train_data", "train_data/label.csv", transform=pipeline)
    test_set = CustomDataset("test_data", "test_data/label.csv", transform=test_pipeline)

    # 下载数据集
    # train_set = datasets.MNIST("data", train=True, download=True, transform=pipeline)
    # test_set = datasets.MNIST("data", train=False, download=True, transform=pipeline)

    # 加载数据
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
    """
    # 显示图片方法一
    import matplotlib.pyplot as plt

    # 从训练加载器中获取一个批次的数据
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # 选择一个图像和标签进行展示
    image = images[0]
    label = labels[0]

    # 图像需要做一下转换，因为它当前是归一化和标准化过的
    image = image.squeeze()  # 去掉批次维度
    image = image * 0.3081 + 0.1307  # 反标准化

    # 绘图显示图像
    plt.imshow(image.numpy(), cmap='gray')
    plt.title(f'Label: {label}')
    plt.show()
    """
    """
    # 显示MNIST中的图片
    with open("./data/MNIST/raw/t10k-images-idx3-ubyte", "rb") as f:
        file = f.read()

    # image1 = [int(str(item).encode('ascii'), 16) for item in file[16:16+784]] # 提取图片
    image1 = [int(item) for item in file[16:16 + 784]]  # 提取图片
    print(image1)

    import cv2
    import numpy as np

    image1_np0 = np.array(image1).reshape(28, 28)
    print(image1_np0)
    image1_np = np.array(image1, dtype=np.uint8).reshape(28, 28, 1)
    print(image1_np.shape)
    cv2.imwrite("digit.jpg", image1_np)
    """
    model = Digit().to(DEVICE)
    # 6 定义优化器
    optimizer = optim.Adam(model.parameters())

    # 9 调用方法(7和8)

    for epoch in range(1, EPOCHS+1):
        train_model(model, DEVICE, train_loader, optimizer, epoch)
        ceshi(model, DEVICE, test_loader)

    torch.save(model.state_dict(), "mnist_cnn_with_data_and_attention.pt")
    print(f"run_time: {time.time()-start_time}")