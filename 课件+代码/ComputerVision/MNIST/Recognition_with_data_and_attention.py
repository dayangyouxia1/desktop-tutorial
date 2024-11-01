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

BATCH_SIZE = 16 # 批处理，每次处理的数据
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #使用CPU还是GPU
EPOCHS = 10  # 轮次，一个数据集循环运行几次

pipeline = transforms.Compose([
    transforms.ToPILImage(),  # 转换为 PIL 图像
    transforms.Resize((28, 28)),  # 调整大小
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移
    transforms.RandomRotation((-10,10)), # 随机旋转 [-10, 10] 度
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化
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
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width*height).permute(0, 2, 1)  # B * (N) * C
        proj_key = self.key_conv(x).view(batch_size, -1, width*height)  # B * C * (N)
        energy = torch.bmm(proj_query, proj_key)  # batch的对应矩阵
        attention = torch.softmax(energy, dim=-1)  # batch
        proj_value = self.value_conv(x).view(batch_size, -1, width*height)  # batch*（N）*channels
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(batch_size, channels, width, height)
        out = self.gamma*out + x
        return out


# 5 构建网络模型
class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)  # 1: 灰度图片的通道，10：输出通道，5：kernel
        self.conv2 = nn.Conv2d(10, 20, 3)  # 10: 输入通道，20: 输出通道，3: kernel
        self.selfattn1 = SelfAttention(10)  # 添加自注意力层
        self.selfattn2 = SelfAttention(20)  # 添加自注意力层
        self.fc1 = nn.Linear(20 * 10 * 10, 500)  # 20*10*10: 输入通道， 500: 输出通道
        self.fc2 = nn.Linear(500, 10)  # 500: 输入通道，10:输出通道

    def forward(self, x):
        input_size = x.size(0)  # batch_size
        x = self.conv1(x)  # 输入: batch*1*28*28, 输出: batch*10*24*24 （28-5+1）
        x = self.selfattn1(x)  # 自注意力层
        x = F.relu(x)  # 保持shape不变，输出：batch*10*24*24
        x = F.max_pool2d(x, 2, 2)  # 输入：batch*10*24*24, 输出: batch*10*12*12

        x = self.conv2(x)  # 输入: batch*10*12*12, 输出: batch*20*10*10
        x = self.selfattn2(x)  # 自注意力层
        x = F.relu(x)  # 保持shape不变，输出：batch*20*10*10

        x = x.view(input_size, -1)  # 拉平，-1 自动计算维度， 20*10*10=2000

        x = self.fc1(x)  # 输入: batch*2000, 输出: batch*500
        x = F.relu(x)  # 保持shape不变，输出: batch*500

        x = self.fc2(x)  # 输入: batch*500, 输出: 10
        output = F.log_softmax(x, dim=1)  # 计算分类后，每个数字的概率值

        return output


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