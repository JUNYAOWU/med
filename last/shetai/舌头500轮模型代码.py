import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import pandas as pd
import time
from tqdm import tqdm  # 引入 tqdm 库用于显示进度条

# 定义图片路径
data_dir = r'E:\Bigdata\tongue'

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # 数据集根目录
        self.transform = transform  # 数据增强和预处理变换
        self.classes = [class_name for class_name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, class_name))]  # 获取所有类别名称
        self.images = []  # 存储图像路径
        self.labels = []  # 存储对应的标签

        # 遍历每个类别并收集图像路径和标签
        for label, class_name in enumerate(self.classes):
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                # 跳过非文件或隐藏文件（如 .DS_Store）
                if not os.path.isfile(img_path) or img_name.startswith('.'):
                    continue
                self.images.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)  # 返回数据集的大小

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')  # 打开图像并转换为 RGB 格式

        if self.transform:
            image = self.transform(image)  # 应用数据增强和预处理

        label = torch.tensor(self.labels[idx], dtype=torch.long)  # 将标签转换为张量
        return image, label  # 返回图像和对应的标签

# 数据增强和预处理
transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),  # 随机裁剪调整为指定大小
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # 调整亮度、对比度、饱和度和色调
    transforms.RandomRotation(degrees=(5, 15)),  # 随机旋转 5 到 15 度
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),  # 随机仿射变换
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),  # 随机透视变换
    transforms.RandomGrayscale(p=0.05),  # 以一定概率将图像转为灰度
    transforms.ToTensor(),  # 转换为 PyTorch 张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化处理
])

if __name__ == '__main__':
    # 加载数据集
    dataset = CustomDataset(data_dir, transform=transform)

    # 使用 pandas 处理数据集统计信息
    label_counts = pd.Series(dataset.labels).value_counts()  # 统计每个类别的样本数量
    print("Label distribution:")
    print(label_counts)

    # 动态计算数据集中的类别数
    num_classes = len(set(dataset.labels))

    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)  # 增加 num_workers，提高数据加载效率，减小 batch_size

    # 构建改进后的CNN模型，添加更多的卷积层和残差连接
    class CNNModel(nn.Module):
        def __init__(self, num_classes):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 第一层卷积，输入通道为 3，输出通道为 64
            self.bn1 = nn.BatchNorm2d(64)  # 第一层批归一化
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 第二层卷积，输出通道为 128
            self.bn2 = nn.BatchNorm2d(128)  # 第二层批归一化
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 第三层卷积，输出通道为 256
            self.bn3 = nn.BatchNorm2d(256)  # 第三层批归一化
            self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # 添加额外的卷积层，输出通道为 512
            self.bn4 = nn.BatchNorm2d(512)  # 第四层批归一化

            self.pool = nn.MaxPool2d(2, 2)  # 最大池化层，窗口大小为 2x2
            self.fc1 = nn.Linear(512 * 14 * 14, 1024)  # 全连接层，输入特征数量为 512 * 14 * 14，输出为 1024
            self.fc2 = nn.Linear(1024, 256)  # 第二个全连接层，输出为 256
            self.fc3 = nn.Linear(256, num_classes)  # 最后的全连接层，输出类别数
            self.dropout = nn.Dropout(0.3)  # Dropout 层，防止过拟合

        def forward(self, x):
            x = self.pool(torch.Mash(self.bn1(self.conv1(x))))  # 卷积 -> 批归一化 -> ReLU 激活 -> 池化
            x = self.pool(torch.relu(self.bn2(self.conv2(x))))  # 第二层卷积块
            x = self.pool(torch.relu(self.bn3(self.conv3(x))))  # 第三层卷积块
            x = self.pool(torch.relu(self.bn4(self.conv4(x))))  # 第四层卷积块
            x = x.view(-1, 512 * 14 * 14)  # 展平张量，为全连接层做准备
            x = torch.relu(self.fc1(x))  # 第一层全连接层，ReLU 激活
            x = self.dropout(x)  # Dropout 防止过拟合
            x = torch.relu(self.fc2(x))  # 第二层全连接层，ReLU 激活
            x = self.fc3(x)  # 最后一层全连接层，输出分类结果
            return x

    # 实例化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 动态选择使用 GPU 或 CPU
    model = CNNModel(num_classes=num_classes).to(device)  # 将模型移动到设备上

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 使用交叉熵损失函数并添加标签平滑以提高泛化能力
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # 使用 AdamW 优化器，并添加 L2 正则化（权重衰减）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 学习率调度器，每 10 个 epoch 将学习率乘以 0.1

    # 训练模型
    scaler = torch.cuda.amp.GradScaler()  # 初始化梯度缩放器，以支持混合精度训练
    num_epochs = 500  # 将训练轮数设置为500
    total_start_time = time.time()  # 记录训练开始时间
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0  # 初始化累计损失
        start_time = time.time()  # 记录当前 epoch 的开始时间
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):  # 使用 tqdm 显示训练进度
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)  # 将数据移动到设备上

            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):  # 使用混合精度加速训练
                outputs = model(images)  # 前向传播
                loss = criterion(outputs, labels)  # 计算损失

            # 反向传播和优化
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播，计算梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，以防止梯度爆炸
            optimizer.step()  # 更新参数

            running_loss += loss.item()  # 累计损失

        scheduler.step()  # 更新学习率
        epoch_time = time.time() - start_time  # 计算当前 epoch 的耗时
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}, Time: {epoch_time:.2f} seconds")

    total_time = time.time() - total_start_time  # 计算训练的总时间
    print(f"Total training time: {total_time:.2f} seconds")

    # 保存模型
    torch.save(model.state_dict(), 'my_pytorch_model.pth')  # 保存模型参数

    # 加载已保存的模型
    model = CNNModel(num_classes=num_classes)  # 实例化模型
    model.load_state_dict(torch.load('my_pytorch_model.pth', map_location=device))  # 加载保存的模型参数
    model.to(device)  # 将模型移动到设备上

    # 评估模型
    model.eval()  # 设置模型为评估模式
    correct = 0  # 初始化预测正确的样本数量
    total = 0  # 初始化总样本数量
    with torch.no_grad():  # 在评估时不计算梯度
        for images, labels in tqdm(train_loader, desc="Evaluating"):  # 使用 tqdm 显示评估进度
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)  # 将数据移动到设备上
            outputs = model(images)  # 前向传播
            _, predicted = torch.max(outputs, 1)  # 获取预测结果
            total += labels.size(0)  # 更新总样本数量
            correct += (predicted == labels).sum().item()  # 更新预测正确的样本数量

    print(f'训练集上的准确率: {100 * correct / total:.2f}%')  # 输出训练集上的准确率
