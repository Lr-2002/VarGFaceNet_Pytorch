import numpy as np
import torchvision.transforms as transforms
import torch
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Dataset
def load_np():
    # 对于 SVM 或随机森林（加载 .npy 格式）
    X_train = np.load("/home/lr-2002/code/VarGFaceNet_Pytorch/data/X_train.npy")
    X_test = np.load("/home/lr-2002/code/VarGFaceNet_Pytorch/data/X_test.npy")
    y_train = np.load("/home/lr-2002/code/VarGFaceNet_Pytorch/data/y_train.npy")
    y_test = np.load("/home/lr-2002/code/VarGFaceNet_Pytorch/data/y_test.npy")
    return X_train, X_test, y_train, y_test
def load_pt():
    # 对于 VarGFaceNet（加载 .pt 格式）
    train_data = torch.load("/home/lr-2002/code/VarGFaceNet_Pytorch/data/train_data.pt")
    test_data = torch.load("/home/lr-2002/code/VarGFaceNet_Pytorch/data/test_data.pt")
    X_train, y_train = train_data
    X_test, y_test = test_data
    return X_train, X_test, y_train, y_test


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        # transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        #     transforms.RandomRotation(degrees=10),  # 随机旋转 ±10 度
        #     transforms.Normalize((0.5,), (0.5,))  # 标准化到 [-1, 1] 范围，适合灰度图像
        # ])
        #
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            # transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # 应用转换
        if self.transform:
            image = self.transform(image)

        return image, label


def get_pt_dataloader(batch_size = 64):
    # 设置批量大小
    X_train, X_test, y_train, y_test = load_pt()
    # 创建 TensorDataset

    # 将每个图像从 1850 还原为 (50, 37) 的形状
    X_train_images = X_train.view(-1, 50, 37)
    X_test_images = X_test.view(-1, 50, 37)

    # 例如，使用 unsqueeze 添加一个通道维度
    X_train_images = X_train_images.unsqueeze(1)  # 结果为 (N, 1, 50, 37)
    X_test_images = X_test_images.unsqueeze(1)

    train_dataset = CustomDataset(X_train_images, y_train)
    test_dataset = CustomDataset(X_test_images, y_test)
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

    # 保存报告为 CSV

def save_csv(report, file_name):
    import os
    file_name = os.path.join('/home/lr-2002/code/VarGFaceNet_Pytorch/report', file_name)
    report_df = pd.DataFrame(report).transpose()     #转置数据框以便于查看
    report_df.to_csv(file_name, index=True)  # 保存到 CSV 文件
    print(f"Classification report saved to {file_name}")
if __name__=='__main__':
    # load_np()
    train_loader, test_loader = get_pt_dataloader(1)
    print(len(train_loader), len(test_loader))