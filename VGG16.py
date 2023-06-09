import os, torch
import sys

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
from torch import optim

import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# 實作一個可以讀取 stanford dog (mini) 的 Pytorch dataset
class CovidDataset(Dataset):

    def __init__(self, filenames, labels, transform):
        self.filenames = filenames  # 資料集的所有檔名
        self.labels = labels  # 影像的標籤
        self.transform = transform  # 影像的轉換方式

    def __len__(self):
        return len(self.filenames)  # return DataSet 長度

    def __getitem__(self, idx):  # idx: Inedx of filenames
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)  # Transform image
        label = np.array(self.labels[idx])
        return image, label  # return 模型訓練所需的資訊


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transformer = transforms.Compose([
    transforms.Resize([299,299]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

test_transformer = transforms.Compose([
    transforms.Resize([299,299]),
    transforms.ToTensor(),
    normalize
])


def split_Train_Val_Data(data_path):
    dataset_train = ImageFolder(data_path + '/train')
    dataset_test = ImageFolder(data_path + '/test')

    train_inputs, test_inputs = [], []
    train_labels, test_labels = [], []

    for x, label in dataset_train.samples:
        train_inputs.append(x)
        train_labels.append(label)

    for x, label in dataset_test.samples:
        test_inputs.append(x)
        test_labels.append(label)

    # print(len(train_inputs), len(test_inputs))

    train_dataloader = DataLoader(CovidDataset(train_inputs, train_labels, train_transformer),
                                  batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(CovidDataset(test_inputs, test_labels, test_transformer),
                                 batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader


# 參數設定
batch_size = 32                                  # Batch Size
lr = 1e-3                                        # Learning Rate
epochs = 10                                      # epoch 次數
dataset = "./CT"


train_dataloader, test_dataloader = split_Train_Val_Data(dataset)
C = models.vgg16(pretrained=True).to(device)     # 使用內建的 model
optimizer_C = optim.SGD(C.parameters(), lr = lr) # 選擇你想用的 optimizer

# Loss function
criterion = nn.CrossEntropyLoss()                # 選擇想用的 loss function

loss_epoch_C = []
train_acc, test_acc = [], []
best_acc, best_auc = 0.0, 0.0

if __name__ == '__main__':
    best = C
    best_acc = 0.0

    for epoch in range(epochs):
        start_time = time.time()
        iter = 0
        correct_train, total_train = 0, 0
        correct_test, total_test = 0, 0
        train_loss_C = 0.0

        C.train()  # 設定 train 或 eval
        print('epoch: ' + str(epoch + 1) + ' / ' + str(epochs))

        # ---------------------------
        # Training Stage
        # ---------------------------
        for i, (x, label) in enumerate(train_dataloader):

            x, label = x.to(device), label.to(device)
            optimizer_C.zero_grad()  # 清空梯度
            train_output = C(x)  # 將訓練資料輸入至模型進行訓練 (Forward propagation)
            label = label.long()

            train_loss = criterion(train_output, label)  # 計算 loss
            train_loss.backward()  # 將 loss 反向傳播
            optimizer_C.step()  # 更新權重

            # 計算訓練資料的準確度 (correct_train / total_train)
            _, predicted = torch.max(train_output.data, 1)  # 取出預測的 maximum
            total_train += label.size(0)
            correct_train += (predicted == label).sum()
            train_loss_C += train_loss.item()
            iter += 1

        if best_acc < correct_train / total_train:
            best = C
            best_acc = correct_train / total_train

        print('Training epoch: %d / loss_C: %.3f | acc: %.3f' % (epoch + 1, train_loss_C / iter, correct_train / total_train))
        train_acc.append(100 * (correct_train / total_train).cpu())  # training accuracy
        # test_acc.append(100 * (correct_test / total_test).cpu())  # testing accuracy
        loss_epoch_C.append((train_loss_C / iter))  # loss

        end_time = time.time()
        print('Cost %.3f(secs)' % (end_time - start_time))

    # --------------------------
    # Testing Stage
    # --------------------------
    C=best
    C.eval()  # 設定 train 或 eval

    true_negatives, false_negatives = 0, 0
    true_positives, false_positives = 0, 0
    for i, (x, label) in enumerate(test_dataloader):
        with torch.no_grad():
            x, label = x.to(device), label.to(device)
            label = label.long()
            test_output = C(x)
            test_loss = criterion(test_output, label)

            _, predicted = torch.max(test_output.data, 1)
            total_test += label.size(0)
            correct_test += (predicted == label).sum()

            pred = np.array(predicted.cpu())
            lbl = np.array(label.cpu())
            # print(pred)
            # print(lbl)
            for idx in range(len(pred)):
                if pred[idx] == 0:
                    if lbl[idx] == 0:
                        true_negatives += 1
                    else:
                        false_negatives += 1
                else:
                    if lbl[idx] == 1:
                        true_positives += 1
                    else:
                        false_positives += 1

    # print('fp',false_positives)
    # print('tp',true_positives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    print('Testing F1 score: ', (2 * precision * recall) / (precision + recall))

    print('Testing acc: %.3f' % (correct_test / total_test))


fig_dir = './fig/'
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

plt.figure()
plt.plot(list(range(epochs)), loss_epoch_C) # plot your loss
plt.title('Training Loss')
plt.ylabel('loss'), plt.xlabel('epoch')
plt.legend(['loss_C'], loc = 'upper left')
plt.savefig(os.path.join(fig_dir, 'loss.png'))
plt.show()

plt.figure()
plt.plot(list(range(epochs)), train_acc)    # plot your training accuracy
plt.title('Training acc')
plt.ylabel('acc (%)'), plt.xlabel('epoch')
plt.legend(['training acc'], loc = 'upper left')
plt.savefig(os.path.join(fig_dir, 'acc.png'))
plt.show()
