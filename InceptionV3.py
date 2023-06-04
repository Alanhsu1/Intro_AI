import numpy as np 
import torchvision.transforms as transforms
import torchvision.models as models
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import sklearn

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.datasets import ImageFolder
from torch import optim
from torchsummary import summary


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CovidDataset(Dataset):

    def __init__(self, filenames, labels, transform):
        self.filenames = filenames  # all file
        self.labels = labels  # image tag
        self.transform = transform  # transform method

    def __len__(self):
        return len(self.filenames)  # return DataSet 長度

    def __getitem__(self, idx):  # idx: Inedx of filenames
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)  # Transform image
        label = np.array(self.labels[idx])
        return image, label  # return 模型訓練所需的資訊


# Transformer
train_transformer = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

test_transformer = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def split_Train_Val_Data(data_path):
    dataset_train = ImageFolder(data_path + '/train')
    dataset_test = ImageFolder(data_path + '/test')
    
    
    character_train = [[] for c in range(len(dataset_train.classes))]
    character_test = [[] for c in range(len(dataset_test.classes))]


    for img, label in dataset_train.samples:
        character_train[label].append(img)

    for img, label in dataset_test.samples:
        character_test[label].append(img)
    # print(character)

    train_inputs, test_inputs = [], []
    train_labels, test_labels = [], []

    for label, data in enumerate(character_train):
        for x in data:
            train_inputs.append(x)
            train_labels.append(label)

    for label, data in enumerate(character_test):
        for x in data:
            test_inputs.append(x)
            test_labels.append(label)
    
    print(len(train_inputs), len(test_inputs))

    train_dataloader = DataLoader(CovidDataset(train_inputs, train_labels, train_transformer),
                                  batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(CovidDataset(test_inputs, test_labels, test_transformer),
                                 batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader
        

batch_size = 32
learning_rate = 1e-3
epochs = 10
dataset = './test'

train_dataloader, test_dataloader = split_Train_Val_Data(dataset)
C = models.inception_v3(pretrained=False).to(device)
optimizer_C = optim.SGD(C.parameters(), lr=learning_rate)
summary(C, (3, 299, 299))
criterion = nn.CrossEntropyLoss()

loss_epoch_C = []
train_accuracy, test_accuracy = [], []
best_accuracy, best_auc = 0.0, 0.0

# main
if __name__ == '__main__':
    best = C
    best_acc = 0.0
    for epoch in range(epochs):
        # count time
        start_time = time.time()
        iteration = 0
        correct_train, total_train = 0, 0
        correct_test, total_test = 0, 0
        train_loss_C = 0.0
        
        C.train()
        print('epoch: ' + str(epoch + 1) + ' / ' + str(epochs))

        # Training Stage
        for i, (x, label) in enumerate(train_dataloader):

            x, label = x.to(device), label.to(device)
            optimizer_C.zero_grad()
            train_output = C(x)
            label = label.long()
            train_output = train_output.logits

            train_loss = criterion(train_output, label)
            train_loss.backward()
            optimizer_C.step()

            # 計算訓練資料的準確度 (correct_train / total_train)
            _, predicted = torch.max(train_output.data, 1)  # 取出預測的 maximum
            total_train += label.size(0)
            correct_train += (predicted == label).sum()
            train_loss_C += train_loss.item()
            iteration += 1

        if best_acc < correct_train / total_train:
            best = C
            best_acc = correct_train / total_train

        print('Training epoch: %d / loss_C: %.3f | acc: %.3f' % \
              (epoch + 1, train_loss_C / iteration, correct_train / total_train))

        # --------------------------
        # Testing Stage
        # --------------------------

        train_accuracy.append(100 * (correct_train / total_train).cpu())  # training accuracy
        # test_accuracy.append(100 * (correct_test / total_test).cpu())  # testing accuracy
        loss_epoch_C.append((train_loss_C / iteration))  # loss

        end_time = time.time()
        print('Cost %.3f(secs)' % (end_time - start_time))

    C = best
    C.eval()
    for i, (x, label) in enumerate(test_dataloader):
        with torch.no_grad():
            x, label = x.to(device), label.to(device)
            label = label.long()
            test_output = C(x)
            test_loss = criterion(test_output, label)

            _, predicted = torch.max(test_output.data, 1)
            total_test += label.size(0)
            correct_test += (predicted == label).sum()

            y_pred = []
            y_label = []
            y_label.append(label)
            y_pred.append(predicted)
    print('Testing F1 score: ', sklearn.metrics.f1_score(y_label, y_pred))
    print('Testing acc: %.3f' % (correct_test / total_test))

plt.figure()
plt.plot(list(range(epochs)), loss_epoch_C) # plot your loss
plt.title('Training Loss')
plt.ylabel('loss'), plt.xlabel('epoch')
plt.legend(['loss_C'], loc = 'upper left')
plt.savefig('loss.png')
plt.show()

plt.figure()
plt.plot(list(range(epochs)), train_accuracy)    # plot your training accuracy
plt.title('Training acc')
plt.ylabel('acc (%)'), plt.xlabel('epoch')
plt.legend(['training acc'], loc = 'upper left')
plt.savefig('acc.png')
plt.show()
