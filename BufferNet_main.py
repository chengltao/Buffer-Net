# coding:UTF-8
import os
# -*- coding : utf-8-*-
# coding:unicode_escape
from BufferNet import Buffer
import torch.nn.functional as F
import torch.optim as optim
import os
from sklearn.metrics import confusion_matrix
import torch
import datetime


import matplotlib.pyplot as plt

import matplotlib

matplotlib.rc("font", family='DengXian')
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
dic_bac = {
"Stenotrophomonas": 10,
"Staphylococcus": 0,

"Serratia": 1,
"Pseudomonas":	2,
"Proteus":	3,

"Morganella": 4,
"Klebsiella": 5,

"Escherichia": 6,
"Enterococcus": 7,
"Burk":	8,
"Acinetobacter": 9


}
tick_marks = np.array(range(11)) + 0.5
##Draw the confusion matrix
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(11))
    xloc = []
    for ii, jj in enumerate(list(xlocations)):
        for k, v in dic_bac.items():
            if v == jj:
                xloc.append(k)
                break
    plt.xticks(xlocations, xloc, rotation=90)
    plt.yticks(xlocations, xloc)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

##Loading training data
class MyDatasets(Dataset):

    def __init__(self):
        self.data_dir = r"data"
        self.image_target_list = []
        with open(os.path.join(self.data_dir, 'train_address.txt'), 'r') as fp:
            content = fp.readlines()
            str_list = [s.rstrip().split() for s in content]
            self.image_target_list = [(x[0], int(x[1])) for x in str_list]

    def __getitem__(self, index):
        image_label_pair = self.image_target_list[index]
        img = np.load(image_label_pair[0])
        img = cv2.resize(img, (25,25))
        img = np.swapaxes(img, 0, 2)
        img = img[50:250, :, :]
        img = img[None, :, :, :]
        return img, image_label_pair[1]

    def __len__(self):
        return len(self.image_target_list)
##Loading testing data
class MyDatasets_validation(Dataset):

    def __init__(self):
        self.data_dir = r"data"
        self.image_target_list = []
        with open(os.path.join(self.data_dir, 'test_address.txt'), 'r') as fp:
            content = fp.readlines()
            str_list = [s.rstrip().split() for s in content]
            self.image_target_list = [(x[0], int(x[1])) for x in str_list]

    def __getitem__(self, index):
        image_label_pair = self.image_target_list[index]
        img = np.load(image_label_pair[0])
        img = cv2.resize(img, (25,25))
        img = np.swapaxes(img, 0, 2)
        img = img[50:250, :, :]
        img = img[None, :, :, :]
        return img, image_label_pair[1]

    def __len__(self):
        return len(self.image_target_list)
Loss_list = []
Accuracy_list = []
time_list = []

#iterating training
def train(args, model, device, dataloader_kwargs):
    train_loader = DataLoader(MyDatasets(), batch_size=args['batch_size'], shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
    print("training\n")
    for epoch in range(1, args['epochs'] + 1):
        start_time = datetime.datetime.now()
        loss = train_epoch(epoch, model, device, train_loader, optimizer)
        end_time = datetime.datetime.now()
        time_list.append((end_time - start_time).seconds)
        test(model, device)
        Loss_list.append(loss)
    return model

#testing
def test(model, device):
    print("testing\n")
    model.eval()
    validation_loader = DataLoader(MyDatasets_validation(), batch_size=args['batch_size'], shuffle=True, num_workers=5 )
    predicted = []
    target = []
    count = 1
    with torch.no_grad():
        acc = 0
        all_samples = 0
        for item in validation_loader:
            img, tag = item
            outputs = model(img.float().to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            predicted.extend(predict_y.cpu().numpy().tolist())
            tag = tag.cpu()
            all_samples = all_samples + len(tag)
            target.extend(tag.numpy().tolist())
            acc += torch.eq(predict_y, tag.to(device)).sum().item()
            count = count + 1

        a1 = np.array(target)
        b1 = np.array(predicted)
        accuracy1 = sum(a1 == b1) / len(target)
        print("accuracy----------", accuracy1)
    return target, predicted, accuracy1

#training
def train_epoch(epoch, model, device, data_loader, optimizer):

    model.train()
    pid = os.getpid()
    sum_loss = 0.0
    count = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        count = count + 1
        optimizer.zero_grad()
        output = model(data.float().to(device))
        loss = F.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()
        sum_loss = sum_loss + loss.item()
        if batch_idx % 10 == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                            100. * batch_idx / len(data_loader), loss.item()))
    return sum_loss/count



args = {
    'batch_size': 128,
    'test_batch_size': 128,
    'epochs': 30,
    'lr': 0.0001,
    'momentum': 0.9,
    'seed': 1,
    'log_interval': 10
}

#bufferNet_keys = ["layer1.0.downsampling.weight", "layer1.0.buffering1.weight", "layer1.0.buffering2.weight", "layer1.0.bn3.weight", "layer1.0.bn3.bias", "layer1.0.bn3.running_mean", "layer1.0.bn3.running_var", "layer1.0.buffering3.weight", "layer1.0.bn4.weight", "layer1.0.bn4.bias", "layer1.0.bn4.running_mean", "layer1.0.bn4.running_var", "layer2.0.downsampling.weight", "layer2.0.buffering1.weight", "layer2.0.buffering2.weight", "layer2.0.bn3.weight", "layer2.0.bn3.bias", "layer2.0.bn3.running_mean", "layer2.0.bn3.running_var", "layer2.0.buffering3.weight", "layer2.0.bn4.weight", "layer2.0.bn4.bias", "layer2.0.bn4.running_mean", "layer2.0.bn4.running_var", "layer3.0.downsampling.weight", "layer3.0.buffering1.weight", "layer3.0.buffering2.weight", "layer3.0.bn3.weight", "layer3.0.bn3.bias", "layer3.0.bn3.running_mean", "layer3.0.bn3.running_var", "layer3.0.buffering3.weight", "layer3.0.bn4.weight", "layer3.0.bn4.bias", "layer3.0.bn4.running_mean", "layer3.0.bn4.running_var", "layer4.0.downsampling.weight", "layer4.0.buffering1.weight", "layer4.0.buffering2.weight", "layer4.0.bn3.weight", "layer4.0.bn3.bias", "layer4.0.bn3.running_mean", "layer4.0.bn3.running_var", "layer4.0.buffering3.weight", "layer4.0.bn4.weight", "layer4.0.bn4.bias", "layer4.0.bn4.running_mean", "layer4.0.bn4.running_var"]
#ori_keys = ["layer1.1.conv1.weight", "layer1.1.bn1.weight", "layer1.1.bn1.bias", "layer1.1.bn1.running_mean", "layer1.1.bn1.running_var", "layer1.1.bn1.num_batches_tracked", "layer1.1.conv2.weight", "layer1.1.bn2.weight", "layer1.1.bn2.bias", "layer1.1.bn2.running_mean", "layer1.1.bn2.running_var", "layer1.1.bn2.num_batches_tracked", "layer1.0.conv1.weight", "layer1.0.conv2.weight", "layer2.1.conv1.weight", "layer2.1.bn1.weight", "layer2.1.bn1.bias", "layer2.1.bn1.running_mean", "layer2.1.bn1.running_var", "layer2.1.bn1.num_batches_tracked", "layer2.1.conv2.weight", "layer2.1.bn2.weight", "layer2.1.bn2.bias", "layer2.1.bn2.running_mean", "layer2.1.bn2.running_var", "layer2.1.bn2.num_batches_tracked", "layer2.0.conv1.weight", "layer2.0.conv2.weight", "layer3.1.conv1.weight", "layer3.1.bn1.weight", "layer3.1.bn1.bias", "layer3.1.bn1.running_mean", "layer3.1.bn1.running_var", "layer3.1.bn1.num_batches_tracked", "layer3.1.conv2.weight", "layer3.1.bn2.weight", "layer3.1.bn2.bias", "layer3.1.bn2.running_mean", "layer3.1.bn2.running_var", "layer3.1.bn2.num_batches_tracked", "layer3.0.conv1.weight", "layer3.0.conv2.weight", "layer4.1.conv1.weight", "layer4.1.bn1.weight", "layer4.1.bn1.bias", "layer4.1.bn1.running_mean", "layer4.1.bn1.running_var", "layer4.1.bn1.num_batches_tracked", "layer4.1.conv2.weight", "layer4.1.bn2.weight", "layer4.1.bn2.bias", "layer4.1.bn2.running_mean", "layer4.1.bn2.running_var", "layer4.1.bn2.num_batches_tracked", "layer4.0.conv1.weight", "layer4.0.conv2.weight"]
if __name__ == '__main__':
    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}
    print(device)
    model = Buffer().to(device)
    #ori = torch.load("./CNN_3D_1.pth")
    #model.load_state_dict(torch.load("./CNN_3D_1.pth"))
    # 训练
    #test(model, device)
    #model["layer1.0.downsampling.weight"] = ori["layer1.1.conv1.weight"]
    model = train(args, model, device, dataloader_kwargs)
    torch.save(model.state_dict(), "model/BufferNet.pth")

    model.load_state_dict(torch.load("model/BufferNet.pth"))
    target, predict_y, g = test(model, device)
    a = np.array(target)
    b = np.array(predict_y)
    accuracy = sum(a == b) / len(target)
    print("accuracy----------", accuracy)
    with open(r"CNN_3D_transferlearning.txt", "a+") as file:  # ”w"代表着每次运行都覆盖内容
        file.write(str(accuracy) + "\n")
    cm = confusion_matrix(target, predict_y, labels=range(0, 11))
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 8), dpi=120)

    ind_array = np.arange(11)
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    plt.show()



