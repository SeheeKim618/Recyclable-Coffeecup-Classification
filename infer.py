import os
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
import time
import warnings
from dataset.AttrDataset import get_transform
from tools.function import save_model, get_reload_weight
from config import argument_parser
import numpy as np

parser = argument_parser()
args = parser.parse_args()

warnings.filterwarnings("ignore")

train_tsfm, test_tsfm = get_transform()

data_path = "/home/jjunhee98/바탕화면/ksh/data/cafecup/"

#get ready for data
train_set = ImageFolder("/home/jjunhee98/바탕화면/ksh/data/cafecup/train/", train_tsfm)
val_set = ImageFolder("/home/jjunhee98/바탕화면/ksh/data/cafecup/valid/", train_tsfm)
test_set = ImageFolder("/home/jjunhee98/바탕화면/ksh/data/cafecup/test/", test_tsfm)

#load data
train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=8, num_workers=3)
val_loader = torch.utils.data.DataLoader(val_set, shuffle=True, batch_size=1, num_workers=3)
test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=1, num_workers=3)

dataloaders = {'train':train_loader, 'valid':val_loader, 'test': test_loader}
dataset_sizes = {'train': len(train_loader.dataset),'valid': len(val_loader.dataset), 'test':len(test_loader.dataset)}

#class
class_names = train_set.classes

#print info
print("num_train_dataset: ", len(train_set), "num_valid_dataset: ", len(val_set), "num_test_dataset: ", len(test_set))
print("classes: ", class_names)

#model
model = models.resnet18(pretrained=True)

#modify fc part in resnet
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

#model setting
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

model_path = args.model_ckpt

print("reloading pretrained models")
model = get_reload_weight(model_path, model)

# model to GPU
if torch.cuda.is_available():
    model = model.cuda()

def test(model, criterion, optimizer):

    model.train(False)

    running_loss = 0.0
    running_corrects = 0

    # 데이터 반
    for data in dataloaders['test']:
        # 입력 데이터 가져오기
        inputs, labels = data

        # 데이터를 Vaariable로 만듦
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # 파라미터 기울기 초기화
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # 통계
        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss.item() / dataset_sizes['test']
    test_acc = running_corrects.item() / dataset_sizes['test']

    print('{} |\t Loss: {:.4f}\t Accuracy: {:.4f}'.format(
        'Test', test_loss, test_acc))

if __name__ == "__main__":
    model_ft = test(model, criterion, optimizer)