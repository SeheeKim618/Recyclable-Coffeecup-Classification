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

data_path = os.path.join("/home/jjunhee98/바탕화면/ksh/data", f"{args.dataset}")
#data_path = "/home/jjunhee98/바탕화면/ksh/data/cafecup/"

#get ready for data
train_set = ImageFolder("/home/jjunhee98/바탕화면/ksh/data/cafecup/train/", train_tsfm)
val_set = ImageFolder("/home/jjunhee98/바탕화면/ksh/data/cafecup/valid/", train_tsfm)
test_set = ImageFolder("/home/jjunhee98/바탕화면/ksh/data/cafecup/test/", test_tsfm)

#load data
train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=args.train_batchsize, num_workers=3)
val_loader = torch.utils.data.DataLoader(val_set, shuffle=True, batch_size=args.valid_batchsize, num_workers=3)
test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=args.test_batchsize, num_workers=3)

dataloaders = {'train':train_loader, 'valid':val_loader, 'test': test_loader}
dataset_sizes = {'train': len(train_loader.dataset),'valid': len(val_loader.dataset), 'test':len(test_loader.dataset)}

#class
class_names = train_set.classes

#print info
print("num_train_dataset: ", len(train_set), ",\tnum_valid_dataset: ", len(val_set), ",\tnum_test_dataset: ", len(test_set))
print("classes: ", class_names)

#model
model = models.resnet18(pretrained=True)

#modify fc part in resnet
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

#model setting
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
epochs = args.train_epoch
LR_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

'''
#model checkpoint
model_path = './exp_result/checkpoint1'
file_name = 'checkpoint1test_lr_001_SGD_class4_freeze34-2_new_ver2.pt'

#model load
if os.path.isdir(checkpoint_path) and os.path.isfile(checkpoint_path + '/' + file_name):
    checkpoint = torch.load(checkpoint_path + '/' + file_name)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    LR_scheduler.load_state_dict(checkpoint['schedular'])
    epoch_cnt = checkpoint['epoch_cnt']
    print("epoch : ",epoch_cnt)
    print("model_loaded!")
'''
#model to GPU
if torch.cuda.is_available():
    model = model.cuda()

def train(model, criterion, optimizer, scheduler, epochs):

    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # 각 Epoch은 학습 단계와 검증 단계를 거침
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # 학습 모드 설정
            else:
                model.train(False)  # 검증 모드 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터 반
            for data in dataloaders[phase]:
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

                # 학습 단계에서만 수행, 역전파 + 옵티마이즈(최적화)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                # 통계
                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss.item() / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]

            print('{} |\t Loss: {:.4f}\t Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 모델 복사(Deep Copy)
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                state = {
                    'epoch_cnt': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'schedular': scheduler.state_dict()
                }
                save_model(args.model_ckpt, state, args.ckpt_name)

    '''
    #model test
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
    '''

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Valid Accuracy: {:4f}'.format(best_acc))
    #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
    #    'test', test_loss, test_acc))

    # 최적의 모델 가중치 로딩
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    model_ft = train(model, criterion, optimizer, LR_scheduler, epochs)
