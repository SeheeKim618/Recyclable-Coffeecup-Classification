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
import numpy as np

warnings.filterwarnings("ignore")

train_tsfm, test_tsfm = get_transform()

data_path = "./data/cafecup/"


#get ready for data
train_set = ImageFolder("./data/cafecup/train/", train_tsfm)
val_set = ImageFolder("./data/cafecup/valid/", train_tsfm)
test_set = ImageFolder("./data/cafecup/test/", test_tsfm)

#load data
train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=8, num_workers=3)
val_loader = torch.utils.data.DataLoader(val_set, shuffle=True, batch_size=1, num_workers=3)
test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=1, num_workers=3)

dataloaders = {'train':train_loader, 'valid':val_loader, 'test': test_loader}
dataset_sizes = {'train': len(train_loader.dataset),'valid': len(val_loader.dataset), 'test':len(test_loader.dataset)}

#class
class_names = train_set.classes

#print info
print("num_train_dataset : ", len(train_set))
print("num_valid_dataset : ", len(val_set))
print("num_test_dataset : ", len(test_set))
print("classes: ", class_names)

#model
model = models.resnet18(pretrained=True)

#modify fc part in resnet
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

#model setting
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)

LR_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

#model checkpoint
checkpoint_path = './exp_result/checkpoint1'
file_name = 'checkpoint1test_lr_001_SGD_class4_freeze34-2_new_ver2.pt'

#model load
'''
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

def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  
            else:
                model.train(False) 

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:

                inputs, labels = data

                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss.item() / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                state = {
                    'epoch_cnt': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'schedular': scheduler.state_dict()
                }
                if not os.path.isdir('./exp_result/checkpoint1'):
                    os.mkdir('./exp_result/checkpoint1')
                torch.save(state, './exp_result/checkpoint1/' + file_name)
                print('Model Saved!')

    #model test
    model.train(False)

    running_loss = 0.0
    running_corrects = 0

    for data in dataloaders['test']:

        inputs, labels = data

        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss.item() / dataset_sizes['test']
    test_acc = running_corrects.item() / dataset_sizes['test']

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        'test', test_loss, test_acc))

    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    model_ft = train_model(model, criterion, optimizer, LR_scheduler, num_epochs=50)
