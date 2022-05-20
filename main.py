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

warnings.filterwarnings("ignore")

#transform image function define
'''
#train-valid set
transform_train = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize((224, 224)),
                                      transforms.RandomHorizontalFlip(p=1),
                                      transforms.RandomVerticalFlip(),
                                      transforms.RandomRotation(90),
                                      transforms.ColorJitter(brightness=(0.2,3)),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#test set
transform_test = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
'''
train_tsfm, test_tsfm = get_transform()

data_path = "./data/cafecup/"

dataset = ImageFolder(data_path)

train_set, test_set = torch.utils.data.random_split(dataset, [750, 150])
train_set, val_set = torch.utils.data.random_split(train_set, [700, 50])

train_set = train_tsfm(train_set)
val_set = train_tsfm(val_set)
test_set = test_tsfm(test_set)

'''
#get ready for data
train_set = ImageFolder("./data/train", transform_train)
valid_set = ImageFolder("./data/valid", transform_train)
test_set = ImageFolder("./data/test", transform_test)
'''
#class
class_names = train.classes

#load data
train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=32, num_workers=3)
val_loader = torch.utils.data.DataLoader(val_set, shuffle=True, batch_size=1, num_workers=3)
test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=1, num_workers=3)

dataloaders = {'train':train_data_gen, 'valid':valid_data_gen, 'test': test_data_gen}
dataset_sizes = {'train': len(train_data_gen.dataset),'valid': len(valid_data_gen.dataset), 'test':len(test_data_gen.dataset)}

#print info
print("sizeof_train_dataset : ", len(train_set))
print("sizeof_validation_dataset : ", len(valid_set))
print("sizeof_test_dataset : ", len(test_set))
print("class_names : ", class_names)

#model
model = models.resnet18(pretrained=True)

#modify fc part in resnet
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

#model setting
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9, weight_decay=1e-5)

LR_scheduler = lr_scheduler.StepLR(optimizer,step_size=3, gamma=0.1)

#model checkpoint
checkpoint_path = './exp_result/checkpoint1'
file_name = 'checkpoint1test_lr_001_SGD_class4_freeze34-2_new_ver2.pt'

#model load
if os.path.isdir(checkpoint_path) and os.path.isfile(checkpoint_path + '/' + file_name):
    checkpoint = torch.load(checkpoint_path + '/' + file_name)
    net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    LR_scheduler.load_state_dict(checkpoint['schedular'])
    epoch_cnt = checkpoint['epoch_cnt']
    print("epoch : ",epoch_cnt)
    print("model_loaded!")

#model to gpu
if torch.cuda.is_available():
    net = net.cuda()

def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
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
                if not os.path.isdir('/home/jjunhee98/model1_plastic_ver2/trained_model/checkpoint1'):
                    os.mkdir('/home/jjunhee98/model1_plastic_ver2/trained_model/checkpoint1')
                torch.save(state, '/home/jjunhee98/model1_plastic_ver2/trained_model/checkpoint1/' + file_name)
                print('Model Saved!')

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

    test_loss = running_loss.item() / dataset_sizes[test]
    test_acc = running_corrects.item() / dataset_sizes[test]

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        'test', test_loss, test_acc))

    # 최적의 모델 가중치 로딩
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    model_ft = train_model(net, criterion, optimizer, LR_scheduler, num_epochs=50)
