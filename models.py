import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import time
import squeezenet as sqnet
import resnet

class Classifier(nn.Module):
    def __init__(self, in_features, n_classes, p=0, pooling='max'):
        super().__init__()
        if pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pooling == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(nn.BatchNorm1d(in_features),
                                        nn.Dropout(p),
                                        nn.Linear(in_features, 512),
                                        nn.ReLU(inplace=True),
                                        nn.BatchNorm1d(512),
                                        nn.Dropout(p*2),
                                        nn.Linear(512, n_classes))

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)



class CnnTransfer(nn.Module):
    def __init__(self, n_classes, head='original', p=.25):
        super().__init__()
        model = []
        if isinstance(head, nn.Module):
            model.append(sqnet.squeezenet1_1(pretrained=True, remove_top=True))
            # model.append(resnet.resnet50(pretrained=True, remove_top=True))
            model.append(head)
        elif head == 'original':
            model.append(sqnet.squeezenet1_1(pretrained=True, num_classes=n_classes))
            # model.append(resnet.resnet50(pretrained=True, num_classes=n_classes))
        elif head == 'fastai':
            model.append(sqnet.squeezenet1_1(pretrained=True, remove_top=True))
            # model.append(resnet.resnet50(pretrained=True, remove_top=True))
            model.append(Classifier(512, 2, p=p))
        self.model = nn.Sequential(*model)
        for name, param in self.named_parameters():
            if 'classifier' not in name and 'fc' not in name and 'bn' not in name:
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def compile(self, optimizer, lossfunc, scheduler=None):
        self.optimizer = optimizer
        self.lossfunc = lossfunc
        self.scheduler = scheduler

    def score(self, data):
        device = torch.device("cuda:0" if next(self.parameters()).is_cuda else "cpu")
        self.eval()   # Set model to evaluate mode
        with torch.no_grad():
            running_loss = 0.0
            running_corrects = 0
            total_size = 0
            # Iterate over data.
            for inputs, labels in data:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.forward(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.lossfunc(outputs, labels)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_size += inputs.size(0)
            epoch_loss = running_loss / total_size
            epoch_acc = running_corrects.double() / total_size
        return (epoch_loss, epoch_acc)

    def predict_prob(self, data):
        device = torch.device("cuda:0" if next(self.parameters()).is_cuda else "cpu")
        self.eval()   # Set model to evaluate mode
        with torch.no_grad():
            out = torch.cat([F.softmax(self.forward(inputs.to(device)), 1).cpu() for inputs in data])
        return out.numpy()

    def predict_class(self, data):
        device = torch.device("cuda:0" if next(self.parameters()).is_cuda else "cpu")
        self.eval()
        with torch.no_grad():
            out = torch.cat([torch.max(self.forward(inputs.to(device)), 1)[1].cpu() for inputs in data])
        return out.numpy()

    def fit(self, train_data, val_data, num_epochs):
        device = torch.device("cuda:0" if next(self.parameters()).is_cuda else "cpu")
        for epoch in range(num_epochs):
            since = time.time()
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            self.train()  # Set model to training mode
            running_loss = 0.0
            running_corrects = 0
            total_size = 0
            # Iterate over data.
            for inputs, labels in train_data:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.lossfunc(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_size += inputs.size(0)
                # print('.',  end='')
            print()
            epoch_loss = running_loss / total_size
            epoch_acc = running_corrects.double() / total_size
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            val_loss, val_acc = self.score(val_data)
            print(f'Validation loss: {val_loss:.4f} Acc: {val_acc:.4f}')
            print(f'{time.time()-since:.3f} seconds per epoch')
            print()
