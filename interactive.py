import seaborn as sns
from matplotlib import pyplot as plt
import json
import glob
import numpy as np
import cv2
import skimage
import skimage.io as io
import skimage.transform as transform
import os.path
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from skimage.exposure import adjust_gamma, adjust_log, adjust_sigmoid
import image.cv2transforms as cv2transforms
import time
import importlib
import image.utils
from image.utils import CropDataset
import  models as mymodels

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
# import keras
from fastai import DataBunch, accuracy, Recorder
from fastai.vision import ConvLearner, Image, tvm

# Read data
imgs, labels = image.utils.getCropData('/home/fast/Automate/20x/', 60, 30)

posidxs = np.nonzero(labels)[0]
negidxs = np.nonzero(labels<1)[0]

testidx = np.concatenate((posidxs[-50:], negidxs[-50:]))
np.random.shuffle(testidx)
valididx = np.concatenate((posidxs[-100:-50], negidxs[-100:-50]))
np.random.shuffle(valididx)
trainidx = np.concatenate((np.repeat(posidxs[:-100], 6), negidxs[:-100]))
np.random.shuffle(trainidx)

testImgs, validImgs, trainImgs = imgs[testidx], imgs[valididx], imgs[trainidx]
testY, validY, trainY = labels[testidx], labels[valididx], labels[trainidx]
chan_means = np.mean(imgs, axis=(0,1,2))
chan_std = np.std(imgs, axis=(0,1,2))

##### Fast.ai
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_size = 224

data_transforms = {'train': transforms.Compose([cv2transforms.AutoContrast(),
                                                cv2transforms.Resize(input_size),
                                                cv2transforms.RandomHflip(),
                                                cv2transforms.RandomVflip(),
                                                cv2transforms.RandomRotation(5),
                                                cv2transforms.Typecast(np.float32),
                                                transforms.ToTensor()]),
                    'val': transforms.Compose([cv2transforms.AutoContrast(),
                                               cv2transforms.Resize(input_size),
                                               cv2transforms.Typecast(np.float32),
                                               transforms.ToTensor()])}




image_datasets = {'train': CropDataset((trainImgs, trainY), data_transforms['train']),
                  'val': CropDataset((validImgs, validY), data_transforms['val'])}

importlib.reload(image.utils)
importlib.reload(image.cv2transforms)
from image.utils import CropDataset
#
# test_dataset = CropDataset((validImgs, validY), cv2transforms.Resize(input_size))
#
# test_dataset[2:13][0].shape

image_datasets['train'][1:3][0].shape

databanch = DataBunch.create(image_datasets['train'],image_datasets['val'], bs = 32, device=device)

img, label = databanch.train_ds[17]

databanch.c = 2

Image(img)


# torch.cuda.empty_cache()

learn = ConvLearner(databanch, tvm.vgg19_bn, metrics=accuracy)

learn.fit(5)

yhat, y = learn.get_preds(is_test = False)
_, yhatidx = torch.max(yhat, 1)

learn.opt.opt_keys


confusion = confusion_matrix(yhatidx.numpy(), y.numpy())
sns.heatmap(confusion, annot=True, cmap='Blues')

np.where(np.logical_and(y.numpy(), y.numpy()!=yhatidx.numpy()))

np.nonzero(y.numpy())

image.utils.showChannels(validImgs[30])
image.utils.showChannels(validImgs[7])

learn.sched.plot_lr()



[name for name, param in learn.model.named_parameters() if param.requires_grad]

sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, learn.model.parameters())])



# Transfer learning

importlib.reload(image.cv2transforms)

rotate = cv2transforms.RandomRotation(30)

importlib.reload(image.utils)

image.utils.showChannels(imgs[0])

image.utils.showChannels(rotate(imgs[0]))



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

importlib.reload(mymodels)

model = mymodels.ResnetTransfer().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
input_size = 224
data_transforms = {
    'train': transforms.Compose([
        cv2transforms.Resize(input_size),
        cv2transforms.RandomHflip(),
        cv2transforms.RandomVflip(),
        cv2transforms.RandomRotation(5),
        cv2transforms.Typecast(np.float32),
        transforms.ToTensor(),
        transforms.Normalize(chan_means, chan_std)
    ]),
    'val': transforms.Compose([
        cv2transforms.Resize(input_size),
        cv2transforms.Typecast(np.float32),
        transforms.ToTensor(),
        transforms.Normalize(chan_means, chan_std)
    ])}

image_datasets = {'train': CropDataset((trainImgs, trainY), data_transforms['train']),
                  'val': CropDataset((validImgs, validY), data_transforms['val'])}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                              shuffle=False, num_workers=4)
              for x in ['train', 'val']}

model.compile(optimizer, criterion)

sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])

model.fit(dataloaders['train'], dataloaders['val'], 5)

for name, param in model.resnet.layer4.named_parameters():
    if name.startswith('2'):
        param.requires_grad = True
model.optimizer = optim.Adam(model.parameters(), lr=1e-4)
sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])
model.fit(dataloaders['train'], dataloaders['val'], 5)


time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

imgs_scaled = imgs / np.std(imgs, axis=(1,2)).reshape((imgs.shape[0],1,1,imgs.shape[-1]))

showChannels(imgs_scaled[8])

# data = imgs_scaled.reshape((imgs_scaled.shape[0],-1))
# data_scaled = data

data = imgs.reshape((imgs_scaled.shape[0],-1))
data_scaled = preprocessing.scale(data)

# clf = svm.SVC(class_weight='balanced', verbose=2, max_iter=10000, C=100)
# clf = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=0)
clf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=0)
# clf = LogisticRegression(random_state=0, solver='liblinear', penalty='l2', class_weight='balanced', C=1)
posSamples = np.nonzero(labels)[0]
negSamples = np.nonzero(labels<1)[0]

%time clf.fit(data_scaled[:500,:], labels[:500])
# clf.score(data[501:1000], labels[501:1000])
# sum(labels[501:1000])
testidx = tuple(posSamples[-100:])+tuple(negSamples[-100:])

Yhat = clf.predict(data_scaled[testidx,])
Yscores = clf.predict_proba(data_scaled[testidx,])[:,1]

# sum(Yhat)
confusion = confusion_matrix(labels[testidx,], Yhat)
precision, recall, thresholds = precision_recall_curve(labels[tuple(posSamples[-100:])+tuple(negSamples[-100:]),], Yscores)
fpr, tpr, thresholds = roc_curve(labels[tuple(posSamples[-100:])+tuple(negSamples[-100:]),], Yscores)


incorrect = np.array(testidx)[np.nonzero(Yhat != labels[testidx,])]
correctpos = np.array(testidx)[np.nonzero(np.logical_and(Yhat == 1, labels[testidx,]))]
negatives = np.array(testidx)[np.nonzero(labels[testidx,]==0)]

showChannels(imgs[negSamples[2521]])

showChannels(imgs[negatives[60]])

showChannels(imgs[correctpos[20]])

showChannels(imgs[incorrect[5]])

sns.lineplot(fpr, tpr)

sns.lineplot(recall, precision)

sns.heatmap(confusion, annot=True, cmap='Blues')
clf.score(data_scaled[testidx,], labels[testidx,])
clf.score(data_scaled[:500,], labels[:500,])

boxes = []
pathnames = glob.iglob(DIR+'nuclei/'+'*boxesF.json')
# Read selection boxes
for pathname in pathnames:
    with open(pathname, 'r') as f:
        boxes = boxes + json.load(f)
# Filter nuclei
boxes = [box for box in boxes if box['type'] == NUCLEUS]
Ntotal = len(boxes)
Nfused = len(list(filter(lambda x: x['fusion']==FUSION, boxes)))
print(Nfused, Ntotal)
sizes = np.array([max(box['bounds'][-2:]) for box in boxes])
ratios = np.array([min(box['bounds'][-2:])/max(box['bounds'][-2:]) for box in boxes])
sns.distplot(sizes)
imgsize = (2240,1828)
# cropsizes = [getCrop(box, 60, 30, imgsize) for box in boxes]
# sum([crop[0] for crop in cropsizes if len(crop)==1])
sns.distplot(cropsizes, hist_kws={'log':True})

img = cv2.imread('/home/fast/Automate/20x/nuclei/image000000.tif', cv2.IMREAD_ANYDEPTH)
img2 = io.imread('/home/fast/Automate/20x/red/image000162.tif').T
with open('/home/fast/Automate/20x/nuclei/image000162boxesF.json', 'r') as f:
    img2boxes = json.load(f)
x, y, w, h = img2boxes[30]['bounds']
plt.imshow(adjust_gamma(img2[x-50:x+w+50,y-50:y+h+50], .1), cmap='gray')
plt.imshow(img2, cmap='gray')

#### Keras model

importlib.reload(image.utils)

datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    horizontal_flip=True,
    vertical_flip=True)

datagen.fit(trainImgs)

batch_size=32

train_generator = datagen.flow(trainImgs,trainY, batch_size=batch_size)

valid_generator = datagen.flow(validImgs,validY, batch_size=batch_size)

base_model = keras.applications.ResNet50(include_top=False, pooling='avg')

input1 = keras.Input(shape=(120,120,3))
hidden = keras.layers.UpSampling2D(size=(2, 2))(input1)
hidden = keras.layers.Cropping2D(8)(hidden)
hidden = base_model(hidden)
hidden = keras.layers.ReLU()(hidden)
hidden = keras.layers.Dropout(.2)(hidden)
hidden = keras.layers.Dense(256)(hidden)
hidden = keras.layers.ReLU()(hidden)
out = keras.layers.Dense(1, activation ='sigmoid')(hidden)
kerasmodel = keras.Model(input1, out)

for layer in base_model.layers:
    layer.trainable = False

kerasmodel.summary()


kerasmodel.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
kerasmodel.fit_generator(train_generator, steps_per_epoch=trainImgs.shape[0]//batch_size, validation_data= valid_generator, validation_steps=validImgs.shape[0]//batch_size,epochs=10)
