
#Libraries
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# My files
import DenseNet
import util

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

###################################
# Load Data
###################################

DATA_DIR = "C:\\Users\\James\\Documents\\ml projects\\Chest_DNN\\data\\data.csv"
IMAGE_DIR = "C:\\Users\\James\\Documents\\ml projects\\Chest_DNN\\data\\images_01\\images\\"

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#import data
chestxray_dataset = util.ChestXrayDataset(csv_file=DATA_DIR, root_dir=IMAGE_DIR,transform=transform)

dataloader = DataLoader(chestxray_dataset, batch_size=4,
                        shuffle=True, num_workers=0)

batch_size = 16
validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(chestxray_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(chestxray_dataset, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(chestxray_dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

###################################
# Train Model
###################################
#From the paper
growth_rate = 32
block_config = (6, 12, 24, 16) # this is for densenet121
num_init_features= 64
model = DenseNet(growth_rate, block_config, num_init_features)

epoches = 50
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(epoches):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
#save model
PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)
#reload model
model = DenseNet()
model.load_state_dict(torch.load(PATH))