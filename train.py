import numpy as np
import time
import torch
import sys
from PIL import Image
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict

# Define your transforms for the training, validation, and testing sets
data_transforms = {
     'training' : transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),transforms.RandomRotation(30),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]),
                                                            
    'validation' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]),

    'testing' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])}
# Load the datasets with ImageFolder
image_datasets = {
    'training' : datasets.ImageFolder(train_dir, transform=data_transforms['training']),
    'testing' : datasets.ImageFolder(test_dir, transform=data_transforms['testing']),
    'validation' : datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])}

# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    'training' : torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
    'testing' : torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64, shuffle=False),
    'validation' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64, shuffle=True)
}

# network model
model = models.densenet121(pretrained=True)


# Build and train your network
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, in_arg.hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(in_arg.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier
criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr= in_arg.learning_rate)

steps = 0
epochs = in_arg.epochs
Gpu = False
print_every = 10
train_losses, test_losses = [], []

if torch.cuda.is_available() and in_arg.gpu:
    Gpu = True
    model.cuda()
else:
    model.cpu()

for epoch in range(epochs):
    running_loss = 0
    for imgs, labels in iter(dataloaders['training']):
        steps += 1
        if Gpu:
            imgs = Variable(imgs.float().cuda())
            labels = Variable(labels.long().cuda()) 
        else:
            imgs = Variable(imgs)
            labels = Variable(labels) 

        optimizer.zero_grad()
        output = model.forward(imgs)
        loss = criterion(output, labels)

        running_loss += loss.data[0]
        if steps % print_every == 0:
            accuracy = 0
            test_loss = 0
    
            for imgs, labels in iter(dataloaders['validation']):
                if torch.cuda.is_available() and in_arg.gpu:
                    imgs = Variable(imgs.float().cuda(), volatile=True)
                    labels = Variable(labels.long().cuda(), volatile=True) 
                else:
                    imgs = Variable(imgs, volatile=True)
                    labels = Variable(labels, volatile=True)

                output = model.forward(imgs)
                test_loss += criterion(output, labels).data[0]
                ps = torch.exp(output).data 
                equality = (labels.data == ps.max(1)[1])
                accuracy += equality.type_as(torch.FloatTensor()).mean()
            running_loss = 0

# Do validation on the test set
accuracy = 0
test_loss = 0
for imgs, labels in iter(dataloaders['testing']):
    if torch.cuda.is_available() and in_arg.gpu:
        imgs = Variable(imgs.float().cuda(), volatile=True)
        labels = Variable(labels.long().cuda(), volatile=True) 
    else:
        imgs = Variable(imgs, volatile=True)
        labels = Variable(labels, volatile=True)
    output = model.forward(imgs)
    test_loss += criterion(output, labels).data[0]
    ps = torch.exp(output).data 
    equality = (labels.data == ps.max(1)[1])
    accuracy += equality.type_as(torch.FloatTensor()).mean()

print( f"Test loss: {test_loss/len(dataloaders['testing']):.3f}.. "
       f"Test accuracy: {accuracy/len(dataloaders['testing']):.3f}")

# Save the checkpoint 
model.class_to_idx = image_datasets['training'].class_to_idx
model.cpu()
torch.save({ 'arch': in_arg.arch,
             'learning_rate': in_arg.learning_rate,
             'hidden_units': in_arg.hidden_units,
             'epochs': in_arg.epochs,
             'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict(),
             'class_to_idx' : model.class_to_idx}, 
             in_arg.save_dir + 'checkpoint.pth')
