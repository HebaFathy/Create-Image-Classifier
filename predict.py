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
from get_input_args import get_input_args
import json

in_arg = get_input_args()

with open(in_arg.json, 'r') as f:
    cat_to_name = json.load(f)
  
# Function that loads a checkpoint and rebuilds the model
def load_checkpoint(checkpoint_path):
    state = torch.load(checkpoint_path)
    learning_rate = state['learning_rate']
    class_to_idx = state['class_to_idx']
    
    if state['arch']== 'densenet121':
        model = models.densenet121(pretrained=True)
    elif state['arch']== 'vgg13':
        model = models.vgg13(pretrained=True)
    elif state['arch']== 'vgg19':
        model = models.vgg19(pretrained=True)
    elif state['arch']== 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError('Unkown network architecture', arch)
    c = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, state['hidden_units'])),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(state['hidden_units'], 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = c
    model.load_state_dict(state['state_dict'])
    optimizer = optim.Adam(model.classifier.parameters(), lr= learning_rate)
    optimizer.load_state_dict(state['optimizer'])
    print("Load checkpoint: '{}' with (arch={}, hidden_units={}, epochs={})".format( checkpoint_path, state['arch'], state['hidden_units'], 
    state['epochs']))
    return model

model= load_checkpoint(in_arg.checkpoint)

## Image Preprocessing

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    image = Image.open(image)
    processed_img = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    image = processed_img(image)
    image = np.array(image)
    print("Processed Image: {}".format(image.shape))
    return image

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    # Implement the code to predict the class from an image file
    model.eval()
    Gpu = False 
    if torch.cuda.is_available() and in_arg.gpu:
        Gpu = True
        model = model.cuda()
    else:
        model = model.cpu()
    
    processed_img= process_image(image_path)
    tensor = torch.from_numpy(processed_img).type(torch.FloatTensor) 
    if Gpu:
        imags = Variable(tensor.float().cuda(), volatile=True)
    else:       
        imags = Variable(tensor, volatile=True)
    imags = imags.unsqueeze(0)
    output = model.forward(imags)  
    ps = torch.exp(output).data.topk(topk)
    probabilities = ps[0].cpu() if Gpu else ps[0]
    classes = ps[1].cpu() if Gpu else ps[1]
    classes = classes.numpy()[0]
    probabilities = probabilities.numpy()[0]
    #model.train()    
    labels = list(cat_to_name.values())
    classes = [labels[x] for x in classes]
    return probabilities, classes

img_dr = in_arg.image
probs, classes = predict(img_dr, model, in_arg.topk)
print("Probabilities: {}".format(probs))
print("Classes: {}".format(classes))