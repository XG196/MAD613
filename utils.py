import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
from matplotlib.pyplot import imsave
import torchvision.transforms.functional as TF

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


nc = 3
imsize = 256
loader = transforms.Compose([
    #transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor
unloader = transforms.ToPILImage()  # reconvert into PIL image


def step_size(lamda0 ,opt ,rate1 ,rate2 ,iteration):


    if iteration < opt:
        lamda = lamda0 *(rate1**iteration)
    else:
        lamda = lamda0 *(rate1**iteration) *(rate2**(iteration-opt))

    return lamda





def cv_converter(img):
    image = Image.fromarray(img[... ,::-1])
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)

# convert to tensor and unsqueeze （NCWH）
def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)




def imshow(tensor, title=None,full_name = None):
    tensor = torch.clamp(tensor ,0 ,1)
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    image = np.array(image)
    imsave(full_name+'.jpg',image,dpi = 300)


def imshow1(tensor, title=None):
    tensor = torch.clamp(tensor ,0 ,1)
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    #plt.savefig('pebbles_noise6_3.jpg')
    plt.show()


# add gaussian noise to image
# noise variance is specified
def gaussian_noise(level,ref):

    ref = ref * 255
    noise = torch.randn(1, nc, imsize, imsize) * torch.sqrt((torch.tensor([2.0]) ** level))
    imgn = (ref + noise.to(device)) / 255
    imgn = torch.clamp(imgn, 0, 1)

    return imgn

def gaussian_blur(level,ref,name):

    temp =  cv2.imread("./data/texture/" + name + ".jpg")
    temp = cv2.GaussianBlur(temp,ksize = (0,0),sigmaX= level)
    temp = temp[...,::-1]
    temp = np.transpose(temp,(2,0,1))[np.newaxis,...]/255.0
    imgn = torch.from_numpy(temp).float().to(device)
    imshow(imgn, title=None,full_name = 'distorted image'+name)

    return imgn


def gamma(level,ref,name):


    temp = Image.open("./data/texture/" + name + ".jpg")
    temp = TF.adjust_gamma(temp, gamma= level, gain=1)
    imgn = TF.to_tensor(temp).unsqueeze(0).to(device)

    #print(imgn.shape)
    #plt.imshow(temp)
    #plt.show()
    
    return imgn


def jpeg(level,ref,name):


    temp = Image.open("./data/texture/" + name + ".jpg")
    #print(level)
    temp.save('jpeg'+'_'+ str(level) + '_' +name+'.jpg', quality= level)
    temp = Image.open('jpeg'+'_'+ str(level) + '_' + name + ".jpg")

    #plt.imshow(temp)
    #plt.show()
    
    temp = np.transpose(temp,(2,0,1))[np.newaxis,...]/255.0
    imgn = torch.from_numpy(temp).float().to(device)
    #print(imgn.shape)

    return imgn



















