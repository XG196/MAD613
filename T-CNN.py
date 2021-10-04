import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import copy

from models import *

alexnet = models.alexnet(pretrained=True)
tcnn = alexnet.features[0:5]
_, _, pool_size, _= tcnn(ref_img).shape
global_avg_pool = nn.AvgPool2d(kernel_size = pool_size)
tcnn.add_module('global_avg_pool',global_avg_pool)
nc = 3
imsize = 256

weight_tcnn = 1e5
class TcnnLoss(nn.Module):
    def __init__(self, target_feature):
        super(TcnnLoss, self).__init__()
        self.target = target_feature.detach()
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def get_tcnn_model_and_losses(normalization_mean, normalization_std,
                              style_img,
                              device):
    net = copy.deepcopy(tcnn)

    style_img = style_img.to(device)

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    model = nn.Sequential(normalization)
    model.add_module('tcnn', net)
    losses = []

    target_feature = model(style_img)

    loss = TcnnLoss(target_feature)
    model.add_module('loss', loss)

    losses.append(loss)

    return model, losses


# called when gradients are not needed

def model_tcnn(img, ref):
    img = img.reshape(1, nc, imsize, imsize)
    img = img.to(device)
    ref = ref.to(device)

    tcnn, losses = get_tcnn_model_and_losses(
        cnn_normalization_mean, cnn_normalization_std, ref, device)

    tcnn(img)

    score = 0
    for sl in losses:
        score += weight_tcnn * sl.loss

    score.backward()
    grad = img.grad.cpu()

    return score.cpu(), grad.flatten()
