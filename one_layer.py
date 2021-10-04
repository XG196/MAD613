import torch
import torch.nn as nn

from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_feature_maps = 128
scales = [3, 5, 7 ,11, 15, 23, 37, 55]
weight_onelayer = 8e10
imsize = 256
nc = 3


class Onelayer_Net(nn.Module):

    def __init__(self, scales, n_feature_maps):
        super(Onelayer_Net, self).__init__()

        self.multiple_scales = [nn.Conv2d(nc, n_feature_maps, filter_size, 1, filter_size // 2).to(device)
                                for filter_size in scales]

        self.nonliners = [nn.ReLU().to(device) for filter_size in scales]

    def forward(self, x):
        out = [conv(x) for conv in self.multiple_scales]
        out = [f(out[i]) for i, f in enumerate(self.nonliners)]
        out = torch.cat(out, 1)

        return out


def get_onelayer_model_and_losses(normalization_mean, normalization_std,
                                  style_img,
                                  device):
    net = Onelayer_Net(scales, n_feature_maps)

    style_img = style_img.to(device)

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    model = nn.Sequential(normalization)
    model.add_module('onelayer', net)
    style_losses = []

    target_feature = model(style_img).detach()

    style_loss = StyleLoss(target_feature)
    model.add_module('style_loss', style_loss)

    style_losses.append(style_loss)

    del style_img
    torch.cuda.empty_cache()


    return model, style_losses

# called when gradients are not needed
def one_layer_forward(img, ref):
    #start = time.time()
    img = img.reshape(1, nc, imsize, imsize)
    img = img.to(device)
    ref = ref.to(device)

    onelayer, style_losses = get_onelayer_model_and_losses(
        cnn_normalization_mean, cnn_normalization_std, ref, device)




    with torch.no_grad():
        onelayer(img)




    style_score = 0
    for sl in style_losses:
        style_score += weight_onelayer * sl.loss

    grad = 0


    #del ref, img, onelayer

    #torch.cuda.empty_cache()


    return style_score.cpu(), grad

# called when gradients are needed
def one_layer(img, ref):

    img = img.reshape(1, nc, imsize, imsize)
    img = img.to(device)
    ref = ref.to(device)

    img.requires_grad_()

    onelayer, style_losses = get_onelayer_model_and_losses(
        cnn_normalization_mean, cnn_normalization_std, ref, device)

    onelayer(img)

    style_score = 0
    for sl in style_losses:
        style_score += weight_onelayer * sl.loss
    style_score.backward()
    grad = img.grad.cpu()

    #del ref, img, onelayer
    #torch.cuda.empty_cache()

    return style_score.cpu(), grad.flatten()

# called when in Adam
def one_layer_opt(m0, img, ref):

    img = img.reshape(1, nc, imsize, imsize)
    img = img.to(device)
    ref = ref.to(device)
    m0 = m0.to(device)

    img.requires_grad_()

    onelayer, style_losses = get_onelayer_model_and_losses(
        cnn_normalization_mean, cnn_normalization_std, ref, device)

    onelayer(img)



    style_score = 0
    for sl in style_losses:
        style_score += weight_onelayer * sl.loss

    comp = (m0 - style_score) ** 2

    comp.backward()
    grad = img.grad.cpu()



    #del ref, img, onelayer
    #torch.cuda.empty_cache()

    return comp.cpu(), grad