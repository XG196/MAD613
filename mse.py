import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 256
nc = 3




def mse(img, ref,weight_mse):
    img = img.reshape(1, nc, imsize, imsize)

    #img = img.to(device)
    #ref = ref.to(device)

    img.requires_grad_()

    N = nc * imsize * imsize
    loss = weight_mse * ((img - ref) ** 2).sum() / (N)
    loss.backward()
    #
    #    del ref
    #    torch.cuda.empty_cache()
    return loss, img.grad.flatten()


def mse_opt(m0, temp, ref,weight_mse):
    temp = temp.reshape(1, nc, imsize, imsize)

    #m0 = m0.to(device)
    #temp = temp.to(device)
    #ref = ref.to(device)

    temp.requires_grad_()

    N = nc * imsize * imsize
    loss_mse = weight_mse * ((temp - ref) ** 2).sum() / (N)
    comp = (m0 - loss_mse) ** 2
    #print('comp',comp,m0-loss_mse,m0,loss_mse)
    comp.backward()

    return comp, temp.grad
