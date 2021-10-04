import torch
import pickle

from models import *
from opt import *
from utils import *
from mse import *
from ssim import *
from one_layer import *

import time
import warnings
warnings.filterwarnings("ignore")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nc = 3
imsize = 256




"""

AUTOMATIC PER MODEL COMPARISON

models: model_defence, model_attack
images: ref, input_img(Gpu tensor)
    gradients and loss computed based on images

control param: gd (1,2,3,4)(str)
1: model attack: Gram / model defence: SSIM / finding best image wrt Gram
2: model attack: Gram / model defence: SSIM / finding worst image wrt Gram
3: model attack: SSIM / model defence: Gram / finding best image wrt SSIM
4: model attack: SSIM / model defence: Gram / finding worst image wrt SSIM

name param: full_name (method name,gd,suffix)(str)

"""
method = 'Gram-ssim'
distortions = {'noise':'8','blur':'6','jpeg':'25','gamma':'2'}
#distortions = {'blur':'1.5'}
#loss = []
def MAD(model_defence, model_attack, weight_attack, weight_defence,ref = None, input_img = None, full_name = None, gd = None,iterations = None):

    # change by hand
    #iterations = 1201
    start = time.time()
    iters = 50
    prev_loss2 = 0
    count = 0
    lamda = 0.12
    start = time.time()
    lamda2 = -1
    comp = 0
    for i in range(iterations):

        if i%iters == 0:
            end = time.time()
            print('time per',iters,'iterations',end-start)
            start = end
            print('iteration',i)
            print('lamda',lamda)    

        """
        keep model
        """
        #gpu_tracker.track()
        # model which needed keep same
        loss1, g1 = model_defence(input_img,ref,weight_defence)
        if i ==0:
            m0 = loss1
        else:
            pass
        if i%iters == 0:
            print('loss1',loss1)
            print('g1',g1.max(),g1.min(),torch.mean(torch.abs(g1)))
    
        """
        max/min model
        """
        if i > 0:
            prev_loss2 = loss2
        #gpu_tracker.track()
        # min/max this model

        loss2, g2 = model_attack(input_img.detach(), ref,weight_attack)
        #loss.append(loss2.detach())
        if i%iters == 0:
        # print('\n')
            print('loss2',loss2)
            print('g2',g2.max(),g2.min(),torch.mean(torch.abs(g2)))
  
        """
        adjust learning rate
        decrease lamda 
        """
        if i == 0:
            fix = lamda*torch.norm(g2) 
            #fix = torch.load('temp_fix.pt')*0.1
        lamda = fix/torch.norm(g2) 
        if torch.abs(prev_loss2-loss2) < 1e-3*torch.abs(loss2):
            lamda = lamda*0.8
            
        if comp > 0.5:
            #full_name = 'attention!'+'_'+full_name
            imshow(torch.clamp(y,0,1),None,'attention'+full_name)
            fix = fix*0.8

        lamda = step_size(lamda0 = lamda, opt = 200, rate1 = 1, rate2 = 0.999, iteration = i)

        """
        early stopping
        """
    
        if torch.abs(loss2-prev_loss2) < 1e-4*loss2:
            count += 1
        else:
            count = 0
        #if count > 100:
        #    print('early stopping!!')
        #    imshow(torch.clamp(input_img,0,1),None,full_name)
        #    break
        """
        using MAD
        """
        mkeep = model_defence
        #gpu_tracker.track()
        # change mkeep and xxx_opt in opt.py search_grad function
        y, comp, lamda2  = search_grad(ref,
                                    g = g2, gkeep = g1,
                                    img = input_img,
                                    mkeep = mkeep,
                                    weight = weight_defence,
                                    init_loss = m0, 
                                    lamda = lamda,
                                    lamda2 = lamda2,
                                    gd = gd)
        """
        print information
        """
        if i %iters == 0:
            #print('\n')
            print('cumulate comp:', comp)
            #plt.figure()
        if i == iterations-1:
            imshow(torch.clamp(input_img,0,1),None,full_name)
            #torch.save(input_img,'temp.pt')
            #torch.save(fix,'temp_fix.pt')
            #print('\n\n')
        # if comp > 2:
            # #plt.figure()
            # #imshow(torch.clamp(y,0,1),None,'break!'+ '_' + str(int(loss2))+ '_' +full_name)
            # lamda = lamda*0.5
            # print('too big step size, change lamda!!')
            # torch.save(input_img,'temp.pt')  
            # torch.save(fix,'temp_fix.pt')
            # break
        if comp < 0.5:
            input_img = y
        imshow(torch.clamp(input_img,0,1),None,full_name)

if __name__ == '__main__':
    
    weight_gram = 1e3
    weight_mse = 1e3
    weight_ssim = 1e3

    aim_grad = 0.015
    
     ####################################################

    
    #method = 'Gram-Ssim'
    #level = '6'
    #Type = 'blur'
    
    for dis in distortions.keys():
        
        Type = dis
        level = distortions[dis]
    
        for im_num in range(1,35):
            im_name = str(im_num)
        
            ref_img = image_loader("./data/texture/" + im_name + ".jpg")
            ref_img = ref_img[:,:,0:256,0:256]
            #print(ref_img.shape)
            #imgn = image_loader("./data/texture/jpeg_10_radish.jpg")
        
            seed = 999
            torch.manual_seed(seed)
        
        
            if Type == 'noise' :
                imgn = gaussian_noise(float(level),ref_img)
            elif Type == 'blur' :
                imgn = gaussian_blur(float(level),ref_img,im_name)
            elif Type == 'gamma':
                imgn = gamma(float(level),ref_img,im_name)
            elif Type == 'jpeg':
                imgn = jpeg(int(level),ref_img,im_name)
            else:
                pass
        
            imgn.data.clamp_(0,1)
            input_img = imgn
            #input_img = torch.load('temp.pt')
            ref = ref_img
        



            """

            set gradients
    
            """
            _,g = model_gram(imgn,ref,weight_gram)
            temp_grad = torch.mean(torch.abs(g))
            adjust = aim_grad/temp_grad
            weight_gram = adjust*weight_gram

            _,g = ssim(imgn,ref,weight_ssim)
            temp_grad = torch.mean(torch.abs(g))
            adjust = aim_grad/temp_grad
            weight_ssim = adjust*weight_ssim

            _,g = mse(imgn,ref,weight_mse)
            temp_grad = torch.mean(torch.abs(g))
            adjust = aim_grad/temp_grad
            weight_mse = adjust*weight_mse


            ##gd : gradient direction
            for gd in range(1,5):
                gd = str(gd)
            
                if gd == '1' or gd == '2':
    
                
                    suffix = Type+'_'+level+'_'+gd
                    full_name = method+'_'+im_name+'_'+suffix
                
                    model_attack = model_gram
                    weight_attack = weight_gram

                    model_defence = ssim    #keep same
                    weight_defence = weight_ssim
                
                    MAD(model_defence, model_attack,weight_attack, weight_defence, ref , imgn, full_name, gd, 800)
 
            
                else:
                    suffix = Type+'_'+level+'_'+gd
                    full_name = method+'_'+im_name+'_'+suffix
                
                    model_attack = ssim
                    weight_attack = weight_ssim

                    model_defence = model_gram   #keep same
                    weight_defence = weight_gram

                    MAD(model_defence, model_attack,weight_attack, weight_defence,ref , imgn, full_name, gd, 400)

            #with open('loss'+ gd +'.pkl', 'wb') as f:
            #    pickle.dump(loss, f)
            #loss = []
