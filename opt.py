import torch
import numpy as np
from models import *
import time

from utils import *
from mse import *
from ssim import *
from one_layer import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nc = 3
imszie = 256
acount = 0
count = 0

"""
Opt file include some algorithms:  
Try to keep the score for one metric fixed (model defence)  
while finding a better score for another metric (model attack)
"""

"""
Adam
"""
def Adam(m0, xm, ref, mkeep_opt,weight):
    
    global acount,count
    acount = acount + 1
    start = time.time()
    xm = xm.reshape(1,nc,imsize,imsize)
    lr = 1e-5  # vgg+gram 2e-5
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8
    theta_0 = 0
    m_t = 0 
    v_t = 0 
    t = 0
    var = 1
    
    while var == 1:
        t += 1
        #print('t',t)
        #if t > 10: 
        #    lr = lr*0.9
        comp, g_t = mkeep_opt(m0,xm,ref,weight)
        if comp < 5e-5:    #vgg+gram 1e-6mm0
            break
        if t > 60:
            break
        m_t = beta_1*m_t + (1-beta_1)*g_t     # consider 90% of previous, and 10% of current
        v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t) # 99.9% of previous (square grad), and 1% of current
        m_cap = m_t/(1-(beta_1**t))      #calculates the bias-corrected estimates
        v_cap = v_t/(1-(beta_2**t))      #calculates the bias-corrected estimates
        
        #xm_prev = xm
        xm = xm - (lr*m_cap)/(torch.sqrt(v_cap)+epsilon)
        
    end = time.time()
    if count > 50:
        print('Adam fre',acount/count,'\n')
        print('Adam:',end-start,'s\n')
        acount = 0
        count = 0
    return comp, xm 

"""
g: gradient at xm

"""
def show(m0,model,weight,xm,ref):
    
    xm = xm.flatten()
    range = 0.1
    N = 1000
    vsearch = np.linspace(-range,range,N)
    m, g = model(xm.detach(),ref,weight)
    comp = torch.abs(m0-m)
    for i,v in enumerate(vsearch):
        
        x = xm + v*g 
        m, _ = model(x,ref,weight)
        if i%20 == 0:
            print(i,torch.abs(m0-m),m0,m,v)
        if torch.abs(m0-m) < comp:
            comp = torch.abs(m0-m)
    print('\n',comp,'\n')

# Bad implmentation
# Brute force all situations (maybe not all)
def bisection(f,weight, lower, upper, g, ref, init_loss, xm):
    start = time.time()
    global count
    count = count + 1
    xm = xm.reshape(1,nc,imsize,imsize)
    obj = init_loss
    var = 1
    a = lower
    b = upper
    m = (a+b)/2
#    print('\n\n\n')
#    print('range adjustment')
#    print(f(xm+a*g,ref)[0],f(xm+m*g,ref)[0],f(xm+b*g,ref)[0])
#    print('\n\n\n')
    flag = 0
    m1, _ = f((xm+a*g),ref,weight)
    m2, _ = f((xm+m*g),ref,weight)
    m3, _ = f((xm+b*g),ref,weight)
    tol = 20
    x = 0.01
    
    while var == 1:
       # if (f(xm+b*g,ref)[0]-obj) <= (f(xm+m*g,ref)[0]-obj) or (f(xm+m*g,ref)[0]-obj) <= (f(xm+a*g,ref)[0]-obj):
        if (m3-m2) <= 0  or (m2-m1) <= 0:
            a = m
            m = (a+b)/2
            m1, _ = f((xm+a*g),ref,weight)
            m2, _ = f((xm+m*g),ref,weight)
            #m3, _ = f(xm+b*g,ref)
            if flag > tol :
                #print('!!!!!!!!!!!')
                break
            else:
                flag += 1
                continue

        if (m1-obj) > 0 and (m3-obj) > 0: 
            a = a-x
            m = (a+b)/2
            m1, _ = f((xm+a*g),ref,weight)
            m2, _ = f((xm+m*g),ref,weight)

            if flag > tol :

                break
            else:
                flag += 1
                continue
        elif (m1-obj) < 0 and (m3-obj) < 0:
            b = b+x
            m = (a+b)/2

            m2, _ = f((xm+m*g),ref,weight)
            m3, _ = f((xm+b*g),ref,weight)
            if flag > tol :

                break
            else:
                flag += 1
                continue
        else:
            pass
            
    
        
        if (m3-obj) < 0 or (m1-obj) > 0:
            continue
        
        
        if (m1-obj)*(m2-obj) <= 0:
            b = m
            m = (a+b)/2
            m2, _ = f((xm+m*g),ref,weight)
            m3, _ = f((xm+b*g),ref,weight)
        elif (m2-obj)*(m3-obj) <= 0:
            a = m
            m = (a+b)/2
            m1, _ = f((xm+a*g),ref,weight)
            m2, _ = f((xm+m*g),ref,weight)
        elif flag > tol :

            break
        else:
            pass
         
            
        if b-a < 1e-5:
             break

    comp = m2-obj
    end = time.time()
    if count%50 == 0:
        
        print('Bisection:',end-start,'s\n')
       
    return comp, (xm + m*g)

# A solution to convex model
def prof_wang(mkeep,xm,ref,gkeep,init_loss,lamda2):

    mb, _ = mkeep(xm ,ref)
    
    temp_im = xm + lamda2*gkeep
    mt, _ = mkeep(temp_im ,ref)
    lamda2 = (lamda2*(init_loss - mb)/(mt - mb)).detach()
    xk = xm + lamda2*gkeep
    mk, _ = mkeep(xk ,ref)
    comp = mk-init_loss
    y = xk.reshape(1,nc,imsize,imsize)
    
    return comp,y,lamda2




def search_grad(ref, g, gkeep, img = None, mkeep = None, weight = None ,init_loss = None, lamda = None, lamda2 = None,gd = None):
    
  
    #####   project
    gm = g - torch.mul(torch.div(torch.dot(g,gkeep), torch.dot(gkeep,gkeep)) ,gkeep)
     
    ################# 
    if int(gd) == 1 or int(gd) == 3:
        xm = torch.sub(img.flatten(), torch.mul(lamda, gm))
    else:
        xm = torch.add(img.flatten(), torch.mul(lamda, gm))
    
    ##############################################
    xm = torch.clamp(xm, 0, 1)
    dim = torch.clamp((xm-ref.flatten()), -1, 1)
    xm = ref.flatten() + dim
    ################################################
    
    """
    
    bisection
    
    """
    #gn = mkeep(xm, ref)[1]
    #comp,y,lamda2 = prof_wang(mkeep,xm.detach(),ref,gn,init_loss,lamda2)
    gn = mkeep(xm, ref,weight)[1].reshape(1,nc,imsize,imsize)
    comp, y = bisection(mkeep,weight, -0.2, 0.2, gn, ref, init_loss, xm.detach())
        
    """
    Adam
    """
    if int(gd) == 1 or int(gd) == 2:
        opt = ssim_opt
    else:
        opt = model_gram_opt
    if torch.abs(comp) > 0.01:
        comp, y = Adam(init_loss.detach(),xm,ref,opt,weight)

    return y, comp, lamda2
