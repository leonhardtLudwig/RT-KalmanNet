#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

A parameters script of the non-linear model from Zorzis REKF paper - USED FOR DEBUGGING AND TRANSLATION

"""

import torch
import math
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

#########################
### Design Parameters ###
#########################
m = 2 #dimension of the state (see paper pag.10 formulae 17a 17b)
n = 1 #dimension of the output
#variance = 0
#first order and second order moment of the initial condition
m2x_0 = 1e-3*torch.eye(m)
m1x_0 = torch.zeros(m,1)+torch.sqrt(m2x_0)@torch.randn(m,1)

a1 = torch.tensor([0.1]);
##############################################################
### State evolution function f for Zorzis non-linear model:###
##############################################################

def f(x):
    # Assume tensor input
    #print("x=",x.requires_grad)
    x = torch.squeeze(x) # in case we have a dimension too much
    x_out_1 = x[0]/10 + x[1] + torch.cos(x[1]/10) - 1
    x_out_2 = (49*x[1])/50
    
    x_out = torch.tensor([[x_out_1],[x_out_2]])
    return x_out


##########################################################
### Observation function f for Zorzis non-linear model:###
##########################################################

def h(x):
    # Assume tensor input
    x = torch.squeeze(x)
    x_out = - x[0]**2 + x[0] + x[1]**2 - x[1]
    return torch.transpose(torch.tensor([[x_out]]),0, 1)

###############################################
### process noise Q and observation noise R ###
###############################################

B = torch.cholesky(torch.tensor([[1.9608, 0.0195],[0.0195, 1.9605]]))
B = torch.cat((B, torch.zeros(m,n)),1)
D = torch.tensor([[1]])
D = torch.cat((torch.zeros(n,m),D),1)

Q = B @ torch.transpose(B, 0, 1)
R = D @ torch.transpose(D, 0, 1)

