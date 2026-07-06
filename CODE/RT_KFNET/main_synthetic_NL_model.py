# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:11:27 2025

@author: ricca
"""
#%%
import torch
import Simulations.config as config

from datetime import datetime
from Simulations.Synthetic_NL_model.parameters import Q_structure, R_structure,m,n,m1x_0,m2x_0, \
    f,h
from Simulations.Extended_sysmdl import SystemModel
from Simulations.utils import DataGen
from KNet.KalmanNet_nn import KalmanNetNN
from Pipelines.Pipeline_EKF import Pipeline_EKF
from Filters.EKF_test import EKFTest
from RobustKalmanPY.robust_kalman import RobustKalman

print("Pipeline Start")
################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

###################
###  Settings   ###
###################
args = config.general_settings()
### dataset parameters
# N is the number of sequences to be generated!
args.N_E = 1000 #length of training dataset (I kept this length there is no specificagtion in the paper)
args.N_CV = 100 #length of validation dataset (I kept this length there is no specificagtion in the paper)
args.N_T = 200 #length of test dataset (I kept this length there is no specificagtion in the paper)
args.T = 100 #input sequence length (see paper pag.10 section c)
args.T_test = 100 #input test sequence length (I kept this length there is no specificagtion in the paper)

### training parameters (I kept all these parameters I found them in the Lorenz attractor main file)
args.use_cuda = False # use GPU or not (True = use GPU)
args.n_steps =  200 #number of training steps (default: 1000)
args.n_batch = 30 #input batch size for training (default: 20)
args.lr = 1e-3 #learning rate (default: 1e-3)
args.wd = 1e-3 #weight decay (default: 1e-4)

if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

#offset = 0 # offset for the data
#chop = False # whether to chop data sequences into shorter sequences
path_results = 'KNet/'
DatafolderName = 'Simulations/Synthetic_NL_model/data' + '/'
#flag that will be used later on in the code
#switch = 'partial' # 'full' or 'partial' or 'estH'v 

# noise q and r
r2 = torch.tensor([0.1]) # [100, 10, 1, 0.1, 0.01]
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10) #vdb = 10*log(q2/r2)
q2 = torch.mul(v,r2) #see paper pag. 8 

Q = q2[0] * Q_structure #defining the transition matrix noise
R = r2[0] * R_structure #defining the output matrix noise

print("1/r2 [dB]: ", 10 * torch.log10(1/r2[0]))
print("1/q2 [dB]: ", 10 * torch.log10(1/q2[0]))

traj_resultName = ['traj_synNL_T100.pt']
dataFileName = ['data_synNL_T100.pt'] #used to load data below

#########################################
###  Generate and load data DT case   ###
#########################################

sys_model = SystemModel(f, Q, h, R, args.T, args.T_test, m, n)# parameters for GT
sys_model.InitSequence(m1x_0, m2x_0)# x0 and P0

print("Start Data Gen")
DataGen(args, sys_model, DatafolderName + dataFileName[0])
print("Data Load")
print(dataFileName[0])
[train_input,train_target, cv_input, cv_target, test_input, test_target,_,_,_] =  torch.load(DatafolderName + dataFileName[0], map_location=device)
print("trainset size:",train_target.size())
print("cvset size:",cv_target.size())
print("testset size:",test_target.size())
#%%
#####################
### Evaluate KNet ###
#####################
KNet_model = KalmanNetNN()
KNet_model.NNBuild(sys_model, args)
# ## Train Neural Network
KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KNet")
KNet_Pipeline.setssModel(sys_model)
KNet_Pipeline.setModel(KNet_model)
print("Number of trainable parameters for KNet:",sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
KNet_Pipeline.setTrainingParams(args) 

[MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results)
## Test Neural Network
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,Knet_out,RunTime] = KNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)
print("fine")
