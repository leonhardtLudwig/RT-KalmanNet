#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import torch
import Simulations.config as config
import matplotlib.pyplot as plt


from datetime import datetime
from Simulations.REKF_model.parameters import n, m, Q, R, f, h, m1x_0, m2x_0 #(Original model parameters)
#from Simulations.Synthetic_NL_model.parameters import n, m, Q_structure, R_structure, f, h, m1x_0, m2x_0 #synthetic_NL_model
from Simulations.Extended_sysmdl import SystemModel
from Simulations.utils import DataGen
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
args.N_E = 1 #length of training dataset (I kept this length there is no specificagtion in the paper)
args.N_CV = 1 #length of validation dataset (I kept this length there is no specificagtion in the paper)
args.N_T = 1 #length of test dataset (I kept this length there is no specificagtion in the paper)
args.T = 100 #input sequence length (see paper pag.10 section c)
args.T_test = 100 #input test sequence length (I kept this length there is no specificagtion in the paper)

### training parameters (I kept all these parameters I found them in the Lorenz attractor main file)
args.use_cuda = False # use GPU or not (True = use GPU)
args.n_steps =  200 #number of training steps (default: 1000)
args.n_batch = 30 #input batch size for training (default: 20)
args.lr = 1e-3 #learning rate (default: 1e-3)
args.wd = 1e-3 #weight decay (default: 1e-4)

#offset = 0 # offset for the data
#chop = False # whether to chop data sequences into shorter sequences
path_results = 'KNet/'
DatafolderName = 'Simulations/REKF_model/data' + '/'
#flag that will be used later on in the code
#switch = 'partial' # 'full' or 'partial' or 'estH'v 

traj_resultName = ['traj_REKF_T100.pt']
dataFileName = ['data_REKF_T100.pt'] #used to load data below
    
device = torch.device('cpu')

#########################################
###  Generate and load data DT case   ###
#########################################

sys_model = SystemModel(f, Q, h, R, args.T, args.T_test, m, n)# parameters for GT (original model)
#sys_model = SystemModel(f, Q_structure, h, R_structure, args.T, args.T_test, m, n)# parameters for GT (synthetic NL model)
sys_model.InitSequence(m1x_0, m2x_0)# x0 and P0

print("Start Data Gen")
DataGen(args, sys_model, DatafolderName + dataFileName[0])
print("Data Load")
print(dataFileName[0])
[train_input,train_target, _, _, _, _,_,_,_] =  torch.load(DatafolderName + dataFileName[0], map_location=device)
print("trainset input size:",train_input.size())
print("trainset target size:",train_target.size())

train_input = torch.squeeze(train_input,1) #(for original model)
#train_input = torch.squeeze(train_input) #(for synthetic NL model)
train_target = torch.squeeze(train_target)
print("trainset input size:",train_input.size())
print("trainset target size:",train_target.size())

#print(train_input)
#print(train_target)
#print(sys_model.m1x_0)
#%% Test plot of data

plt.figure()

plt.subplot(211)
plt.plot(torch.transpose(train_target,0, 1))#exchange 0 and 1 dimension
plt.title("X_n")

plt.subplot(212)
plt.plot(torch.transpose(train_input,0, 1))
plt.title("Y_n")

# %% test of REKF on a test sample
sys_model.m1x_0 = torch.zeros(m,1)
# The test shows something is broken .. must resech tomorrow
REKF = RobustKalman(sys_model, train_input, 1e-3, True, False)

[Xrekf, _] = REKF.fnREKF()
#print(Xrekf.shape)
#print(Xrekf)


#%% Test plot of data

plt.figure()

plt.subplot(211)
plt.plot(torch.transpose(train_target,0, 1),color ='red' , label='train target')
plt.plot(torch.transpose(Xrekf.detach(), 0, 1),color = 'blue' , label = 'estimated values')
plt.title("X_n")

plt.subplot(212)
plt.plot(torch.transpose(train_input,0, 1))
plt.title("Y_n")


#%% ##################### FROM HERE WE TEST THE IMPLEMENTATION OF RT-KalmanNet #########################
import torch.optim as optim
import torch.nn as nn

sys_model.m1x_0 = torch.zeros(m,1)
RT_KalmanNet = RobustKalman(sys_model, train_input,1e-3,True,True)
model = RT_KalmanNet

lr = 0.01 #learning rate
epochs = 200 #defining the number of epochs

optimizer = optim.Adam(RT_KalmanNet.nn.parameters(), lr=lr)
criterion = nn.MSELoss()  # Minimizza l'errore della stima dello stato

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # Generate data
    DataGen(args, sys_model, DatafolderName + dataFileName[0])
    [train_input, train_target, _, _, _, _, _, _, _] = torch.load(DatafolderName + dataFileName[0], map_location=device)
    #print("Data generated")

    # Preprocess data
    train_input = torch.squeeze(train_input, 1)  # Remove redundant dimensions
    train_target = torch.squeeze(train_target)
    #print("Data squeezed")

    # Zeroing gradients
    optimizer.zero_grad()

    #change the input
    RT_KalmanNet.y = train_input

    # Forward pass
    [Xrekf, _] = RT_KalmanNet.fnREKF()
    Xrekf = Xrekf[:, 1:]
    #print("Passing through Kalman Filter\n")

    # Check gradient flow
    #print("requires_grad:", Xrekf.requires_grad)

    # Compute loss
    loss = criterion(Xrekf, train_target)
    print(f"Loss: {loss.item()}")

    # Backward pass
    loss.backward(retain_graph=True )

    # Optimization step
    optimizer.step()
    #print("Optimizer step completed")

    #print("Epoch completed\n")

print("Training finished")
#%% ##################### TEST OF RT-KalmanNet #########################

DataGen(args, sys_model, DatafolderName + dataFileName[0])
[train_input, train_target, _, _, _, _, _, _, _] = torch.load(DatafolderName + dataFileName[0], map_location=device)
# Preprocess data
train_input = torch.squeeze(train_input, 1)  # Remove redundant dimensions
train_target = torch.squeeze(train_target)

#change the input
RT_KalmanNet.y = train_input

# Forward pass
[Xrekf, _] = RT_KalmanNet.fnREKF()
Xrekf = Xrekf[:, 1:]

#output = Xrekf.detach().numpy()

plt.figure()

plt.subplot(211)
plt.plot(torch.transpose(train_target,0, 1),color ='red' , label='train target')
plt.plot(torch.transpose(Xrekf.detach(), 0, 1),color = 'blue' , label = 'estimated values')
plt.title("X_n")

