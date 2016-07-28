# coding: utf-8
import shutil
get_ipython().magic(u'colors lightbg')
import numpy as npy
import os 
import subprocess
import matplotlib.plt
import matplotlib.pyplot as plt
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'cd ..')
1-0.005
(1-0.005)/0.0005
(1-0.05)/0.005
(1-0.01)/0.005
import sys
npy.linspace(0.01,1,198)
npy.linspace(0.01,1,197)
npy.linspace()?
get_ipython().magic(u'pinfo npy.linspace')
npy.linspace(0.01,1,190)
npy.linspace(0.01,1,196)
npy.linspace(0.01,1,195)
npy.linspace(0.01,1,199)
l_space=npy.linspace(0.01,1,199)
l_space
for learning_rate in l_space:
    print learning_rate
    
(1-0.01)/0.05
npy.linspace(0.01,1,20)
npy.linspace(0,1,20)
npy.linspace(0,1,21)
npy.linspace(0,1,21).remove(0.)
npy.linspace(0,1,21).delete(0.)
a=npy.linspace(0,1,21)
a
a.delete(1)
npy.delete
get_ipython().magic(u'pinfo npy.delete')
a
npy.delete(a,0)
npy.delete(a,0)
npy.delete(a,0.)
a=npy.linspace(0,1,21)
a
npy.delete(a,0.06)
l_space_2=npy.linspace(0,1,21)
l_space_2
npy.delete(l_space_2,0)
for learning_rate in l_space_2:
    command="scripts/conv_back_prop/learn_trans_po.py %f 0" %learning_rate
    print command.split()
    
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd data/BP_Experiments/')
os.mkdir("Linear_Decay")
get_ipython().magic(u'ls ')
get_ipython().magic(u'pwd ')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'ls ')
8*0.05/5
1000()
l_space_2
def movethings(learn_r):
    shutil.move("estimated_transition.txt","data/BP_Experiments/Linear_Decay/estimated_{0}".format(learn_r))
    
npy.delete(l_space_2,0.)
l_space_2=npy.delete(l_space_2,0.)
l_space_2
for l in l_space_2:
    command = "scripts/conv_back_prop/learn_trans_po_penalty.py {0} 0".format(l)
    subprocess.call(command.split(),shell=False)
    movethings(l)
    
get_ipython().magic(u'ls ')
get_ipython().magic(u'ls ')
print l
get_ipython().magic(u'ls ')
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd data/BP_Experiments/')
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd Linear_Decay/')
get_ipython().magic(u'ls ')
get_ipython().magic(u'mv estimated_0.05 estimated_0.05.txt')
get_ipython().magic(u'ls ')
get_ipython().magic(u'mv estimated_0.1 estimated_0.1.txt')
get_ipython().magic(u'mv estimated_0.15 estimated_0.15.txt')
get_ipython().magic(u'mv estimated_0.2 estimated_0.2.txt')
get_ipython().magic(u'ls ')
subl estimated_0.2.txt
def movethings(learn_r):
    shutil.move("estimated_transition.txt","data/BP_Experiments/Linear_Decay/estimated_{0}.txt".format(learn_r))
    
get_ipython().magic(u'ls ')
get_ipython().magic(u'ls ')
cross_entropy_error=0.1
lsq_error=1.2
with file('error.txt','w') as outfile:
    	outfile.write('#Cross Entropy Error:')
    	outfile.write(cross_entropy_error)
    	outfile.write('#Least Square Error:')
    	outfile.write(lsq_error)
    
get_ipython().magic(u'ls ')
get_ipython().magic(u'pinfo outfile.write')
def movethings(learn_r,trial):
    shutil.move("estimated_transition.txt","data/BP_Experiments/Linear_Decay/LR_{0}/estimated_trans_{1}.txt".format(learn_r,trial))
    
get_ipython().magic(u'ls ')
get_ipython().magic(u'rm error.txt')
get_ipython().magic(u'ls ')
for l in l_space_2:
    os.mkdir("LR_{0}".format(l))
    
get_ipython().magic(u'ls ')
for l in l_space_2:
    for i in range(1,6):
        command="scripts/conv_back_prop/learn_trans_po_penalty.py {0} 0".format(l)
        subprocess.call(command.split(),shell=False)
        movethings(l,i)
        
get_ipython().magic(u'cd ..')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'cd ..')
for l in l_space_2:
    for i in range(1,6):
        command="scripts/conv_back_prop/learn_trans_po_penalty.py {0} 0".format(l)
        subprocess.call(command.split(),shell=False)
        movethings(l,i)
        
for l in l_space_2:
    for i in range(1,6):
        command="scripts/conv_back_prop/learn_trans_po_penalty.py {0} 0".format(l)
        subprocess.call(command.split(),shell=False)
        movethings(l,i)
        
l_space_2
lspace = copy.deepcopy(l_space_2)
import copy
lspace = copy.deepcopy(l_space_2)
npy.delete(lspace,0)
npy.delete(lspace,0)
lspace=npy.delete(lspace,0)
lspace
lspace=npy.delete(lspace,0)
lspace
lspace
for l in l_space_2:
        for i in range(1,6):
                command="scripts/conv_back_prop/learn_trans_po_penalty.py {0} 0".format(l)
                subprocess.call(command.split(),shell=False)
                movethings(l,i)
        
for l in l_space_2:
        for i in range(1,6):
                command="scripts/conv_back_prop/learn_trans_po_penalty.py {0} 0".format(l)
                subprocess.call(command.split(),shell=False)
                movethings(l,i)
        
get_ipython().magic(u'ls ')
for l in lspace:
        for i in range(1,6):
                command="scripts/conv_back_prop/learn_trans_po_penalty.py {0} 0".format(l)
                subprocess.call(command.split(),shell=False)
                movethings(l,i)
        
for l in lspace:
        for i in range(1,6):
                command="scripts/conv_back_prop/learn_trans_po_penalty.py {0} 0".format(l)
                subprocess.call(command.split(),shell=False)
                movethings(l,i)
        
x = npy.ones((20,20))
x
x[1,2]
x[-3,6]
x[-19,45]
x[-19,0]
x[-20,0]
x[-21,0]
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd data/BP_Experiments/')
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd Linear_Decay/')
get_ipython().magic(u'ls ')
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'ls ')
os.mkdir("Old")
get_ipython().magic(u'ls ')
get_ipython().magic(u'ls ')
movethings()
movethings()?
get_ipython().magic(u'pinfo movethings')
get_ipython().magic(u'pinfo movethings.def')
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd ..')
lspace
l_space_2
get_ipython().magic(u'ls ')
os.path.isfile("create_rewards/reward_1/reward_1.txt")
os.path.isfile("create_rewards/reward_1/reward_122.txt")
for l in l_space_2:
    for i in range(1,6):
        command="scripts/conv_back_prop/learn_trans_po_penalty.po {0} 0".format(l)
        subprocess.call(command.split(),shell=False)
        if os.path.isfile("estimated_transition.txt"):
            movethings(l,i)
            
for l in l_space_2:
    for i in range(1,6):
        command="scripts/conv_back_prop/learn_trans_po_penalty.py {0} 0".format(l)
        subprocess.call(command.split(),shell=False)
        if os.path.isfile("estimated_transition.txt"):
            movethings(l,i)
            
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd ..')
for l in l_space_2:
    for i in range(1,6):
        command="scripts/conv_back_prop/learn_trans_po_penalty.py {0} 0".format(l)
        subprocess.call(command.split(),shell=False)
        if os.path.isfile("estimated_transition.txt"):
            movethings(l,i)
            
get_ipython().magic(u'ls ')
command = "scripts/conv_back_prop/learn_trans_po_penalty.py {0} 0".format(l)
l
l=0.2
command = "scripts/conv_back_prop/learn_trans_po_penalty.py {0} 0".format(l)
get_ipython().magic(u'ls ')
subprocess.call(command.split(),shell=False)
get_ipython().magic(u'ls ')
movethings(l,5)
get_ipython().magic(u'ls ')
l = 0.25
command = "scripts/conv_back_prop/learn_trans_po_penalty.py {0} 0".format(l)
subprocess.call(command.split(),shell=False)
get_ipython().magic(u'ls ')
movethings(l,2)
command = "scripts/conv_back_prop/learn_trans_po_penalty.py {0} 0".format(l)
subprocess.call(command.split(),shell=False)
movethings(l,3)
l=0.45
command = "scripts/conv_back_prop/learn_trans_po_penalty.py {0} 0".format(l)
subprocess.call(command.split(),shell=False)
get_ipython().magic(u'ls ')
movethings(l,3)
l=0.7
command = "scripts/conv_back_prop/learn_trans_po_penalty.py {0} 0".format(l)
subprocess.call(command.split(),shell=False)
movethings(l,3)
l=0.75
command = "scripts/conv_back_prop/learn_trans_po_penalty.py {0} 0".format(l)
subprocess.call(command.split(),shell=False)
get_ipython().magic(u'ls ')
movethings(l,1)
subprocess.call(command.split(),shell=False)
movethings(l,4)
l=0.8
movethings(l,1)
command = "scripts/conv_back_prop/learn_trans_po_penalty.py {0} 0".format(l)
subprocess.call(command.split(),shell=False)
movethings(l,1)
subprocess.call(command.split(),shell=False)
subprocess.call(command.split(),shell=False)
movethings(l,2)
l=0.9
command = "scripts/conv_back_prop/learn_trans_po_penalty.py {0} 0".format(l)
subprocess.call(command.split(),shell=False)
movethings(l,2)
subprocess.call(command.split(),shell=False)
movethings(l,5)
l=0.95
command = "scripts/conv_back_prop/learn_trans_po_penalty.py {0} 0".format(l)
subprocess.call(command.split(),shell=False)
movethings(l,1)
get_ipython().magic(u'ls ')
orig = npy.loadtxt("actual_transition.txt")
orig=orig.reshape((8,3,3))
orig
l_space_2
l_space_2.shape()
npy.shape(l_shape_2)
npy.shape(l_space_2)
errors = npy.zeros((20,5))
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd data/BP_Experiments/Linear_Decay/')
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd AR_0/')
get_ipython().magic(u'ls ')
lsq_errors=npy.zeros((20,5))
ce_errors=npy.zeros((20,5))
for i in range(0,20):
    for j in range(1,6):
        trans=npy.loadtxt("LR_{0}/estimated_transition.txt".format(l_space_2[i]))
        trans=trans.reshape((8,3,3))
        ce_errors[i,j-1]=-npy.sum(orig[:,:,:]*npy.log(trans[:,:,:]))
        lsq_errors[i,j-1]=npy.sum((orig[:,:,:]-trans[:,:,:])**2)
        
get_ipython().magic(u'ls ')
for i in range(0,20):
    for j in range(1,6):
        trans=npy.loadtxt("LR_{0}/estimated_transition_{1}.txt".format(l_space_2[i],j))
        trans=trans.reshape((8,3,3))
        ce_errors[i,j-1]=-npy.sum(orig[:,:,:]*npy.log(trans[:,:,:]))
        lsq_errors[i,j-1]=npy.sum((orig[:,:,:]-trans[:,:,:])**2)
        
for i in range(0,20):
    for j in range(1,6):
        trans=npy.loadtxt("LR_{0}/estimated_trans_{1}.txt".format(l_space_2[i],j))
        trans=trans.reshape((8,3,3))
        ce_errors[i,j-1]=-npy.sum(orig[:,:,:]*npy.log(trans[:,:,:]))
        lsq_errors[i,j-1]=npy.sum((orig[:,:,:]-trans[:,:,:])**2)
        
ce_errors
lsq_errors
get_ipython().magic(u'pinfo npy.mean')
npy.mean(lsq_errors,axis=1)
get_ipython().magic(u'save EXPT.py 1-228')
