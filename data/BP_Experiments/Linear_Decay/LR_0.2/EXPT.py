# coding: utf-8
get_ipython().magic(u'colors lightbg')
import shutil
import os
import numpy as npy
import subprocess
get_ipython().magic(u'pinfo command')
dummy="hello ffdfd"
dummy.split()
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd data/BP_Experiments/Linear_Decay/AR_0/')
lsq_errors = npy.loadtxt("LSQ_Errors.txt")
ce_errors = npy.loadtxt("CE_Errors.txt")
ce
ce_errors
npy.min(npy.mean(ce_errors,axis=1))
(npy.mean(ce_errors,axis=1))
npy.mean(lsq_errors,axis=1)
npy.min(npy.mean(lsq_errors,axis=1))
npy.argmin(npy.mean(lsq_errors,axis=1))
x = range(1,20)
x
npy.shape(x)
lspace = npy.linspace(0,1,21)
lspace
npy.shape(lspace)
npy.delete(lspace,0)
lspace=npy.delete(lspace,0)
import matplotlib.pyplot as plt
get_ipython().magic(u'pinfo plt.line')
get_ipython().magic(u'pinfo pyplot')
plt.plot(lsq_errors,lspace)
plt.show()
plt.plot(lspace,lsq_errors))
plt.plot(lspace,lsq_errors)
plt.show()
plt.plot(lspace,npy.mean(lsq_errors,axis=1))
plt.show()
plt.plot(lspace,npy.mean(ce_errors,axis=1))
plt.show()
plt.plot(lspace,npy.mean(lsq_errors,axis=1))
plt.show()
plt.plot(lspace,npy.mean(lsq_errors,axis=1)*npy.mean(ce_errors,axis=1))
plt.show()
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'ls ')
get_ipython().magic(u'mkdir LR_0.2')
get_ipython().magic(u'ls ')
0.2/5
0.2 - 0.04*1000
0.2/5/8
0.2 - 0.005*1000
0.05*8/5
0.08/1000
0.05-0.08
aspace = npy.linspace(0.2,0.05,10)
aspace
aspace = npy.linspace(0.2,0.05,11)
aspace
0.2-1000*0.2/1000
0.2-1000*0.1/1000
0.2-0.15
0.15/1000
get_ipython().magic(u'cd ..')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'cd ..')
x = npy.loadtxt("estimated_transition.txt")
get_ipython().magic(u'ls ')
get_ipython().magic(u'ls ')
aspace
0.2-0.185
npy.delete(aspace,0)
aspace=npy.delete(aspace,0)
get_ipython().magic(u'ls ')
def movethings(arate,trial):
    shutil.move("estimated_transition.txt","AR_{0}/estimated_trans_{1}.txt".format(arate,trial))
    
for arate in aspace:
    for i in range(1,6):
        command = "scripts/conv_back_prop/learn_trans_po_penalty.py 0.2 {0}".format(arate)
        subprocess.call(command.split(),shell=False)
        if (os.path.isfile("estimated_transition.txt")):
            movethings(arate,i)
            
get_ipython().magic(u'ls ')
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd data/BP_Experiments/Linear_Decay/LR_0.2/')
for arate in apsace:
    os.mkdir("AR_{0}".format(arate))
    
for arate in aspace:
    os.mkdir("AR_{0}".format(arate))
    
get_ipython().magic(u'ls ')
get_ipython().magic(u'pwd ')
aspace
get_ipython().magic(u'cd ..')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'cd ..')
for arate in aspace:
    for i in range(1,6):
        command = "scripts/conv_back_prop/learn_trans_po_penalty.py 0.2 {0}".format(arate)
        subprocess.call(command.split(),shell=False)
        if (os.path.isfile("estimated_transition.txt")):
            movethings(arate,i)
            
def movethings(arate,trial):
    shutil.move("estimated_transition.txt","data/BP_Experiments/Linear_Decay/LR_0.4/AR_{0}/estimated_trans_{1}.txt".format(arate,trial))
    
for arate in aspace:
    for i in range(1,6):
        command = "scripts/conv_back_prop/learn_trans_po_penalty.py 0.4 {0}".format(arate)
        subprocess.call(command.split(),shell=False)
        if (os.path.isfile("estimated_transition.txt")):
            movethings(arate,i)
            
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd data/BP_Experiments/Linear_Decay/LR_0.2/')
get_ipython().magic(u'ls ')
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'ls ')
orig = npy.loadtxt("actual_transition.txt")
orig=orig.reshape((8,3,3))
orig
orig+=0.00001
for i in range(0,8):
    orig[i]/=orig[i].sum()
    
orig
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd data/BP_Experiments/Linear_Decay/LR_0.2/')
npy.shape(aspace)
lsq_error_2 = npy.zeros((10,5))
ce_error_2 = npy.zeros((10,5))
for i in range(0,10):
    for j in range(1,6):
        trans = npy.loadtxt("AR_{0}/estimated_trans_{1}.txt".format(aspace[i],j))
        trans=trans.reshape((8,3,3))
        for k in range(0,8):
            trans[k]/=trans[k].sum()
        lsq_error_2[i,j-1] = npy.sum((orig[:,:,:]-trans[:,:,:])**2)
        ce_error_2[i,j-1]=-npy.sum(orig[:,:,:]*npy.log(trans[:,:,:]))
        
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'ls ')
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd LR_0.2/')
for i in range(0,10):
    for j in range(1,6):
        trans = npy.loadtxt("AR_{0}/estimated_trans_{1}.txt".format(aspace[i],j))
        trans=trans.reshape((8,3,3))
        for k in range(0,8):
            trans[k]/=trans[k].sum()
        lsq_error_2[i,j-1] = npy.sum((orig[:,:,:]-trans[:,:,:])**2)
        ce_error_2[i,j-1]=-npy.sum(orig[:,:,:]*npy.log(trans[:,:,:]))
        
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd AR_0.17/')
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'cd .')
get_ipython().magic(u'cd ../../..')
get_ipython().magic(u'cd ..')
command = "scripts/conv_back_prop/learn_trans_po_penalty.py 0.2 {0}".format(a)
aspace
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd data/BP_Experiments/Linear_Decay/LR_0.2/')
get_ipython().magic(u'ls ')
aspace /1000
for a in aspace:
    os.mkdir("AR_{0}".format(a))
    
get_ipython().magic(u'ls ')
movethings()
get_ipython().magic(u'cd ..')
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'cd ..')
aspace *8
0.2- 1.48/1000*250
1.48/8
250*1.48/1000
npy.delete(aspace,0)
aspace=npy.delete(aspace,0)
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd data/BP_Experiments/Linear_Decay/LR_0.2/')
get_ipython().magic(u'ls ')
get_ipython().magic(u'rm -r AR_0.185/')
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'cd LR_0.2/')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'cd ..')
0.17*250/1000
0.2-0.17*250/1000
0.2-0.17*250/1000*8
8*0.17/1000
250*8*0.17/1000
0.185*125*8/1000
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd data/BP_Experiments/Linear_Decay/LR_0.2/')
os.mkdir("AR_0.185")
get_ipython().magic(u'ls ')
aspace
aspace2 = npy.linspace(0.2,0.05,10)
aspace2
aspace2 = npy.linspace(0.2,0.05,11)
aspace2
npy.delete(aspace2,0)
aspace2=npy.delete(aspace2,0)
for i in range(0,10):
    for j in range(1,6):
        command="scripts/conv_back_prop/learn_trans_po_penalty.py 0.2 {0}".format(aspace2[i]*8/1000)
        subprocess.call(command.split(),shell=False)
        movethings(aspace2[i],j)
        
get_ipython().magic(u'cd ../../')
get_ipython().magic(u'cd ../..')
for i in range(0,10):
    for j in range(1,6):
        command="scripts/conv_back_prop/learn_trans_po_penalty.py 0.2 {0}".format(aspace2[i]*8/1000)
        subprocess.call(command.split(),shell=False)
        movethings(aspace2[i],j)
        
for i in range(0,10):
    for j in range(1,6):
        command="scripts/conv_back_prop/learn_trans_po_penalty.py 0.2 {0}".format(aspace2[i]*8/1000)
        subprocess.call(command.split(),shell=False)
        
for i in range(0,10):
    for j in range(1,6):
        command="scripts/conv_back_prop/learn_trans_po_penalty.py 0.2 {0}".format(aspace2[i]*8/1000)
        subprocess.call(command.split(),shell=False)
        if (os.path.isfile("estimated_transition.txt")):
            movethings(aspace2[i],j)
            
a
command
subprocess.call(command.split(),shell=False)
movethings(a,3)
get_ipython().magic(u'ls ')
a = 0.125*8/1000
a
0.05*8/1000
command="scripts/conv_back_prop/learn_trans_po_penalty.py 0.2 {0}".format(a)
subprocess.call(command.split(),shell=False)
movethings(a,1)
get_ipython().magic(u'ls ')
a=0.125
command="scripts/conv_back_prop/learn_trans_po_penalty.py 0.2 {0}".format(a)
subprocess.call(command.split(),shell=False)
subprocess.call(command.split(),shell=False)
command="scripts/conv_back_prop/learn_trans_po_penalty.py 0.2 {0}".format(a*8/1000)
subprocess.call(command.split(),shell=False)
movethings(a,1)
a=0.17
command="scripts/conv_back_prop/learn_trans_po_penalty.py 0.2 {0}".format(a*8/1000)
subprocess.call(command.split(),shell=False)
subprocess.call(command.split(),shell=False)
movethings(a,1)
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd data/BP_Experiments/Linear_Decay/LR_0.2/')
get_ipython().magic(u'ls ')
orig
for i in range(0,10):
    for j in range(1,6):
        trans=npy.loadtxt("AR_{0}/estimated_trans_{1}.txt".format(aspace2[i],j))
        trans=trans.reshape((8,3,3))
        for k in range(0,8):
            trans[k]+=trans[k].sum()
        ce_error_2[i,j-1]=-npy.sum(orig[:,:,:]*npy.log(trans[:,:,:]))
        lsq_error_2[i,j-1]=npy.sum((orig[:,:,:]-trans[:,:,:])**2)
        
lsq_error_2
ce_error_2
npy.mean(ce_error_2,axis=1)
npy.min(npy.mean(ce_error_2,axis=1))
npy.max(npy.mean(ce_error_2,axis=1))
npy.mean(lsq_error_2,axis=0)
for i in range(0,10):
    for j in range(1,6):
        trans=npy.loadtxt("AR_{0}/estimated_trans_{1}.txt".format(aspace2[i],j))
        trans=trans.reshape((8,3,3))
        trans+=0.00001
        for k in range(0,8):
             trans[k]/=trans[k].sum()
       ce_error_2[i,j-1]=-npy.sum(orig[:,:,:]*npy.log(trans[:,:,:]))
       lsq_error_2[i,j-1]=npy.sum((orig[:,:,:]-trans[:,:,:])**2)
       
for i in range(0,10):
    for j in range(1,6):
        trans=npy.loadtxt("AR_{0}/estimated_trans_{1}.txt".format(aspace2[i],j))
        trans=trans.reshape((8,3,3))
        trans+=0.00001
        for k in range(0,8):
             trans[k]/=trans[k].sum()
       ce_error_2[i,j-1]=-npy.sum(orig[:,:,:]*npy.log(trans[:,:,:]))
       lsq_error_2[i,j-1]=npy.sum((orig[:,:,:]-trans[:,:,:])**2)
       
for i in range(0,10):
    for j in range(1,6):
        trans=npy.loadtxt("AR_{0}/estimated_trans_{1}.txt".format(aspace2[i],j))
        trans=trans.reshape((8,3,3))
        trans+=0.00001
        for k in range(0,8):
            trans[k]/=trans[k].sum()
        ce_error_2[i,j-1]=-npy.sum(orig[:,:,:]*npy.log(trans[:,:,:]))
        lsq_error_2[i,j-1]=npy.sum((orig[:,:,:]-trans[:,:,:])**2)
        
lsq_error_2
ce_error_2
npy.mean(ce_error_2,axis=1)
npy.min(npy.mean(ce_error_2,axis=1))
npy.min(npy.mean(lsq_error_2,axis=1))
(npy.mean(lsq_error_2,axis=1))
get_ipython().magic(u'ls ')
get_ipython().magic(u'save EXPT.py 1-225')
