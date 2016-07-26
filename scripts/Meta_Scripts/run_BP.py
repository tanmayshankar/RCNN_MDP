def movethings(arate,trial):
    shutil.move("estimated_transition.txt","data/BP_Experiments/Linear_Decay/LR_0.4/AR_{0}/estimated_trans_{1}.txt".format(arate,trial))
    
for arate in aspace:
    for i in range(1,6):
        command = "scripts/conv_back_prop/learn_trans_po_penalty.py 0.4 {0}".format(8*arate/1000)
        subprocess.call(command.split(),shell=False)
        if (os.path.isfile("estimated_transition.txt")):
            movethings(arate,i)

for i in range(0,8):
	for j in range(1,6):
		trans=npy.loadtxt("AR_{0}/estimated_trans_{1}.txt".format(aspace[i],j))
		trans=trans.reshape((8,3,3))
		trans+=0.00001
        for k in range(0,8):
        	trans[k]/=trans[k].sum()
		ce_error_3[i,j-1]=-npy.sum(orig[:,:,:]*npy.log(trans[:,:,:]))
		lsq_error_3[i,j-1]=npy.sum((orig[:,:,:]-trans[:,:,:])**2)