def movethings(l, a, t):
    shutil.move("estimated_transition.txt","data/BP_Experiments/Linear_Decay/LR_range/LR_{0}/AR_{1}/estimated_trans_{2}.txt".format(l,a,t))

# for arate in aspace:
for l in lspace: 
    for a  in apace
    for i in range(1,6):
        command = "scripts/conv_back_prop/learn_trans_po_penalty.py {0} {1}".format(8*arate/1000)
        subprocess.call(command.split(),shell=False)
        if (os.path.isfile("estimated_transition.txt")):
            movethings(l,i)

for i in range(17,20):
    for j in range(0,10):
        print "We are doing Learning Rate {0} with Annealing Rate {1}".format(lspace[i],aspace[i,j])
        for k in range(1,6):
            command = "scripts/conv_back_prop/learn_trans_po_penalty.py {0} {1}".format(lspace[i], lspace[i]-aspace[i,j])
            subprocess.call(command.split(),shell=False)
            if (os.path.isfile("estimated_transition.txt")):
                movethings(lspace[i],aspace[i,j],k)
            else:             
                subprocess.call(command.split(),shell=False)
                if (os.path.isfile("estimated_transition.txt")):
                    movethings(lspace[i],aspace[i,j],k)
def done(i,j,k):
    command = "scripts/conv_back_prop/learn_trans_po_penalty.py {0} {1}".format(lspace[i], lspace[i]-aspace[i,j])
    subprocess.call(command.split(),shell=False)
    if (os.path.isfile("estimated_transition.txt")):
        movethings(lspace[i],aspace[i,j],k)

for i in range(0,8):
	print i 
	for j in range(1,6):
		trans=npy.loadtxt("AR_{0}/estimated_trans_{1}.txt".format(aspace[i],j))
        print aspace[i]
        trans=trans.reshape((8,3,3))
        trans+=0.00001
        for k in range(0,8):
        	trans[k]/=trans[k].sum()
        ce_error_4[i,j-1]=-npy.sum(orig[:,:,:]*npy.log(trans[:,:,:]))
        lsq_error_4[i,j-1]=npy.sum((orig[:,:,:]-trans[:,:,:])**2)

for j in range(1,6):
    trans=npy.loadtxt("AR_{0}/estimated_trans_{1}.txt".format(a_val,j))
    trans=trans.reshape((8,3,3))
    trans+=0.00001
    for k in range(0,8):
        trans[k]/=trans[k].sum()
    print -npy.sum(orig[:,:,:]*npy.log(trans[:,:,:]))
    print npy.sum((orig[:,:,:]-trans[:,:,:])**2)

with file('CE_Error.txt','w') as outfile: 
	npy.savetxt(outfile, ce_error_4, fmt='%-7.5f')

with file('estimated_trans_linear.txt','w') as outfile: 
	npy.savetxt(outfile, trans, fmt='%-7.5f')

def prep_trans(i,j,k):
    global trans
    trans = npy.loadtxt("LR_{0}/AR_{1}/estimated_trans_{2}.txt".format(lspace[i],aspace[i,j],k))
    trans=trans.reshape((8,3,3))
    trans+=0.00001
    for k in range(0,8):
        trans[k]/=trans[k].sum()

def prep_orig():
    global trans
    orig = npy.loadtxt("actual_transition.txt")
    orig=orig.reshape((8,3,3))
    orig+=0.00001
    for k in range(0,8):
        orig[k]/=orig[k].sum()

for j in range(6,10):
    for k in range(1,6):
        command = "scripts/conv_back_prop/learn_trans_po_penalty.py {0} {1}".format(lspace[i], lspace[i]-aspace[i,j])
        subprocess.call(command.split(),shell=False)
        if (os.path.isfile("estimated_transition.txt")):
            movethings(lspace[i],aspace[i,j],k)
        else:             
            subprocess.call(command.split(),shell=False)
            if (os.path.isfile("estimated_transition.txt")):
                movethings(lspace[i],aspace[i,j],k)




for i in range(0,20):
    for j in range(0,10):
        for k in range(1,6):
            trans=npy.loadtxt("LR_{0}/AR_{1}/estimated_trans_{2}.txt".format(lspace[i],aspace[i,j],k))
            trans=trans.reshape((8,3,3))
            trans+=0.00001
            for p in range(0,8):
                trans[p]/=trans[p].sum()
            lsq_po[i,j,k-1]=npy.sum((orig[:,:,:]-trans[:,:,:])**2)
















def movethings(l, a, t):
    shutil.move("estimated_transition.txt","data/BP_Experiments/Linear_Decay/FO_LR_range/LR_{0}/AR_{1}/estimated_trans_{2}.txt".format(l,a,t))

def done(i,j,k):
    command = "scripts/conv_back_prop/learn_trans.py {0} {1}".format(lspace[i], lspace[i]-aspace[i,j])
    subprocess.call(command.split(),shell=False)
    print "SET: ",i,j,k
    if (os.path.isfile("estimated_transition.txt")):
        movethings(lspace[i],aspace[i,j],k)


def done(i,j,k):
    command = "scripts/conv_back_prop/weighted_counting.py {0} {1}".format(lspace[i], lspace[i]-aspace[i,j])
    subprocess.call(command.split(),shell=False)
    print "SET: ",i,j,k
    if (os.path.isfile("estimated_transition.txt")):
        movethings(lspace[i],aspace[i,j],k)


def movethings(l, a, t):
    shutil.move("estimated_transition.txt","data/BP_Experiments/Weighted/LR_{0}/AR_{1}/estimated_trans_{2}.txt".format(l,a,t))


for i in range(8,10):
    for j in range(0,6):
        for k in range(1,6):
            done(i,j,k)

            command = "scripts/conv_back_prop/weighted_counting.py {0} {1}".format(lspace[i],aspace[i,j])
            subprocess.call(command.split(),shell=False)
            if (os.path.isfile("estimated_transition.txt")):
                movethings(lspace[i],aspace[i,j],k)
            else:             
                subprocess.call(command.split(),shell=False)
                if (os.path.isfile("estimated_transition.txt")):
                    movethings(lspace[i],aspace[i,j],k)


for i in range(0,10):
    for j in range(0,6):
        for k in range(1,6):
            if not(os.path.isfile("data/BP_Experiments/Weighted/LR_{0}/AR_{1}/estimated_trans_{2}.txt".format(lspace[i],aspace[i,j],k))):
                done(i,j,k)