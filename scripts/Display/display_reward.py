#!/usr/bin/env python

from variables import *

# optimal_policy = npy.loadtxt(str(sys.argv[1]))
# optimal_policy = optimal_policy.astype(int)

reward_function = npy.loadtxt(str(sys.argv[1]))
# value_function = npy.loadtxt(str(sys.argv[3]))

path_plot = copy.deepcopy(reward_function)
max_val = npy.amax(path_plot)

##THE ORIGINAL ACTION SPACE:
action_space = npy.array([[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]])
## UP,  DOWN,  LEFT, RIGHT,UPLEFT,UPRIGHT,DOWNLEFT,DOWNRIGHT ##################

def show_image(image_arg):
	plt.imshow(image_arg, interpolation='nearest', origin='lower', extent=[0,50,0,50], aspect='auto')
	plt.show(block=False)
	plt.colorbar()
	plt.show() 

show_image(reward_function)
# show_image(optimal_policy)
# show_image(value_function)

# N=50
# Y,X = npy.mgrid[0:N,0:N]

# U = npy.zeros(shape=(discrete_size,discrete_size))
# V = npy.zeros(shape=(discrete_size,discrete_size))

# for i in range(0,discrete_size):
# 	for j in range(0,discrete_size):
# 		U[i,j] = action_space[optimal_policy[i,j]][0]
# 		V[i,j] = action_space[optimal_policy[i,j]][1]		

# fig, ax = plt.subplots()
# im = ax.imshow(reward_function, origin='lower',extent=[-1,50,-1,50])

# ax.quiver(V,U)
# # ax.quiver(U,V)
# # ax.quiver(X,Y,U,V)

# fig.colorbar(im)
# ax.set(aspect=1, title='Quiver Plot')
# plt.show()

# fig, ax = plt.subplots()
# # im = ax.imshow(reward_function, origin='lower',extent=[0,50,0,50])
# im = ax.imshow(value_function, origin='lower',extent=[-1,50,-1,50])
# # im = ax.imshow(dummy, origin='lower',extent=[-1,50,-1,50])

# # ax.quiver(X,Y,U,V)
# ax.quiver(V,U)
# # ax.quiver(U,V)

# fig.colorbar(im)
# ax.set(aspect=1, title='Quiver Plot')
# plt.show()

# def main(arg1,arg2,arg3):
# 	print " "

# if __name__ == "__main__":
#     main(sys.argv[1],sys.argv[2],sys.argv[3])