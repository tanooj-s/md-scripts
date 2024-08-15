# read in a 2d density plot (a 2-tensor numpy file) and fit a circle to some section of data (assuming some kind of surface)
# note all calculations are done in internal units (agnostic of bin widths used for density calculations), bin widths needed to be added back for correct xticks and yticks
# z and y are used interchangably in this script

# the radius of the fitted circle serves as an estimate of the local curvature

# (here the wetting angle is also estimated for a circle fit to what's assumed to be a droplet on a surface, which may or may not line up with empirical data)


import argparse
import numpy as np
from scipy.optimize import basinhopping
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="fit circle (estimate wetting angle) to data from MD")
parser.add_argument('-i', action="store", dest="input") # numpy file
parser.add_argument('-title', action="store", dest="title") # title to put on plot
parser.add_argument('-o', action="store", dest="output") # png file
parser.add_argument('-dx', action="store", dest="dx")
parser.add_argument('-dz', action="store", dest="dz")
parser.add_argument('-bump', action="store", dest="bump") # amount to widen initial angular scan below COM in degrees
args = parser.parse_args()

dx, dz = float(args.dz), float(args.dz) # note hardcoded bin widths to plot in true units, this needs to match input file 
title = args.title

def COM(field2d):
    '''
    function to obtain center of mass coordinates of a 2d field
    assuming pixel values are scalars like mass/charge/number density
    '''
    mean_y = np.sum((np.mean(field2d,axis=0) * np.arange(0,field2d.shape[0],1))) / np.sum(np.mean(field2d,axis=0))
    mean_x = np.sum((np.mean(field2d,axis=1) * np.arange(0,field2d.shape[1],1))) / np.sum(np.mean(field2d,axis=1))
    return (mean_x,mean_y)

def dQdY(q):
    '''
    compute dq/dy using finite differences
    compute the derivative at point x as 0.5 * ((q(x+1)-q(x)) + (q(x)-q(x-1))) / dx
    here dy is just the width of one grid point
    assume x is axis 0, y is axis 1
    '''
    dq = np.zeros(q.shape)
    for i in range(1,q.shape[1]-1):
        dq[:,i] = 0.5 * ((q[:,i+1]-q[:,i]) + (q[:,i]-q[:,i-1]))
    dq[:,0] = 0.5 * ((q[:,1]-q[:,0]) + (q[:,0]-q[:,-1]))
    dq[:,-1] = 0.5 * ((q[:,0]-q[:,-1]) + (q[:,-1]-q[:,-2]))
    return dq # note this is in terms of grid spacings

def dQdX(q):
    '''
    compute dq/dx using finite differences
    '''
    dq = np.zeros(q.shape)
    for i in range(1,q.shape[0]-1):
        dq[i,:] = 0.5 * ((q[i+1,:]-q[i,:]) + (q[i,:]-q[i-1,:]))
    dq[0,:] = 0.5 * ((q[1,:]-q[0,:]) + (q[0,:]-q[-1,:]))
    dq[-1,:] = 0.5 * ((q[0,:]-q[-1,:]) + (q[-1,:]-q[-2,:]))
    return dq 

# use above to obtain gradient

# -------------------------------------------
# file input and computation

with open(args.input,'rb') as f: field = np.load(f)
if len(field.shape) == 3: 
    field = np.mean(field,axis=0)
print(field.shape)
assert field.shape[0] == field.shape[1]

# get center of mass
x0, y0 = COM(field)
print(f"(x0, y0) = ({x0}, {y0})")

# compute scalar gradient
dqdx = dQdX(field)
dqdy = dQdY(field)
gradient = (dqdx**2 + dqdy**2) ** 0.5


print(gradient)
print(gradient.shape)
print(np.max(gradient))


# the polar way for ground truth points
# for a range of thetas, find points on the edge that correspond to each theta
# given (x0,y0), (r,theta), find cartesian indices of the pixel corresponding to that polar coordinate
xarc, yarc = [], [] # points along arc
radii = [] # estimates of radius along angular scan, average this for initial guess for later hoptimization
dtheta = 0.333333 # degrees
bump = float(args.bump) # angle value to bump limits by
downL = 0 - bump
upL = 181 + bump # upper and lower limits for angular scan, parameters we would want to vary
thetascan = np.arange(downL,upL,dtheta) # widen this range for angle as a function of selected data 
for t in thetascan:
    # sample large radii i.e. trace this line out a fair bit
    x = x0 + np.arange(0,int(0.5*gradient.shape[0]),1)*np.cos(t * (np.pi/180))
    y = y0 + np.arange(0,int(0.5*gradient.shape[0]),1)*np.sin(t * (np.pi/180)) # note internal coordinates
    xline = np.array([int(a) for a in np.floor(x)])
    yline = np.array([int(a) for a in np.floor(y)]) # the line of points along this angle
    line = gradient[xline,yline]
    radius_idx = np.argmax(line) # line along gradient, maximum value should be on the boundary
    #fd = field[xline,yline][1:]-field[xline,yline][:-1] # finite difference for point of steepest slope of density if not using gradient
    #radius_idx = np.argmin(fd)
    radii.append(radius_idx)
    xarc.append(x0 + radius_idx * np.cos(t * (np.pi/180)))
    yarc.append(y0 + radius_idx * np.sin(t * (np.pi/180)))
arc = np.vstack((np.array(xarc),np.array(yarc))).T
radius_guess = np.mean(radii)
print(f"Initial guess for droplet radius: {radius_guess*0.05*(dx+dz)} nm")

# ------------------------------------------

# loss function for basinhoptimization
def arcLoss(params):
    '''
    assume params are r, h
    arc is a (npoints,2) shaped array to calculate loss from
    but note that it is not an argument of this function
    '''
    # the way to do this is to just sum distances from the arc points to this guessed radius
    r, h = params # guessed radius and center, x0 is fixed
    distances = []
    for pt in arc:
        distances.append(((pt[1]-h)**2 + (pt[0]-x0)**2) ** 0.5)
    distances = np.array(distances)
    distance_loss = np.sum(((distances-r) / distances) ** 2)
    return distance_loss

rinit, hinit = radius_guess, y0 # initial guess for the parameters we care about
print(f"guessed (r,h): {[rinit*0.05*(dx+dz), hinit*0.05*(dx+dz)]}")
opt_result = basinhopping(arcLoss,[rinit,hinit],niter=500,T=0.7,stepsize=0.1)
[rfit,hfit] = opt_result.x
print(f"fitted (r,h): {[rfit*0.05*(dx+dz), hfit*0.05*(dx+dz)]}")

# range of thetas for constructed circle
thetafit = (np.pi/180) * np.arange(downL,upL,1)
xfit = x0 + rfit*np.cos(thetafit)
yfit = hfit + rfit*np.sin(thetafit)
angle = 90 + (180/np.pi) * np.arcsin(hfit/rfit)
zselect = np.min(arc[:,1])
print(f"wetting angle of fitted circle: {angle}")
print(f"as a func of amount of data used")
print(f"zselect: {zselect}")


print("Generating model values....")
# model values over the entire z > 0 range for output
theta = np.arange(-90,271,1) * (np.pi/180)
xmodel = x0 + rfit*np.cos(theta)
ymodel = hfit + rfit*np.sin(theta)
xmodel = xmodel[ymodel > 0]
ymodel = ymodel[ymodel > 0]


# calculate an appropriate error function from ground truth to assess goodness of fit
# % error doesn't really make sense here
# so instead just calculate a "mean distance" error

calcErr = True
if calcErr == True:
    print("Calculating error metric...")
    xtrue = arc[:,0]
    ytrue = arc[:,1]
    idx = np.argsort(xtrue)
    xtrue, ytrue = xtrue[idx], ytrue[idx]
    error = 0
    counter = 0 
    for idx in range(len(xtrue)):
        xtru, ytru = xtrue[idx], ytrue[idx]
        xmod = np.mean(xmodel[np.abs(xmodel - xtru) < 1.5]) # find all x points close to ground truth
        ymod = np.mean(ymodel[np.abs(xmodel - xtru) < 1.5]) # same for y
        #print(f"{xtru} | {xmod} | {ytru} | {ymod}")
        dist = ((xmod-xtru)**2 + (ymod-ytru)**2) ** 0.5
        error += dist
        counter += 1
    error /= counter
    error *= 0.5*(dx+dz) # bin width for mean distance error in angstroms
else:
    error = 0
print(f"Mean distance of model points from ground truth: {error} A")



# plotting 1 panel 
plt.rcParams['figure.figsize'] = (6,6)
plt.rcParams['font.size'] = 14
plt.imshow(gradient.T,origin='lower')
plt.xlabel("x (A)")
plt.ylabel("z (A)")
plt.xticks(ticks=np.arange(0,field.shape[0],1)[::50],labels=dx*np.arange(0,field.shape[0],1)[::50]),
#plt.xticklabels(labels=dx*np.arange(0,field.shape[0],1)[::50])
plt.yticks(ticks=np.arange(0,field.shape[1],1)[::50],labels=dz*np.arange(0,field.shape[1],1)[::50]),
#plt.yticklabels(labels=dz*np.arange(0,field.shape[1],1)[::50])  
plt.scatter(x=xmodel[::5],y=ymodel[::5],marker='x',color='r',sizes=[20])
plt.scatter(x=[x0],y=[hfit],marker='x',color='r',sizes=[50])
#plt.title(f"θ={round(angle,2)}, error={round(error,2)} A")
plt.title(f"{title}, R={round(rfit*0.05*(dx+dz),2)}nm, θ={round(angle,2)}")
plt.tight_layout()


# plotting 2 panels
#plt.rcParams['figure.figsize'] = (15,5)
#plt.rcParams['font.size'] = 14
#fig, axes = plt.subplots(1,2)
#for i in range(2):
#    axes[i].imshow(gradient.T,origin='lower')
#    axes[i].set_xlabel("x (A)")
#    axes[i].set_ylabel("z (A)")
#    axes[i].set_xticks(ticks=np.arange(0,field.shape[0],1)[::50]),
#    axes[i].set_xticklabels(labels=dx*np.arange(0,field.shape[0],1)[::50])
#    axes[i].set_yticks(ticks=np.arange(0,field.shape[1],1)[::50]),
#    axes[i].set_yticklabels(labels=dz*np.arange(0,field.shape[1],1)[::50])   
#axes[0].scatter(x=arc[:,0][::5],y=arc[:,1][::5],marker='x',color='r',sizes=[20])
#axes[0].scatter(x=[x0],y=[y0],marker='x',color='r',sizes=[50])
#axes[0].set_title("Ground truth along arc")
#axes[1].scatter(x=xmodel[::5],y=ymodel[::5],marker='x',color='r',sizes=[20])
#axes[1].scatter(x=[x0],y=[hfit],marker='x',color='r',sizes=[50])
#axes[1].set_title(f"Fitted circle, (r,h)=({round(rfit,2)},{round(hfit,2)}) => θ={round(angle,2)}, error={round(error,2)} A")


#output = args.input[:-8]+'.png'
plt.savefig(args.output)
plt.clf()
