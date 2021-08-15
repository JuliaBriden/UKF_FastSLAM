# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 12:47:17 2021

@author: julia
"""
from pylab import*
import matplotlib
from matplotlib import collections  as mc
import os
import random
import scipy.linalg
import numpy.matlib


def pi_to_pi(x):
    for i in range(x.shape[0]):
        if x[i]>np.pi:
            while x[i]>np.pi:
                x[i]=x[i]-2*np.pi
        elif x[i]<-np.pi:
            while x[i]<-np.pi:
                x[i]=x[i]+2*np.pi
    return x
                
def multivariate_gauss(x,P,n):
    S = np.linalg.cholesky(P)
    X = np.random.rand(x.shape[0],n)
    return S@X+x@np.ones((1,n))
            

class particle():
    def __init__(self,n):
        self.w = 1/n # weight
        self.xv = np.zeros((3,1)) # vehicle pose
        self.Pv = 0.000001*np.identity(3) # initial robot covariance
        self.Kaiy = [] # keep for msmt update
        self.xf = [] # feature mean states
        self.Pf = [] # feature covariances
        self.zf = [] # known feature locations
        self.idf = [] # known feature index
        self.zn = [] # new feature locations
        
    def compute_jacobians(self,idf,R):
        zp = np.zeros((2,1))
        Hv = []
        Hf = []
        Sf = []
        Sv = []
        for i in range(len(np.matrix(idf))):
            dx = self.xf[0,i]-float(self.xv[0])
            dy = self.xf[1,i]-float(self.xv[1])
            d = np.sqrt(dx**2+dy**2)
            zp[0][0] = d
            zp[1][0] = pi_to_pi(math.atan2(dy,dx)-self.xv[2]) # predicted observation
            Hv.append([-dx/d,-dy/d, 0])
            Hv.append([dy/d**2,-dx/d**2,-1]) # Vehicle states
            Hf.append([dx/d,dy/d])
            Hf.append([-dy/d**2,dx/d**2]) # Feature states
            Pf = np.array([[self.Pf[0][i][i],self.Pf[i][1][i]],[self.Pf[1][i][i],self.Pf[1][1][i]]])
            Sf.append(Hf*Pf*np.transpose(Hf+R)) # innovation covariance
            Sv.append(Hv*self.Pv*np.transpose(Hv)+R)
        return zp,Hv,Hf,Sf,Sv
            
    def add_feature(self,z,R):
        xf = np.zeros((2,len(z)))
        Pf = np.zeros((2,2,len(z)))
        for i in range(len(z)):
            r = z[i][0][0]
            b = z[i][1][0]
            s = float(np.sin(self.xv[2]+b))
            c = float(np.cos(self.xv[2]+b))
            xf[0,i] = float(self.xv[0])+r*c
            xf[1,i] = float(self.xv[1])+r*s
            Gz = np.matrix([[c,-r*s],[s,r*c]])
            Pf[:2,:2,i] = Gz*R*np.transpose(Gz)
        
        if any(self.xf):
            self.xf = np.concatenate((self.xf,xf),axis=1)
            self.Pf = np.concatenate((self.Pf,Pf),axis=2)
        else:
            self.xf = xf
            self.Pf = Pf


            
class vehicle():
    def __init__(self):
        # Vehicle paparameters
        self.dt = 0.025 # seconds btw control signals
        self.L = 2.83 # m
        self.H = 0.76 # m
        self.b = 0.5 # m
        self.a = 3.78 # m
        self.F = np.array([[0, -self.H, -self.H], [0, -1, 1]]) # vehicle
        # Control Noise
        self.sigV = 2; # m/s
        self.sigG = 6*np.pi/180; # rad
        self.Q = np.array([[self.sigV**2, 0], [0, self.sigG**2]])
        self.percept_limit = 30 # m
        
    def sample_proposal(self,particle,zf,idf,R,n,l,wg,wc):
        # batch update
        z_hat = np.zeros((len(particle.zf[0])*len(idf),1)) # observation prediction
        z = np.zeros((len(particle.zf[0])*len(idf),1)) # sensor observation
        A = np.zeros((len(particle.zf[0])*len(idf),2*n+1))
        wc_s = np.sqrt(wc)
        Ksi = np.zeros((n,2*n+1))
        for i in range(len(idf)):
            j = idf[i] # index of observed feature
            x_fi = particle.xf[:,j] # jth feature of mean
            P_fi = particle.Pf[:,:,j] # jth feature of covariance
            z[i:i+2] = zf[:][i] # stack of sensor observations
            
            # state augmentation
            x_aug = np.concatenate((particle.xv,np.transpose(np.matrix(x_fi))),axis=0)
            P_aug = np.concatenate((np.concatenate((particle.Pv, np.zeros((particle.xv.shape[0],len(particle.zf[0])))),axis=1),\
                              np.concatenate((np.zeros((len(particle.zf[0]),particle.xv.shape[0])),P_fi),axis=1)))
            
            # sigma points
            Ps = (n+l)*P_aug+0.000001*np.identity(n)
            Ss = np.transpose(np.linalg.cholesky(Ps))
            Ksi = np.zeros((n,2*n+1))
            Ksi[:,0] = np.transpose(x_aug)
            for k in range(n):
                Ksi[:,k+1] = np.transpose(x_aug+Ss[:,k])
                Ksi[:,k+1+n] = np.transpose(x_aug-Ss[:,k])
            
            # observation model
            A_i = np.matrix(np.zeros((len(particle.zf[0]),2*n+1)))
            bs = np.zeros(2*n+1) # bearing sign
            z_hat_i = 0 # predicted observation
            for k in range(2*n+1): # pass sigma pts through obs model
                d = Ksi[particle.xv.shape[0]+1:particle.xv.shape[0]+len(particle.zf[0]),k]-Ksi[:particle.xv.shape[0]-1,k]
                r = np.sqrt(d[0]**2+d[1]**2)
                bearing = math.atan2(d[1],d[0])
                bs[k] = np.sign(bearing)
                if k>1:
                    if not bs[k] == bs[k-1]:
                        if bs[k] < 0 and -np.pi < bearing and bearing <-np.pi/2:
                            bearing = bearing+2*np.pi
                            bs[k] = np.sign(bearing)
                        elif bs[k]>0 and np.pi/2 < bearing and bearing < np.pi:
                            bearing = bearing-2*np.pi
                            bs[k] = np.sign(bearing)
                A_i[:,k] = np.array([[r],[bearing-Ksi[particle.xv.shape[0],k]]])
                z_hat_i = z_hat_i + wg[k]*A_i[:,k]
            z_hat_i_rep = np.matlib.repmat(z_hat_i,1,2*n+1)
            A[i:i+2,:] = A_i - z_hat_i_rep
            for k in range(2*n+1):
                A[2*i-1:2*i,k] = A[2*i-1:2*i,k]*wc_s[k]
            z_hat_i[1,0] = pi_to_pi(z_hat_i[1])
            z_hat[i:i+2,0] = np.transpose(z_hat_i)
        
        # augmented noise matrix
        R_aug = np.zeros((len(particle.zf[0])*len(idf),len(particle.zf[0])*len(idf)))
        for i in range(len(idf)):
            R_aug[i:i+2,i:i+2] = R
        
        # innovation covariance
        S = np.matmul(A,np.transpose(A)) # vehicle uncertainty, map, msmt noise
        S = (S+np.transpose(S))*0.5+R_aug # make symmetric
        
        # cross covariance with vehicle uncertainty
        X = np.matrix(np.zeros((particle.xv.shape[0],2*n+1))) # stack
        for k in range(2*n+1):
            X[:,k] = wc_s[k]*(Ksi[:1,k]-particle.xv)
        
        # cross covariance
        U = np.matmul(X,np.transpose(A))
        
        # Kalman gain
        K = U@np.linalg.inv(S)
       
        # innovation
        v = z-z_hat
        for i in range(len(idf)):
            v[2*i] = pi_to_pi(v[2*i])
        
        # standard Kalman update
        xv = particle.xv+K*v
        Pv = particle.Pv-K*np.transpose(U)
        
        # compute weight
        w = np.exp(-0.5*np.transpose(v)/S*v)/np.sqrt(2*np.pi*np.linalg.det(S))
        particle.w = particle.w*w
        
        # sample from proposal distribution
        xvs = multivariate_gauss(xv,Pv,1)
        particle.xv = xvs
        particle.Pv = np.identity(3)*0.000001 # initialize covariance
        return particle
    
    def feature_update(self,particle,zf,idf,R,n,l,wg,wc):
        xf = particle.xf[:,idf]
        Pf = particle.Pf[:,:,idf]
        
        # each feature
        for i in range(len(idf)):
            # augmented feature state
            xf_aug = np.hstack((xf[:,i],np.zeros(2)))
            Pf_aug = np.vstack((np.hstack((Pf[:,:,i],np.zeros((2,2)))),np.hstack((np.zeros((2,2)),R))))
            
            # dissemble the covariance
            P = (n+l)*Pf_aug+0.000001*np.identity(n)
            S = np.transpose(np.linalg.cholesky(P))
            
            # get sigma points
            Kai = np.matrix(np.zeros((n,2*n+1)))
            Kai[:,0] = np.transpose(np.matrix(xf_aug))
            for k in range(n):
                Kai[:,k+1] = np.transpose(np.matrix(xf_aug+S[:,k]))
                Kai[:,k+1+n] = np.transpose(np.matrix(xf_aug-S[:,k]))
            
            # transform the sigma points
            Z = np.matrix(np.zeros((particle.xf.shape[0],2*n+1)))
            bs = np.zeros(2*n+1) # bearing
            for k in range(2*n+1):
                d = Kai[1:3,k]-particle.xv[:2]
                r = np.sqrt(float(d[0])**2+float(d[1])**2)+Kai[2,k] # range and noise
                bearing = math.atan2(float(d[1]),float(d[0]))
                bs[k] = np.sign(bearing)
                if k>1: # unify sign
                    if not bs[k] == bs[k-1]:
                        if bs[k]<0 and -np.pi<bearing and bearing<-np.pi/2:
                            bearing = bearing+2*np.pi
                            bs[k] = np.sign(bearing)
                        elif bs[k]>0 and np.pi/2<bearing and bearing<np.pi:
                            bearing = bearing-2*np.pi
                            bs[k] = np.sign(bearing)
                
                # predict sigma points
                Z[:,k] = np.array([[r],[bearing-particle.xv[2]+Kai[3,k]]]) # bearing and noise
                
            # predict observations
            z_hat = 0
            for k in range(2*n+1):
                z_hat = z_hat+wg[k]*Z[:,k]
            
            # innovation covariance
            St = 0
            for k in range(2*n+1):
                St = St+wc[k]*(Z[:,k]-z_hat)*np.transpose(Z[:,k]-z_hat)
            St = (St+np.transpose(St))*0.5 # symmetric
            
            # cross covariance
            Sigma = 0
            for k in range(2*n+1):
                Sigma = Sigma+wc[k]*(Kai[:1,k]-xf[:,i])*(Z[:,k]-z_hat)
                
            # update
            v = zf[:][i]-z_hat
            v[1] = pi_to_pi(v[1])
            
            Kt = Sigma/St
            xf[:,i] = np.transpose(np.transpose(np.matrix(xf[:,i]))+Kt*v)
            Pf[:,:,i] = Pf[:,:,i]-Kt*St*np.transpose(Kt)
            
        particle.xf[:,idf] = xf
        particle.Pf[:,:,idf] = Pf
        return particle
        
                
            

class uslam():
    def __init__(self):
        self.plines = np.zeros((2))
        self.feature_count = 1 # laser feature count init
        self.est_path = []; # estimation path
        
    def predict(self,particle,V,G,Q,vehicle,dt,n,l,wg,wc):
        # state augmentation
        x_a = np.concatenate((particle.xv,np.zeros((Q.shape[0],1))),axis=0)
        P_a = np.concatenate((concatenate((particle.Pv, np.zeros((particle.xv.shape[0],Q.shape[0]))),axis=1),\
                             concatenate((np.zeros((Q.shape[0],particle.xv.shape[0])),Q),axis=1)),axis=0)
        # sigma pts
        Z = (n+l)*P_a+0.00001*np.identity(n)
        S = np.matrix(np.linalg.cholesky(Z))
        Kaix = np.matrix(np.zeros((n,2*n+1)))
        Kaix[:,0] = x_a
        Kaiy = np.zeros((particle.xv.shape[0],2*n+1))
        xv_p = 0
        Pv_p = 0
        
        for k in range(n):
            Kaix[:,k+1] = x_a+S[:,k]
            Kaix[:,k+1+n] = x_a-S[:,k]
            
        for k in range(2*n+1):
            # switched up order
            Vn = V+Kaix[3,k] # add process noise of linear speed
            Gn = G+Kaix[4,k] # add process noise of steering
            Vc = Vn/(1-np.tan(Gn)*vehicle.H/vehicle.L) # transform
            Kaiy[0,k] = Kaiy[0,k]+dt*(Vc*np.cos(Kaix[3,k])-Vc/vehicle.L*np.tan(Gn)*\
                (vehicle.a*np.sin(Kaix[2,k])+vehicle.b*np.cos(Kaix[2,k])))
            Kaiy[1,k] = Kaix[1,k]+dt*(Vc*np.sin(Kaix[2,k])+Vc/vehicle.L*np.tan(Gn)*\
                (vehicle.a*np.cos(Kaix[2,k])-vehicle.b*np.sin(Kaix[2,k])))
            Kaiy[2,k] = Vc*dt*np.tan(Gn)/vehicle.L
        
        xv_p = 0
        for k in range(2*n):
            xv_p = xv_p+wg[k]*Kaiy[:,k]
        
        Pv_p = np.zeros((3,3))
        for k in range(2*n+1):
            Pv_p = Pv_p+wc[k]*np.outer((Kaiy[:,k]-xv_p),np.transpose(Kaiy[:,k]-xv_p))
            
        particle.xv = np.transpose(np.matrix(xv_p))
        particle.Pv = np.matrix(Pv_p)
        particle.Kaiy = np.matrix(Kaiy)
        
        return particle
            
    def get_observation(self,idx,obs_a,obs_b,percept_limit):
        z = []

        for i in range(len(obs_a[0])):
            if (not obs_a[idx][i] == 0) and (obs_a[idx][i] < percept_limit):
                z.append(np.stack([[obs_a[idx][i]],[obs_b[idx][i]-np.pi*0.5]]))
        return z
        
    def data_association(self,particle,z,R,gate1,gate2):                
        def compute_nis(particle,z,R,idf):
            # normalized innovation squared and normalized distance
            z_pred,_,_,sf,sv = particle.compute_jacobians(idf,R)
            v = z-z_pred
            v[1] = pi_to_pi(np.matrix(v[1]))
            nis = np.transpose(v)@np.linalg.inv(sf)@v
            return float(nis)
        # gated nearest neighbor data association
        zf = []
        zn = []
        idf = []
        
        if any(particle.xf):
            nf = len(particle.xf[0]) # number of features on the map already (columns #)
        else:
            nf = 0
        zp = np.zeros((2,1))
        
        xv = particle.xv
        
        # linear search for nearest neighbor
        for i in range(len(z)):
            j_best = -1
            outer = np.inf
            
            if not nf == 0:
                # search for nearest neighbors
                d_min = np.inf
                j_best_s = -1
                
                for j in range(nf):
                    dx = particle.xf[0][j]-xv[0]
                    dy = particle.xf[1][j]-xv[1]
                    z[0][0] = np.sqrt(dx**2+dy**2)
                    z[1][0] = pi_to_pi(math.atan2(dy,dx)-xv[2])
                    v = z[:][i]-zp
                    v[1] = pi_to_pi(np.matrix(v[1]))
                    d_candidate = np.transpose(v)@v
                    if float(d_candidate) < d_min:
                        d_min = d_candidate
                        j_best_s = j
                        
                # Mahalanobis test for candidate neighbor
                nis = compute_nis(particle,z[:][i],R,j_best_s)
                if nis < gate1: # store nearest neighbor
                    j_best = j_best_s
                elif nis < outer:
                    outer = nis
                    
            # add nn to association list
            if not j_best == -1:
                zf.append(z[:][i])
                idf.append(j_best)
            elif outer > gate2: # new feature
                zn.append(z[:][i])
                
        particle.zf = zf
        particle.idf = idf
        particle.zn = zn
            
        return particle
                        
    def resample_particles(self,particles,N_min):
        def stratified_resample(w):
            def stratified_random(N):
                k = 1/N
                d_i = np.arange(k/2,1-k/2+k,k) # interval
                return d_i+np.random.rand(N)*k-k/2
                    
            # normalize set of weights
            w = w/sum(w)
            Neff = 1/sum(w**2)
            
            keep = np.zeros(len(w))
            select = stratified_random(len(w))
            w = np.cumsum(w)
                
            ctr = 0
            for i in range(len(w)):
                while ctr<len(w) and select[ctr]<w[i]:
                    keep[ctr] = i
                    ctr += 1
            return keep,Neff
                         
                
        # resample particles
        N = len(particles)
        w = np.zeros(N)
        for i in range(N):
            w[i] = particles[i].w
        
        ws = sum(w)
        w = w/ws
        
        for i in range(N):
            particles[i].w = particles[i].w/ws
            
        keep,Neff = stratified_resample(w)
        if Neff<=N_min:
            particles = particles[keep]
            for i in range(N):
                particles[i].w = 1/N
                    
        return particles
        
    def get_estimated_path(self,particles,N):
        # vehicle state estimate
        xvp = []
        w = []
        for i in range(N):
            xvp.append(particles[i].xv)
            w.append(particles[i].w)
        
        # normalize
        w = w/sum(w)
        
        # weighed mean vehicle pose
        xv_mean = 0
        for i in range(N):
            xv_mean = xv_mean+particles[i].w*particles[i].xv
        
        # store pose
        return xv_mean # returns est path
    
# main----------------------------------------------------------------------------
# plot
# Some SLAM plotting functions and dataset reading functions are not mine

def init_ground_truth(PATH,GPS):
        """
        Processes the GPS data and rotates it in order to match the control data rotation.
        :return:
        """
        f = open(PATH+GPS)
        data = [[float(num) for num in line.split(',') if len(num) > 0] for line in f]
        f.close()
        t, x, y = [],[],[]
        xoff, yoff =  data[0][1], data[0][2]
        for d in data:
            t.append(d[0])
            x.append(d[1]-xoff)
            y.append(d[2]-yoff)
        return t, x, y
    
def __plot_laser(z, xv, line_col):
    def make_laser_lines(rb, xv):
        def TransformToGlobal(p, b):
                # Transform a list of poses [x;y;phi] so that they are global wrt a base pose
                # rotate
                phi = b[2]
                rot = np.array([[cos(phi), -sin(phi)],
                       [sin(phi), cos(phi)]])
                p[0:2] = np.dot(rot, p[0:2])
        
                # translate
                p[0] = p[0] + b[0]
                p[1] = p[1] + b[1]

                return p
        """
        Creates the laser lines from the estimated position of the car to the detected obstacles.
        :param rb:
        :param xv: vehicle position
        :return: list of laser lines
        """

        len_ = len(rb)
        lnes_x = np.zeros((1, len_)) + xv[0]
        lnes_y = np.zeros((1, len_)) + xv[1]

        if rb:
            lnes_end_pos = TransformToGlobal([np.multiply(rb[0][:], np.cos(rb[1][:])),
                                              np.multiply(rb[0][:], np.sin(rb[1][:]))], np.matrix(xv))
            data = []
            for i in range(len(rb[0])):
                data.append([(lnes_x[0][i], lnes_y[0][i]), (lnes_end_pos[0][i], lnes_end_pos[1][i])])
        else:
            data = []
        return data
    
    """
    Plots laser lines.
    :param z: observed features
    :param xv: vehicle pose.
    """
    lines = make_laser_lines(z, xv)
    if line_col != None : #remove previous laser lines
        line_col.remove()
    lc = mc.LineCollection(lines, colors = np.array(('blue', 'blue', 'blue', 'blue')), linewidths=3)
    axtot.add_collection(lc)
    line_col = lc
    return axtot,line_col

    
    
PATH        = "victoria_park/"
GPS         = "mygps.txt"
IMGPATH     = "output_img"
OUTPATH     = "output"

#Rotational data for GPS
alfa = math.atan(-0.71)
c = np.cos(alfa)
s = np.sin(alfa)
ferr = open(OUTPATH+'error.txt', "w+")
x_map = open(OUTPATH+'x_data.txt', "w+")
y_map = open(OUTPATH+'y_data.txt', "w+")
x_feat = open(OUTPATH+'x_feat.txt', "w+")
y_feat = open(OUTPATH+'y_feat.txt', "w+")
cov = open(OUTPATH+'cov.txt', "w+")

errcount = 0
err_vect = []
error_value = []
path_count = 0
err = []
epath = []
xdata = []
ydata = []
theta = []
xgt = [] #x of ground truth
ygt = [] #y of ground truth
covariances = []
#gttime, gtx, gty = init_ground_truth(PATH,GPS)
line_col = None

# Create output directories
if not os.path.exists(OUTPATH):
    os.makedirs(OUTPATH)
if not os.path.exists(IMGPATH):
    os.makedirs(IMGPATH)
    
"""Enables plotting and its frequency and saving images to output folder."""
    
def plot_est(particle,axtot,est_path):
    def cov_ellipse(particle):
        def make_ellipse(x,P,circ):
            Q,T = scipy.linalg.schur(P)
            R = np.zeros((2,2))
            R[0][0] = sqrt(T[0][0])
            R[1][1] = sqrt(T[1][1])
            R[0][1] = T[0][1]/(R[0][0]+R[1][1])
            r = Q*R*np.transpose(Q)
            
            a = np.matmul(r,circ)
            p = []
            
            p = np.concatenate((np.concatenate((a[0,:]+x[0],np.matrix([None])),axis=1),np.concatenate((a[1,:]+x[1], np.matrix([None])),axis=1)),axis=0)
            
            return p
            
        N = 10 # points in ellipse
        inc = 2*np.pi/N
        phi = np.array(np.linspace(0,2*np.pi,N,endpoint=True))
        circ = 2*np.array([np.cos(phi), np.sin(phi)])
        
        p = make_ellipse(particle.xv[0:2],particle.Pv[0:2,0:2]+eye(2)*0.00001,circ)
        len_f = particle.xf[0].shape[0]
        
        if len_f > 0:
            p = np.concatenate((p,np.zeros((2,len_f*(N+2)))),axis=1)
            
            c = N+3
#            for i in range(len_f):
#                for j in range(c,c+N+1):
#                    p = np.concatenate((make_ellipse(np.vstack((particle.xf[0][i],particle.xf[1][i])),np.array([[particle.Pf[0][i][i],particle.Pf[i][1][i]],[particle.Pf[1][i][i],particle.Pf[1][1][i]]]),circ)),axis=0)
#                c = c+N+2
        return p

        
    xvp = np.matrix(np.zeros((3,len(particles))))
    xfp = []
    w = []
    for i in range(len(particles)):
        xvp[:,i] = particles[i].xv
        xfp.append(particles[i].xf)
        w.append(particles[i].w)
        
    j = np.argwhere(w == max(w))
    
    #if xvp:
    axtot.scatter(np.asarray(xvp[0][:][0]),np.asarray(xvp[1][:][0]),s=3, color='black')
        
    #if len(xfp[0])>1:
    #    axtot.scatter(xfp[0][0],xfp[0][1],s=1, color='red')
        
    #if line_col:
    #    axtot.add_collection(line_col)
    #pcov = []
    #if len(j) == 1:
    #    pcov = cov_ellipse(particles[int(j)])
    #elif len(j)>1:
    #    for i in j:
    #        pcov = cov_ellipse(particles[i])
    #if any(pcov):
    #    axtot.plot(pcov[0],pcov[1],color='blue')
    return axtot
    
    
figtot, axtot = plt.subplots()
figerr, axerr = plt.subplots()
#axtot.set_xlim(-150, 250)
#axtot.set_ylim(-150, 250)
axtot.set_aspect('equal')
axtot.set_xlabel('x [m]')
axtot.set_ylabel('y [m]')
axtot.set_title('Result map against GT')
    
#axerr.set_ylim(0, 17)
axerr.set_xlabel('number steps')
axerr.set_xlabel('number steps')
axerr.set_ylabel('error [m]')
axerr.set_title('Error')
    
line, = axtot.plot([], [], 'r-')

gt, = axtot.plot(xgt, ygt, 'g-')
oldFeatures = axtot.scatter([], [])
olderror = axerr.scatter([], [])

plt.ion()
#mng = plt.get_current_fig_manager()
#mng.window.state('zoomed')
plt.show()

# environment
p_lines = np.zeros((2,1))
feature_count = 0 # laser feature count
est_path = []

# load dataset
path = "victoria_park/"
distance = "ObservationDistance.txt"
angle = "ObservationAngle.txt"
laser_sample_t = "lasersampling.txt"
timesteps = "time.txt"
speed = "speed.txt"
steering = "steering.txt"


f = open(path+distance,'r')
OD = [[float(num) for num in line.split('  ') if len(num)>0] for line in f ]
f.close()

f = open(path+angle)
OA = [[float(num) for num in line.split('  ') if len(num)>0] for line in f ]
f.close()

f = open(path+laser_sample_t)
lasersampling = [float(line) for line in f]
f.close()

f = open(path+timesteps)
sampling = [float(line) for line in f]
f.close()

f = open(path+speed)
V = [float(line) for line in f]
f.close()

f = open(path+steering)
G = [float(line) for line in f]
f.close()

n_laser_sample = len(lasersampling)
n_sample = np.round(len(sampling))
        
# msmst noise
sigmaR = 1
sigmaB = 3*np.pi/180
Re = np.array([[sigmaR**2, 0],[0, sigmaB**2]])

# Resampling criteria
n_particles = 10
n_effective = 0.5*n_particles # min num of effective particles
        
# Data association 
gate_rej = 5.991
gate_aug_NN = 2000
gat_aug = 100
        
# Vehicle update
dimv = 3
dimQ = 2
dimR = 2
dimf = 2
        
n_aug = dimv+dimf
a_aug = 0.9
b_aug = 2
k_aug = 0
        
l_aug = a_aug**2*(n_aug+k_aug)-n_aug
l_aug += dimR
wg_aug = np.zeros((2*n_aug+1))
wc_aug = np.zeros((2*n_aug+1))
wg_aug[0] = l_aug/(n_aug+l_aug)
wc_aug[0] = l_aug/(n_aug+l_aug)+(1-a_aug**2+b_aug)
        
for i in range(1,2*n_aug+1):
    wg_aug[i] = 1/(2*(n_aug+l_aug))
    wc_aug[i] = wg_aug[i]
        
# Vehicle prediction
n_r = dimv+dimQ
a_r = 0.9
b_r = 2
k_r = 0
l_r = a_r**2+(n_r+k_r)-n_r+dimR
wg_r = np.zeros((2*n_r+1))
wc_r = np.zeros((2*n_r+1))
wg_r[0] = l_r/(n_r+l_r)
wc_r[0] = l_r/(n_r+l_r)+(1-a_r**2+b_r)

for i in range(1,2*n_r+1):
    wg_r[i] = 1/(2*(n_r+l_r))
    wc_r[i] = wg_r[i]
        
# Feature updates
n_f_a = dimf+dimR
a_f_a = 0.9
b_f_a = 2
k_f_a = 0
l_f_a = a_f_a**2*(n_f_a+k_f_a)-n_f_a

wg_f_a = np.zeros((2*n_f_a+1))
wc_f_a = np.zeros((2*n_f_a+1))
wg_f_a[0] = l_f_a/(n_f_a+l_f_a)
wc_f_a[0] = l_f_a/(n_f_a+l_f_a)+(1-a_f_a**2+b_f_a)

for i in range(1,2*n_f_a+1):
    wg_f_a[i] = 1/(2*(n_f_a+l_f_a))
    wc_f_a[i] = wg_f_a[i]
    
# Feature initialization
n_f = dimf
a_f = 0.9
b_f = 2
k_f = 0
l_f = a_f**2*(n_f+k_f)-n_f
wg_f = np.zeros((2*n_f+1))
wc_f = np.zeros((2*n_f+1))
wg_f[0] = l_f/(n_f+l_f)
wc_f[0] = l_f/(n_f+l_f)+(1-a_f**2+b_f)

for i in range(1,2*n_f+1):
    wg_f[i] = 1/(2*(n_f+l_f))
    wc_f[i] = wg_f[i]
    
# slam
particles = [particle(n_particles) for i in range(n_particles)] # initialize particles
slam = uslam()
robot = vehicle()

# sampling steps
for t in range(n_sample):
    if not V[t] == 0:
        
        # Predict vehicle state
        i = 0
        for i in range(n_particles):
            particles[i] = slam.predict(particles[i],V[t],G[t],robot.Q,robot,robot.dt,n_r,l_r,wg_r,wc_r)
        # Msmt update
        for i_obs in range(feature_count,n_laser_sample):
            
            # find msmts for time sequence
            if i_obs < n_laser_sample and lasersampling[i_obs]<=sampling[t+1] and lasersampling[i_obs]>sampling[t]:
                
                # get obs
                z = slam.get_observation(feature_count,OD,OA,robot.percept_limit)
                #axtot,line_col = __plot_laser(z,particles[i].xv,line_col)
                #plot_curr_graph(figtot, axtot, figerr, axerr, particles[i].xv[0], particles[i].xv[1])
                
                # Data association
                for i in range(n_particles):
                    particles[i] = slam.data_association(particles[i], z, Re, gate_rej, gate_aug_NN)
                
                # map features known
                for i in range(n_particles):
                    if particles[i].zf:
                        # sample from optimal proposal dist
                        particles[i] = robot.sample_proposal(particles[i],particles[i].zf,particles[i].idf,Re,n_aug,\
                                                             l_aug,wg_aug,wc_aug)
                        # map update
                        particles[i] = robot.feature_update(particles[i],particles[i].zf,particles[i].idf,Re,n_f_a,\
                                                            l_f_a,wg_f_a,wc_f_a)
                        
                # resample
                particles = slam.resample_particles(particles,n_effective)
                
                # augment new features to map
                for i in range(n_particles):
                    if particles[i].zn:
                        # sample from proposal dist
                        if not particles[i].zf:
                            particles[i].xv = multivariate_gauss(particles[i].xv,particles[i].Pv,1)
                            particles[i].Pv = 0.000001*np.identity(3)
                        particles[i].add_feature(particles[i].zn,Re)
                
                feature_count = i_obs
                break
                
        # plot
        epath.append(slam.get_estimated_path(particles,n_particles))
        axtot = plot_est(particles,axtot,epath)
        plt.draw()
        figtot.savefig('uslam_map_victoria.png')

figtot.savefig('uslam_map_victoria_new.png')

        
data = particles