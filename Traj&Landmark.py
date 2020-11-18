import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy.linalg import inv, pinv
from utils import *


if __name__ == '__main__':
    
    filename = "./data/0027.npz"
    t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)
    
    position_now=np.eye(4)
    trajectory = np.zeros((4,4,t.shape[1]-1)) #wTi

    landmarks = np.zeros((4,features.shape[1]))
    
    #IMU-based Localization via EKF Prediction:
    
    for i in range(t.shape[1]-1):
    #for i in range(5):
        #print(i)
        t_d = t[0][i+1]-t[0][i]
        position_now = Prediction(i,rotational_velocity,linear_velocity,t_d,position_now)
        trajectory[:,:,i] = np.linalg.inv(position_now)
    
    #build M
    M = np.zeros((4,4))
    M[0:2,0:3] = K[0:2,0:3]
    M[2:4,0:3] = K[0:2,0:3]
    M[2,3] = (-1)*K[0,0]*b
     
    #covariance matrix
    cov = []
    for k in range(features.shape[1]):
        cov.append(np.eye(3))
        
    cov = np.stack((cov), axis = -1)
    
    #set noise
    noise= 10*np.abs(np.random.randn(1))* np.eye(4)
    
    #set projection matrix
    P = np.block([[np.eye(3)], [np.zeros((1, 3))]])
    
    #Landmark Mapping via EKF Update
    for t in range(t.shape[1]-1):
        #the inverse IMU pose
        U_t = inv(trajectory[:,:,t])
        #position at time stamp t 
        mu_t = trajectory[:,:,t]
        #visual feature observation 
        z0_t = features[:,:,t]
        
        #pick valid data
        index = (z0_t[0,:]!= -1)
        true_index = np.where(index == True)
        
        for i in true_index[0]:
            
            if np.all(landmarks[:,i] == False):
                ul, _ , ur, _ = features[:,i,t]
                landmarks[:,i] = mu_t.dot(inv(cam_T_imu).dot(pinv(M).dot(z0_t[:,i])*M[0,0]*b/(ul-ur)))
            
            q = np.dot(cam_T_imu, np.dot(U_t, landmarks[:,i]))
            dpidq_oT_mu_m = dPidq(q)
            oT_mu_D = np.dot(cam_T_imu, np.dot(U_t, P))
            H = np.dot(M, np.dot(dpidq_oT_mu_m, oT_mu_D))
            
            k = pinv(np.dot(H, np.dot(cov[:,:,i], H.T)) + noise)
            K = np.dot(cov[:,:,i], np.dot(H.T, k))
            
            z_hat = np.dot(M, Pi(q))
            landmarks[:,i] = landmarks[:,i] + np.dot(P, np.dot(K, (features[:,i,t] - z_hat)))
            cov[:,:,i] = np.dot((np.eye(3) - np.dot(K, H)), cov[:,:,i])
            #print(cov) 
    

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    visualize_trajectory(axes[0], trajectory)
    visualize_landmark(axes[1], trajectory, landmarks)
    plt.show()

	# (a) IMU Localization via EKF Prediction

	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM

	# You can use the function below to visualize the robot pose over time
	#visualize_trajectory_2d(world_T_imu,show_ori=True)
