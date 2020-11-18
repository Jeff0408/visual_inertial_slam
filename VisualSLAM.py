import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy.linalg import inv, pinv
from utils import *
from hw3_a_b import *

    
if __name__ == '__main__':
    
    filename = "./data/0027.npz"
    t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)
    
    W = np.eye(6)*0.001
    V = np.eye(4)*100
    #set projection matrix
    P = np.block([[np.eye(3)], [np.zeros((1, 3))]])
    imu_T_cam = np.linalg.inv(cam_T_imu)
    #build M
    M = np.zeros((4,4))
    M[0:2,0:3] = K[0:2,0:3]
    M[2:4,0:3] = K[0:2,0:3]
    M[2,3] = (-1)*K[0,0]*b
     
    # Visuzl -inertia SLAM
    
    position_now_SLAM=np.eye(4)
    trajectory_SLAM = np.zeros((4,4,t.shape[1]-1)) #wTi
    trajectory_SLAM[:,:,0] = np.eye(4) 
    #set covariance
    cov_imu = np.zeros((6, 6, t.shape[1]-1))
    cov_imu[:,:,0] = np.eye(6) 

    cov_SLAM = np.zeros((3*features.shape[1] + 6, 3*features.shape[1] + 6))
    cov_SLAM = np.eye((3*features.shape[1] + 6))
    landmarks_SLAM = np.zeros((4,features.shape[1]))
    
    index = np.flatnonzero(features[0, :, 0] != -1)  # valid observation given current frame with size (N_t,)
    landmarks_SLAM[:, index] = back_projection(features[:, index, 0], M, imu_T_cam, trajectory_SLAM[:, :, 0])
    print (landmarks_SLAM[:,0])
    
    #imu pose update
    for time in tqdm.trange(1, t.shape[1]-1, desc = 'SLAM', unit = 'frame'):

        t_d = t[0][time+1]-t[0][time]

        iTw = Prediction(time,rotational_velocity,linear_velocity,t_d,iTw)
        trajectory_SLAM[:,:,time] = np.linalg.inv(iTw)
    
        
        u = np.hstack((linear_velocity[:, time], rotational_velocity[:, time])).squeeze()
        Exp = expm(-t_d * cur_hat(u))
        cov_imu[:, :, time] = np.dot(Exp, np.dot(cov_imu[:, :, time - 1], Exp.T)) + t_d ** 2 * W
        
        cov_SLAM[-6:, -6:] = cov_imu[:, :, time]
        cov_SLAM[:3*features.shape[1], 3*features.shape[1]:] = np.dot(cov_SLAM[:3*features.shape[1], 3*features.shape[1]:], Exp.T)
        cov_SLAM[3*features.shape[1]:, :3*features.shape[1]] = cov_SLAM[:3*features.shape[1], 3*features.shape[1]:].T
        
        #the inverse IMU pose
        U_t_SLAM = inv(trajectory_SLAM[:,:,time])
        #position at time stamp t 
        mu_t_SLAM = trajectory_SLAM[:,:,time]
        #visual feature observation 
        z0_t_SLAM = features[:,:,time]
        
        #mj = 4*Nt (nummber of valid observed feature)
        indices = np.flatnonzero(features[0, :, time] != -1)
        m_j_SLAM = features[:, indices, time]
        N_t = len(indices)
        if N_t:
            z_lk = features[:, indices, time]
            
            
            # landark = 4*1
            index = (z0_t_SLAM[0,:]!= -1)
            true_index = np.where(index == True)
         
            for i in true_index[0]:
                
                if np.all(landmarks_SLAM[:,i] == False):
                    ul, _ , ur, _ = features[:,i,time]
                    landmarks_SLAM[:,i] = mu_t_SLAM.dot(inv(cam_T_imu).dot(pinv(M).dot(z0_t_SLAM[:,i])*M[0,0]*b/(ul-ur)))
                    
            #calculate z hat
            
            #imu z hat
    
            q_lk = np.dot(cam_T_imu, np.dot(U_t_SLAM, landmarks_SLAM[:,indices]))
            z_hat_lk = np.dot(M, Pi(q_lk))
            
            #calculate H 4Nt*3M
            H  = np.zeros((4*N_t, 3 * features.shape[1] + 6))
            for j in range(N_t):
                #landmarks update part of H
                
                q_0 = np.dot(cam_T_imu, np.dot(U_t_SLAM, landmarks_SLAM[:,indices[j]].reshape(4, -1)))  
                dpidq_oT_mu_m_0 = dPidq(q_0)
                oT_mu_D = np.dot(cam_T_imu, np.dot(U_t_SLAM, P))
                H_lk = np.dot(M, np.dot(dpidq_oT_mu_m_0, oT_mu_D))
                
                #imu pose update part of H 
                q = np.dot(cam_T_imu, np.dot(U_t_SLAM, landmarks_SLAM[:,indices[j]]))    
                dpidq_oT_mu_m = dPidq(q)
                odot_SLAM = odot(np.dot(U_t_SLAM, landmarks_SLAM[:,indices[j]].reshape(4, -1)))
                H_imu = np.dot(M, np.dot(dpidq_oT_mu_m, np.dot(cam_T_imu, odot_SLAM)))
                
                H[4*j:4*j + 4, 3*indices[j]:3*indices[j] + 3] = H_lk
                H[4*j:4*j + 4, -6:] = H_imu
            
            #calculation of K
            K = cov_SLAM @ H.T @ np.linalg.inv(H @ cov_SLAM @ H.T + np.kron(np.eye(N_t), V))    
            #k = np.linalg,inv(np.dot(H, np.dot(cov_SLAM, H.T)) + np.kron(np.eye(N_t), V))
            #K = np.dot(cov_SLAM, np.dot(H.T, k))
               
            #update landmarks mean
            landmarks_SLAM = (landmarks_SLAM.flatten('F') + np.kron(np.eye(features.shape[1]), P) @ K[:3*features.shape[1], :] @
                                       (z_lk - z_hat_lk).flatten('F')).reshape(features.shape[1], 4).T
            
            #update imu mean
            iTw = expm(hat(K[-6:, :] @ (z_lk - z_hat_lk).flatten('F'))) @ iTw   
            trajectory_SLAM[:,:,time] = np.linalg.inv(iTw)
            #update covariance
            cov_SLAM = (np.eye(3*features.shape[1] + 6) - K @ H) @ cov_SLAM
            cov_imu[:, :, time] = cov_SLAM[-6:, -6:]

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    visualize_trajectory(axes[0], trajectory)
    visualize_landmark(axes[1], trajectory, landmarks)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    visualize_trajectory(axes[0], trajectory_SLAM)
    visualize_landmark(axes[1], trajectory_SLAM, landmarks_SLAM)
    plt.show()      
    

                
                
                
                
                
                
                
                
                
                
                
                