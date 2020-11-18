import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy.linalg import inv, pinv
from transforms3d.euler import mat2euler

def load_data(file_name):
  '''
  function to read visual features, IMU measurements and calibration parameters
  Input:
      file_name: the input data file. Should look like "XXX_sync_KLT.npz"
  Output:
      t: time stamp
          with shape 1*t
      features: visual feature point coordinates in stereo images, 
          with shape 4*n*t, where n is number of features
      linear_velocity: IMU measurements in IMU frame
          with shape 3*t
      rotational_velocity: IMU measurements in IMU frame
          with shape 3*t
      K: (left)camera intrinsic matrix
          [fx  0 cx
            0 fy cy
            0  0  1]
          with shape 3*3
      b: stereo camera baseline
          with shape 1
      cam_T_imu: extrinsic matrix from IMU to (left)camera, in SE(3).
          close to 
          [ 0 -1  0 t1
            0  0 -1 t2
            1  0  0 t3
            0  0  0  1]
          with shape 4*4
  '''
  with np.load(file_name) as data:
      t = data["time_stamps"] # time_stamps
      features = data["features"] # 4 x num_features : pixel coordinates of features
      linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
      rotational_velocity = data["rotational_velocity"] # rotational velocity measured in the body frame
      K = data["K"] # intrindic calibration matrix
      b = data["b"] # baseline
      cam_T_imu = data["cam_T_imu"] # Transformation from imu to camera frame
      
  return t,features[:,::10,:],linear_velocity,rotational_velocity,K,b,cam_T_imu


def visualize_trajectory_2d(pose,path_name="Unknown",show_ori=False):
  '''
  function to visualize the trajectory in 2D
  Input:
      pose:   4*4*N matrix representing the camera pose, 
              where N is the number of pose, and each
              4*4 matrix is in SE(3)
  '''
  fig,ax = plt.subplots(figsize=(5,5))
  n_pose = pose.shape[2]
  ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
  ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
  ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
  if show_ori:
      select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
      yaw_list = []
      for i in select_ori_index:
          _,_,yaw = mat2euler(pose[:3,:3,i])
          yaw_list.append(yaw)
      dx = np.cos(yaw_list)
      dy = np.sin(yaw_list)
      dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
      ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
          color="b",units="xy",width=1)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.axis('equal')
  ax.grid(False)
  ax.legend()
  plt.show(block=True)
  return fig, ax



def Prediction(i, rotational_velocity, linear_velocity, t_d, mu_next):
    
    #initialization
    rotational_velocity_now = np.zeros((3,3))
    linear_velocity_now = np.zeros((3,1))
    u_t_hat=np.zeros((4,4))
    
    #rotational velocity
    rotational_velocity_1 = rotational_velocity[0][i]
    rotational_velocity_2 = rotational_velocity[1][i]
    rotational_velocity_3 = rotational_velocity[2][i]
    rotational_velocity_now = np.array([[0, -rotational_velocity_3, rotational_velocity_2],
                                        [rotational_velocity_3, 0, -rotational_velocity_1],
                                        [-rotational_velocity_2, rotational_velocity_1, 0]])
    #linaer velocity          
    linear_velocity_1 = linear_velocity[0][i]
    linear_velocity_2 = linear_velocity[1][i]
    linear_velocity_3 = linear_velocity[2][i]
    linear_velocity_now = np.array([linear_velocity_1,linear_velocity_2,linear_velocity_3])
    
    #set u_t_hat (4*4)
    u_t_hat[0:3,0:3] = rotational_velocity_now
    u_t_hat[0:3,3] = linear_velocity_now
    u_t_hat[3,:] = 0
    
    mu_next = np.dot(expm((-1) * t_d * u_t_hat),mu_next)
    
    return mu_next 

def Pi(q):
    #pi function
    q = q/q[2]
    
    return q

def dPidq(q):
    
    q = q.squeeze()
    
    q = (1/q[2]) * np.array([[1, 0, -q[0]/q[2], 0],
                                 [0, 1, -q[1]/q[2], 0],
                                 [0, 0,          0, 0],
                                 [0, 0, -q[3]/q[2], 1]])    
    return q
    
def hat(x):
    x = x.squeeze()

    if len(x) == 3:
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])
              
    elif len(x) == 6:
        return np.block([[hat(x[3:]), x[:3].reshape((3, 1))], [np.zeros((1, 4))]])
    
    
def cur_hat(x):
    
    return np.block([[hat(x[3:]), hat(x[:3])], [np.zeros((3, 3)), hat(x[3:])]])

def odot(x):
    
    return np.block([[x[3]*np.eye(3), -hat(x[0:3])],
                     [np.zeros((1, 6))]])

def back_projection(observations, M, imu_T_cam, wld_T_imu):
    d = observations[0] - observations[2]
    fsub = -M[2, 3]
    z = fsub/d
    x = (observations[0] - M[0, 2]) * z / M[0, 0]
    y = (observations[1] - M[1, 2]) * z / M[1, 1]
    po = np.vstack((x, y, z, np.ones(x.shape)))       # homogeneous coord in camera frame
    pw = wld_T_imu @ imu_T_cam @ po
    return pw

def visualize_trajectory(ax, pose, title=None):

    ax.plot(pose[0, 3, :], pose[1, 3, :], 'r-')
    ax.scatter(pose[0, 3, 0], pose[1, 3, 0], marker='s', label="start")
    ax.scatter(pose[0, 3, -1], pose[1, 3, -1], marker='o', label="end")
    ax.set_xlabel('x / m')
    ax.set_ylabel('y / m')
    ax.set_title(title)
    ax.axis('equal')
    ax.grid(False)
    ax.legend()


def visualize_landmark(ax, pose, landmarks, title=None):

    ax.plot(landmarks[0], landmarks[1], 'k.')
    ax.plot(pose[0, 3, :], pose[1, 3, :], 'g-')
    ax.set_xlabel('x / m')
    ax.set_ylabel('y / m')
    ax.set_title(title)
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
