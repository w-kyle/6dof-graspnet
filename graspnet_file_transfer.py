'''Allows graspnet to be called from a python 3 program. Specifically
intended to allow comparison to GPD and MA grasping methods

Written by Colin Keil'''
import numpy as np
import sys
#graspnet path
g_path = '/home/colin/Documents/Research/graspnet_project/6dof-graspnet'
sys.path.append(g_path)
import argparse
import grasp_estimator
import os
import tensorflow as tf
import glob

import mayavi.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from visualization_utils import *
from grasp_data_reader import regularize_pc_point_count
# import visualization_utils
vae_checkpoint_folder = '/home/colin/Documents/Research/graspnet_project/6dof-graspnet/checkpoints/latent_size_2_ngpus_1_gan_1_confidence_weight_0.1_npoints_1024_num_grasps_per_object_256_train_evaluator_0_'
evaluator_checkpoint_folder = '/home/colin/Documents/Research/graspnet_project/6dof-graspnet/checkpoints/npoints_1024_train_evaluator_1_allowed_categories__ngpus_8_'
#required by some error in my gpu setup??
tf.test.is_gpu_available()
# tf.


#path where files will transfer
transfer_path = '/home/colin/Documents/Research/rpn_gpd2/graspnet_file_transfer/'
sample_path = transfer_path + 'samples.npy'
points_path = transfer_path + 'pointcloud.npy'
grasps_path = transfer_path + 'grasps.npy'
grasps_path_temp = transfer_path + 'graspstemp.npy'


#wait for signal from comparison code
running = False
last_modified = None
print('Initialized, waiting...\n')
while True:

    #set the number of samples in the configuration
    if os.path.exists(sample_path):

        #setup graspnet configuration
        cfg = grasp_estimator.joint_config(
            vae_checkpoint_folder,
            evaluator_checkpoint_folder,
        )

        cfg['threshold'] = -1
        cfg['sample_based_improvement'] = 1
        cfg['num_refine_steps'] = 20
        cfg['num_samples']=np.load(sample_path).item()
        print('configuring number of samples : {}'.format(cfg['num_samples']))
        # input('hit enter to continue')
        estimator = grasp_estimator.GraspEstimator(cfg)
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
        sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))
        print(sess.list_devices())
        estimator.build_network()
        estimator.load_weights(sess)
        #get the file write time
        last_modified = os.path.getmtime(points_path)
        #delete the sample file to send a confirmation
        os.remove(sample_path)
        #allow other code to run
        running = True

    if running:
        #check if a point cloud has been transferred
        if os.path.exists(points_path):
            #get the file write time
            tmp_last_modified = os.path.getmtime(points_path)
            #if the file write time matches the stored time, do nothing
            if not tmp_last_modified == last_modified:
                last_modified = tmp_last_modified
                #read in the pointcloud
                pc = np.load(points_path, allow_pickle=True)
                #check that the format is valid
                if not type(pc) is np.ndarray:
                    continue
                else:
                    print('Cloud recieved')
                #sample and evaluate grasps
                latents = estimator.sample_latents()
                generated_grasps, generated_scores, _ = estimator.predict_grasps(
                    sess,
                    pc[:,:3],
                    latents,
                    num_refine_steps=cfg.num_refine_steps,
                )
                #output the data
                #TODO verify dict key names
                data = {'samples':generated_grasps, 'scores':generated_scores}
                np.save(grasps_path_temp, data, allow_pickle=True)
                os.rename(grasps_path_temp, grasps_path)


                # #plot in matplotlib
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax.view_init(30, 30)
                # # ax.scatter(pc[:,0], pc[:,1], pc[:,2], '.')
                # ax.scatter(pc[:,0], pc[:,1], pc[:,2], '.')
                # grasp_points = np.array([T[:3,3].reshape(-1) for T in generated_grasps])
                # filtered_grasp_points = []
                # filtered_grasp_scores = []
                # filtered_grasps = []
                # for i in range(len(grasp_points)):
                #     if np.linalg.norm(grasp_points[i])<0.8:
                #         filtered_grasps.append(generated_grasps[i])
                #         filtered_grasp_points.append(grasp_points[i])
                #         filtered_grasp_scores.append(generated_scores[i])
                # filtered_grasp_points = np.array(filtered_grasp_points)
                # ax.scatter(filtered_grasp_points[:,0],filtered_grasp_points[:,1],filtered_grasp_points[:,2])
                #
                # # ax.scatter(max_grasp[0],max_grasp[1],max_grasp[2],'*')
                # # ax.scatter(camera_pose[0,3],camera_pose[1,3],camera_pose[2,3],'y')
                #
                # # ax.scatter(pos_a[0], pos_a[1], pos_a[2])
                # # ax.scatter(pos_b[0], pos_b[1], pos_b[2])
                # # print(pos_x)
                # # print(pos_y)
                # # print(pos_z)
                # # ax.plot([origin[0],pos_x[0]],[origin[1],pos_x[1]],[origin[2],pos_x[2]],'b')
                # # ax.plot([origin[0],pos_y[0]],[origin[1],pos_y[1]],[origin[2],pos_y[2]],'r')
                # # ax.plot([origin[0],pos_z[0]],[origin[1],pos_z[1]],[origin[2],pos_z[2]],'y')
                #
                # ax.set_xlabel('X Label')
                # ax.set_ylabel('Y Label')
                # ax.set_zlabel('Z Label')
                #
                # plt.show()
                #
                # #debug
                # mlab.figure(bgcolor=(1,1,1))
                # draw_scene(
                # pc,
                # pc_color=None,
                # grasps=filtered_grasps,
                # grasp_scores=filtered_grasp_scores,
                # )
                # mlab.show()
