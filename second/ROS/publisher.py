import os
import rospy
import math
import sys
from glob import glob

from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from gps_common.msg import GPSFix
from sensor_msgs.msg import Imu

import numpy as np
import pickle
from pathlib import Path
import pykitti
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from second.pytorch.inference import TorchInferenceContext
import second.core.box_np_ops as box_np_ops

point_size = 1.0
axes_str = ['X', 'Y', 'Z']
axes_limits = [
    [-20, 80], # X axis range
    [-20, 20], # Y axis range
    [-3, 10]   # Z axis range
]
num_features = 4

def kitti_anno_to_corners(info, annos=None):
    rect = info['calib/R0_rect']
    P2 = info['calib/P2']
    Tr_velo_to_cam = info['calib/Tr_velo_to_cam']
    if annos is None:
        annos = info['annos']
    dims = annos['dimensions']
    loc = annos['location']
    rots = annos['rotation_y']
    scores = None
    if 'score' in annos:
        scores = annos['score']
    boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
    boxes_lidar = box_np_ops.box_camera_to_lidar(boxes_camera, rect, Tr_velo_to_cam)
    boxes_corners = box_np_ops.center_to_corner_box3d(
        boxes_lidar[:, :3],
        boxes_lidar[:, 3:6],
        boxes_lidar[:, 6],
        origin=[0.5, 0.5, 0],
        axis=2)
    
    return boxes_corners, scores, boxes_lidar


def draw_box(pyplot_axis, vertices, axes=[0, 1, 2], color='black'):
    """
    Draws a bounding 3D box in a pyplot axis.
    
    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=0.5)

def draw_point_cloud(ax, title, points, boxes_corners, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):
    """
    Convenient method for drawing various point cloud projections as a part of frame statistics.
    """
    ax.scatter(*np.transpose(points[:, axes]), s=point_size, c=points[:, 3], cmap='gray')
    ax.set_title(title)
    ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
    ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
    if len(axes) > 2:
        ax.set_xlim3d(*axes_limits[axes[0]])
        ax.set_ylim3d(*axes_limits[axes[1]])
        ax.set_zlim3d(*axes_limits[axes[2]])
        ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
    else:
        ax.set_xlim(*axes_limits[axes[0]])
        ax.set_ylim(*axes_limits[axes[1]])
    # User specified limits
    if xlim3d!=None:
        ax.set_xlim3d(xlim3d)
    if ylim3d!=None:
        ax.set_ylim3d(ylim3d)
    if zlim3d!=None:
        ax.set_zlim3d(zlim3d)

    for boxes_corner in boxes_corners:
        t_rects = np.transpose(boxes_corner)
        draw_box(ax, t_rects, axes=axes, color=(1,0,0))

def draw_point_cloud_color(ax, title, points, colors, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):
    """
    Convenient method for drawing various point cloud projections as a part of frame statistics.
    """
    ax.scatter(*np.transpose(points[:, axes]), s=point_size, c=colors, cmap='gray')
    ax.set_title(title)
    ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
    ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
    if len(axes) > 2:
        ax.set_xlim3d(*axes_limits[axes[0]])
        ax.set_ylim3d(*axes_limits[axes[1]])
        ax.set_zlim3d(*axes_limits[axes[2]])
        ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
    else:
        ax.set_xlim(*axes_limits[axes[0]])
        ax.set_ylim(*axes_limits[axes[1]])
    # User specified limits
    if xlim3d!=None:
        ax.set_xlim3d(xlim3d)
    if ylim3d!=None:
        ax.set_ylim3d(ylim3d)
    if zlim3d!=None:
        ax.set_zlim3d(zlim3d)
        
    
def network_inference_by_path(kitti_info, v_path, sampling, flip):
    # print(v_path)
    points = np.fromfile(v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
    points = points[1::sampling] # sampling

    if flip:
        points[:,0:3] = np.dot(points[:,0:3], np.array([[-1,0,0],[0,-1,0],[0,0,1]]))
    
    selected_points = []
    for point in points:
        if (point[0] > 0):
            selected_points.append(point)
    selected_points = np.asarray(selected_points)

    inputs = inference_ctx.get_inference_input_dict(kitti_info, selected_points)
    with inference_ctx.ctx():
        det_annos = inference_ctx.inference(inputs)

    boxes_corners, scores, boxes_lidar = kitti_anno_to_corners(kitti_info, det_annos[0])
    class_names = det_annos[0]['name']

    # f2 = plt.figure(figsize=(15, 8))
    # ax2 = f2.add_subplot(111, projection='3d')                    
    # draw_point_cloud(ax2, 'Velodyne scan', points, boxes_corners, xlim3d=(-10,30))
    # plt.show()
    
    return selected_points, boxes_corners, class_names

def bounding_box_filter(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf, min_z=-np.inf, max_z=np.inf):
    """ Compute a bounding_box filter on the given points

    Parameters
    ----------                        
    points: (n,3) array
        The array containing all the points's coordinates. Expected format:
            array([
                [x1,y1,z1],
                ...,
                [xn,yn,zn]])

    min_i, max_i: float
        The bounding box limits for each coordinate. If some limits are missing,
        the default values are -infinite for the min_i and infinite for the max_i.

    Returns
    -------
    bb_filter : boolean array
        The boolean mask indicating wherever a point should be keeped or not.
        The size of the boolean mask will be the same as the number of given points.

    """

    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
    bound_z = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)

    bb_filter = np.logical_and(bound_x, bound_y, bound_z)

    return bb_filter


def make_bbox_lines(bounding_box_points, x_max, x_min, y_max, y_min, z_max, z_min):

    bounding_box_points.append([x_max,y_max,z_max])
    bounding_box_points.append([x_max,y_min,z_max])
    bounding_box_points.append([x_max,y_max,z_min])
    bounding_box_points.append([x_max,y_min,z_min])
    bounding_box_points.append([x_min,y_max,z_max])
    bounding_box_points.append([x_min,y_min,z_max])
    bounding_box_points.append([x_min,y_max,z_min])
    bounding_box_points.append([x_min,y_min,z_min])

    bounding_box_points.append([x_max,y_max,z_max])
    bounding_box_points.append([x_max,y_max,z_min])
    bounding_box_points.append([x_min,y_max,z_max])
    bounding_box_points.append([x_min,y_max,z_min])
    bounding_box_points.append([x_min,y_min,z_max])
    bounding_box_points.append([x_min,y_min,z_min])
    bounding_box_points.append([x_max,y_min,z_max])
    bounding_box_points.append([x_max,y_min,z_min])

    bounding_box_points.append([x_max,y_max,z_max])
    bounding_box_points.append([x_min,y_max,z_max])
    bounding_box_points.append([x_max,y_min,z_max])
    bounding_box_points.append([x_min,y_min,z_max])
    bounding_box_points.append([x_max,y_max,z_min])
    bounding_box_points.append([x_min,y_max,z_min])
    bounding_box_points.append([x_max,y_min,z_min])
    bounding_box_points.append([x_min,y_min,z_min])

    return bounding_box_points

def euler_to_quaternion(roll, pitch, yaw):

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

# Configures
lidar_seq_id = '2011_09_30_drive_0018'
lidar_file_path = '/KITTI/%s/%s_sync/velodyne_points/data/*.bin' % (lidar_seq_id[0:10], lidar_seq_id)
lidar_file_paths = sorted(glob(lidar_file_path))
gps_imu_file_path = '/KITTI/%s/%s_sync/oxts/data/*.txt' % (lidar_seq_id[0:10], lidar_seq_id)
gps_imu_file_paths = sorted(glob(gps_imu_file_path))

info_path = '/workspace/data/kitti_infos_train.pkl'
config_path = Path('/workspace/SECOND-ROS/second/configs/all.fhd.config')
ckpt_path = Path('/workspace/pretrained_models/original_model/voxelnet-74240.tckpt')

os.environ["CUDA_VISIBLE_DEVICES"]="1"
labeled_pointcloud_topic_name = '/inference_results'
boudning_box_topic_name = '/bbox_results'
gps_topic_name = '/gps'
imu_topic_name = '/imu'
lidar_frame_step = 1
point_sampling_step = 1


if __name__ == '__main__':

    # Network model load
    with open(info_path, 'rb') as f:
        kitti_infos = pickle.load(f)
    inference_ctx = TorchInferenceContext()
    inference_ctx.build(config_path)
    inference_ctx.restore(ckpt_path)

    # publisher init
    rospy.init_node('SECOND network pub_example')
    labeled_pointcloud_pub = rospy.Publisher(labeled_pointcloud_topic_name, PointCloud2)
    bounding_box_pub = rospy.Publisher(boudning_box_topic_name, Marker)
    gps_pub = rospy.Publisher(gps_topic_name, GPSFix)
    imu_pub = rospy.Publisher(imu_topic_name, Imu)

    rospy.sleep(1.)
    rate = rospy.Rate(3)

    idx = 0
    while not rospy.is_shutdown():

        all_points = []
        all_bounding_box_points = []
        marker = Marker()
        marker.header.frame_id = '/map'
        marker.header.stamp = rospy.Time.now()
        marker.type = marker.LINE_LIST
        marker.scale.x = 0.1
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.5

        for flip in [True, False]:            
            points, boxes_corners, class_names = network_inference_by_path(kitti_infos[0], lidar_file_paths[idx], point_sampling_step, flip)

            
            bounding_box_points = []
            labels = np.zeros(points.shape[0])
            for boxes_corner, class_name in zip(boxes_corners, class_names):
                x_max = np.max(boxes_corner[:,0])
                x_min = np.min(boxes_corner[:,0])
                y_max = np.max(boxes_corner[:,1])
                y_min = np.min(boxes_corner[:,1])
                z_max = np.max(boxes_corner[:,2])
                z_min = np.min(boxes_corner[:,2])

                bounding_box_points = make_bbox_lines(bounding_box_points, x_max, x_min, y_max, y_min, z_max, z_min)


                if class_name == 'Pedestrian':
                    labels = labels + 1*bounding_box_filter(points, x_min, x_max, y_min, y_max, z_min, z_max)
                elif class_name == 'Cyclist':
                    labels = labels + 2*bounding_box_filter(points, x_min, x_max, y_min, y_max, z_min, z_max)
                elif class_name == 'Car':
                    labels = labels + 3*bounding_box_filter(points, x_min, x_max, y_min, y_max, z_min, z_max)
                elif class_name == 'Van':
                    labels = labels + 4*bounding_box_filter(points, x_min, x_max, y_min, y_max, z_min, z_max)

            for i, label in enumerate(labels):
                points[i,3] = label
            
            if flip:
                points[:,0:3] = np.dot(points[:,0:3], np.array([[-1,0,0],[0,-1,0],[0,0,1]]))
                if len(boxes_corners) != 0:
                    bounding_box_points = np.dot(np.asarray(bounding_box_points), np.array([[-1,0,0],[0,-1,0],[0,0,1]]))
            
            all_points.append(points)
            if len(boxes_corners) != 0:
                all_bounding_box_points.append(bounding_box_points)

        all_points = np.vstack(all_points)
        if len(boxes_corners) != 0:
            all_bounding_box_points = np.vstack(all_bounding_box_points)
            for bounding_box_point in all_bounding_box_points:
                marker.points.append(Point(bounding_box_point[0],bounding_box_point[1],bounding_box_point[2]))

        #header
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = '/map'
        
        #create pcl from points
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('intensity', 12, PointField.FLOAT32, 1),
                # PointField('rgba', 12, PointField.UINT32, 1),
                 ]
        scaled_polygon_pcl = pcl2.create_cloud(header, fields, all_points)
        labeled_pointcloud_pub.publish(scaled_polygon_pcl)
        bounding_box_pub.publish(marker)

        # gps/imu topic gen
        with open(gps_imu_file_paths[idx], 'rt') as fp:
            line = fp.readline().split(' ')
            quat_val = euler_to_quaternion(float(line[3]),float(line[4]),float(line[5]))

            navsat = GPSFix()
            navsat.header.frame_id = '/map'
            navsat.header.stamp = rospy.Time.now()
            navsat.latitude = float(line[0])
            navsat.longitude = float(line[1])
            navsat.altitude = float(line[2])
            gps_pub.publish(navsat)

            imu = Imu()
            imu.header.frame_id = '/map'
            imu.header.stamp = rospy.Time.now()
            imu.orientation.x = quat_val[0]
            imu.orientation.y = quat_val[1]
            imu.orientation.z = quat_val[2]
            imu.orientation.w = quat_val[3]
            imu_pub.publish(imu)


        rate.sleep()

        print('[%d / %d]' % (idx, len(lidar_file_paths)))
        idx = idx+lidar_frame_step
        
        if (idx >= len(lidar_file_paths)):
            break