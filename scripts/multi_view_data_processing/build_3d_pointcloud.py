import pickle
import numpy as np
import matplotlib.pyplot as plt


# Visualize the transformed frame
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')

O = np.array([-0.30895138, 0., 0.82001764])

ax2.scatter(O[0], O[1], O[2])


def generate_base(ax, H):
    t_coords = H[:, -1]
    x_coords = H[:, 0] + t_coords
    y_coords = H[:, 1] + t_coords
    z_coords = H[:, 2] + t_coords

    # Plot the base frame
    vx = np.concatenate((t_coords[None,:], x_coords[None,:]), axis=0)
    ax.plot(vx[:, 0], vx[:, 1], vx[:, 2], 'r-', label='X-axis')

    vy = np.concatenate((t_coords[None,:], y_coords[None,:]), axis=0)
    ax.plot(vy[:, 0], vy[:, 1], vy[:, 2],  'g-', label='Y-axis')

    vz = np.concatenate((t_coords[None,:], z_coords[None,:]), axis=0)
    ax.plot(vz[:, 0], vz[:, 1], vz[:, 2],  'b-', label='Z-axis')


def image_to_3d(img, depth, intrinsics):

    img_px = img.shape[0]
    img_py = img.shape[1]
    px = np.arange(0, img_px, 1)
    py = np.arange(0, img_py, 1)
    pxx, pyy = np.meshgrid(px, py)

    x_2d = (pxx - intrinsics[0,2])/(-intrinsics[0,0])
    x = -x_2d*depth.squeeze()

    y_2d = (pyy - intrinsics[1,2])/(-intrinsics[1,1])
    y = -y_2d*depth.squeeze()

    xyz = np.concatenate((x[...,None], y[...,None] ,depth.squeeze()[...,None]),axis=-1)
    img_3d = np.concatenate((xyz, rgb), axis=-1)
    points = img_3d.reshape(-1, 6)

    mask_far = 2.
    mask = points[:,2]<mask_far
    points = points[mask,...]
    return points



with open("reproject/0.replay", "rb") as f:
    obs = pickle.load(f)
    for i, camera in enumerate(["front", "wrist", "left_shoulder", "right_shoulder"]):
        rgb = obs["%s_rgb" % camera]
        rgb = rgb.transpose(1, 2, 0).astype(np.uint8)
        depth = obs["%s_depth" % camera]
        E = obs["%s_camera_extrinsics" % camera]
        I = obs["%s_camera_intrinsics" % camera]

        point_3d = image_to_3d(rgb, depth, I)
        point_3d[:, :3] = np.einsum('mn,bn->bm',E[:3,:3],point_3d[:,:3]) + E[:3,-1]
        ## Eliminate Floor ##
        mask_height = .8
        mask = point_3d[:, 2] > mask_height
        point_3d = point_3d[mask, ...]

        idx = np.random.randint(0, point_3d.shape[0],5000)
        ax2.scatter(point_3d[idx,0], point_3d[idx,1], point_3d[idx,2], c=point_3d[idx,3:]/255)

        gripper_pos = obs['gripper_pose'][:3]


        world_to_cam = np.linalg.inv(E)
        point = np.array([gripper_pos[0], gripper_pos[1], gripper_pos[2], 1])
        point_in_cam_frame = world_to_cam.dot(point)

        px, py, pz = point_in_cam_frame[:3]
        px = I[0, 2] - int(-I[0, 0] * (px / pz))
        py = I[1, 2] - int(-I[1, 1] * (py / pz))

        gripper_point = np.array([px, py])
        gripper_point = np.clip(gripper_point, 0, rgb.shape[0]-1)
        gripper_point = np.round(gripper_point).astype(np.int8)

        #gripper_pos_cam += rgb.shape[0]//2

        ## Place Camera ##
        #plt.show()


        generate_base(ax2, E)

        point = np.array([O[0], O[1], O[2], 1])
        point_in_cam_frame = world_to_cam.dot(point)

        px, py, pz = point_in_cam_frame[:3]
        px = I[0, 2] - int(-I[0, 0] * (px / pz))
        py = I[1, 2] - int(-I[1, 1] * (py / pz))

        origin_point = np.array([px, py])
        origin_point = np.clip(origin_point, 0, rgb.shape[0]-1)
        origin_point = np.round(origin_point).astype(np.int8)


        for k in range(10):
            rgb[k, 0, :] = [255, 255, 255]


        rgb[gripper_point[1], gripper_point[0], :] = [255, 0, 0]
        rgb[origin_point[1], origin_point[0], :] = [0, 255, 0]

    ax2.scatter(gripper_pos[0], gripper_pos[1], gripper_pos[2], c='g', s=500)

    ax2.set_xlim([-3, 3])
    ax2.set_ylim([-3, 3])
    ax2.set_zlim([-3, 3])
    plt.show()

    #plt.show()
