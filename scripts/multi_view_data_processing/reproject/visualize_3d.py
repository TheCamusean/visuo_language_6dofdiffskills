import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_homogeneous_matrix(rotation_matrix, translation_vector):
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = translation_vector
    return homogeneous_matrix

# Generate a random rotation matrix and translation vector
rotation_matrix = np.random.rand(3, 3)
translation_vector = np.random.rand(3)

# Create the homogeneous transformation matrix
homogeneous_matrix = create_homogeneous_matrix(rotation_matrix, translation_vector)

# Define the coordinates of the base frame
base_frame_coords = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

# Transform the base frame coordinates using the homogeneous matrix
transformed_coords = np.dot(homogeneous_matrix, base_frame_coords.T).T

# Extract the transformed x, y, and z coordinates

t_coords = base_frame_coords[:, -1]
x_coords = base_frame_coords[:, 0] +t_coords
y_coords = base_frame_coords[:, 1] +t_coords
z_coords = base_frame_coords[:, 2] +t_coords

# Visualize the transformed frame
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



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



generate_base(ax, base_frame_coords)
generate_base(ax, transformed_coords)


# Set axis labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Add a legend and title
ax.legend()
plt.title('3D Homogeneous Transformation')

# Display the plot
plt.show()
