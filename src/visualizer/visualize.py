import matplotlib.pyplot as plt
import numpy as np
import torch
from theseus.geometry import SO3
from matplotlib.colors import Normalize
from ..metrics import so3 as lie_metrics

def visualize_so3_probabilities(rotations,
                                probabilities=None,
                                rotations_gt=None,
                                ax=None,
                                fig=None,
                                display_threshold_probability=0,
                                to_image=True,
                                show_color_wheel=True,
                                canonical_rotation=np.eye(3)):
    """Plot a single distribution on SO(3) using the tilt-colored method.

    Args:
        rotations: [N, 3, 3] tensor of rotation matrices
        probabilities: [N] tensor of probabilities
        rotations_gt: [N_gt, 3, 3] or [3, 3] ground truth rotation matrices
        ax: The matplotlib.pyplot.axis object to paint
        fig: The matplotlib.pyplot.figure object to paint
        display_threshold_probability: The probability threshold below which to omit
        the marker
        to_image: If True, return a tensor containing the pixels of the finished
        figure; if False return the figure itself
        show_color_wheel: If True, display the explanatory color wheel which matches
        color on the plot with tilt angle
        canonical_rotation: A [3, 3] rotation matrix representing the 'display
        rotation', to change the view of the distribution.  It rotates the
        canonical axes so that the view of SO(3) on the plot is different, which
        can help obtain a more informative view.

    Returns:
        A matplotlib.pyplot.figure object, or a tensor of pixels if to_image=True.
    """
    # def _show_single_marker(ax, rotation, marker, edgecolors=True, facecolors=False):
    #     eulers = tfg.euler.from_rotation_matrix(rotation)
    #     xyz = rotation[:, 0]
    #     tilt_angle = eulers[0]
    #     longitude = np.arctan2(xyz[0], -xyz[1])
    #     latitude = np.arcsin(xyz[2])

    #     color = cmap(0.5 + tilt_angle / 2 / np.pi)
    #     ax.scatter(longitude, latitude, s=2500,
    #             edgecolors=color if edgecolors else 'none',
    #             facecolors=facecolors if facecolors else 'none',
    #             marker=marker,
    #             linewidth=4)

    if fig is None:
        fig = plt.figure(figsize=(8, 4), dpi=100)
        
    # ax = fig.add_subplot(111, projection='mollweide')
    if ax is None:
        ax = fig.add_subplot(111, projection='mollweide')  # Create once
    ax.clear()  # Clear the plot without creating new instances
    ax.set_position([0.4, 0.1, 0.5, 0.75])

    n = len(rotations)
    z_angles = lie_metrics.get_euler_angle(torch.tensor(rotations)).numpy()
    norm = Normalize(vmin=-np.pi, vmax=np.pi)

    canonical_rotation = np.float32([[0.9605305, -0.1947092,  0.1986693],
        [0.2333919,  0.9526891, -0.1947092],
        [-0.1513585,  0.2333919,  0.9605305]])

    display_rotations = rotations @ canonical_rotation
    # so3 = SO3(tensor = torch.tensor(display_rotations))
    cmap = plt.cm.hsv
    colors = cmap(norm(z_angles))
    scatterpoint_scaling = 1
    # euler_angles = so3.to_euler(order='ZYX')
    # print(euler_angles.shape)
    # eulers_queries = tfg.euler.from_rotation_matrix(display_rotations)
    xyz = display_rotations[:, :, 0]
    # tilt_angles = eulers_queries[:, 0]

    longitudes = np.arctan2(xyz[:, 0], -xyz[:, 1])
    latitudes = np.arcsin(np.clip(xyz[:, 2], -1, 1))
    # print(xyz[:, 2])

    probabilities = np.ones(n)

    which_to_display = (probabilities > display_threshold_probability)

    # if rotations_gt is not None:
    #     # The visualization is more comprehensible if the GT
    #     # rotation markers are behind the output with white filling the interior.
    #     display_rotations_gt = rotations_gt @ canonical_rotation

    #     for rotation in display_rotations_gt:
    #         (ax, rotation, 'o')
    #     # Cover up the centers with white markers
    #     for rotation in display_rotations_gt:
    #         _show_single_marker(ax, rotation, 'o', edgecolors=False,
    #                         facecolors='#ffffff')

    # Display the distribution
    ax.scatter(
        longitudes[which_to_display],
        latitudes[which_to_display],
        s=scatterpoint_scaling,
        c=colors)
        # c=cmap(0.5))
        # c=cmap(0.5 + tilt_angles[which_to_display] / 2. / np.pi))

    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return fig, ax