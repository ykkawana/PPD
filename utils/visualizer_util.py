import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

import plotly.graph_objects as go
import seaborn as sns
import io
from loss import chamfer_loss
from utils import common
from collections import abc
import trimesh
import os
import io
import matplotlib.pyplot as plt
import time
from concurrent.futures.process import ProcessPoolExecutor


class Visualizer:
    def __init__(self, class_num, marker_size=3, parallel=False):
        self.class_num = class_num
        self.marker_size = marker_size
        self.color = Visualizer.get_colorpalette('hls', self.class_num)
        self.parallel = parallel

    @staticmethod
    def get_colorpalette(colorpalette, n_colors):
        palette = sns.color_palette(colorpalette, n_colors)
        rgb = [
            'rgb({},{},{})'.format(*[x * 256 for x in rgb]) for rgb in palette
        ]
        return rgb

    def visualize_pointcloud(self,
                             points,
                             indices,
                             rotation_direction=None,
                             rotation_anchor_point=None,
                             primitive_type=None):
        assert points.ndim in [2, 3], (points.shape, indices.shape)
        assert points.shape[-1] in [2, 3], (points.shape, indices.shape)
        assert points.shape[-2] == indices.shape[-1], (points.shape,
                                                       indices.shape)
        assert points.shape[:-1] == indices.shape, (points.shape,
                                                    indices.shape)
        assert indices.max() <= self.class_num, (indices.max(), self.class_num)

        is_visualize_rotation_axis = (rotation_direction is not None
                                      and rotation_anchor_point is not None
                                      and primitive_type is not None)
        if points.ndim == 2:
            points = points[np.newaxis, ...]
            indices = indices[np.newaxis, ...]
            if is_visualize_rotation_axis:
                rotation_direction = rotation_direction[np.newaxis, ...]
                rotation_anchor_point = rotation_anchor_point[np.newaxis, ...]
                primitive_type = primitive_type[np.newaxis, ...]

        batch = points.shape[0]
        images = []
        plotss = []
        for batch_idx in range(batch):
            plots = []
            for idx in range(self.class_num):
                batch_indices = indices[batch_idx, :] == (idx + 1)
                if batch_indices.sum() < 1:
                    continue
                if points.shape[-1] == 2:
                    plot = go.Scatter(x=points[batch_idx, batch_indices, 0],
                                      y=points[batch_idx, batch_indices, 1],
                                      mode='markers',
                                      marker=dict(size=self.marker_size,
                                                  color=self.color[idx]))
                    plots.append(plot)
                elif points.shape[-1] == 3:
                    plot = go.Scatter3d(x=points[batch_idx, batch_indices, 0],
                                        y=points[batch_idx, batch_indices, 1],
                                        z=points[batch_idx, batch_indices, 2],
                                        mode='markers',
                                        marker=dict(size=self.marker_size,
                                                    color=self.color[idx]))
                    plots.append(plot)
                    if is_visualize_rotation_axis and idx >= 1 and (
                            idx - 1) < rotation_direction.shape[1]:
                        rotation_idx = idx - 1
                        direction = rotation_direction[batch_idx, rotation_idx,
                                                       ...]
                        anchor_point = rotation_anchor_point[batch_idx,
                                                             rotation_idx, ...]
                        tip = anchor_point + direction
                        dpoints = np.stack([anchor_point, tip])
                        line_config = dict(width=10, color=self.color[idx])
                        if primitive_type[batch_idx, idx] == 2:
                            line_config[
                                'dash'] = 'dashdot'  # for prismatic joint
                        plot = go.Scatter3d(
                            x=dpoints[:, 0],
                            y=dpoints[:, 1],
                            z=dpoints[:, 2],
                            marker=dict(size=self.marker_size,
                                        color=self.color[idx]),
                            line=line_config,
                        )
                        plots.append(plot)

                else:
                    raise NotImplementedError
            plotss.append(plots)

        if self.parallel:
            with ProcessPoolExecutor(max_workers=10) as executor:
                futures = []
                for plots in plotss:
                    futures.append(executor.submit(gen_image_from_plot, plots))
            images = [f.result() for f in futures]
        else:
            images = [gen_image_from_plot(plots) for plots in plotss]
        return images


def gen_image_from_plot(plots):
    fig = go.Figure(data=plots)
    fig.update_layout(scene=dict(
        xaxis=dict(range=[-0.55, 0.55], ),
        yaxis=dict(range=[-0.55, 0.55], ),
        zaxis=dict(range=[-0.55, 0.55], ),
    ),
                      scene_aspectmode='cube')
    image_data = fig.to_image(format="jpg", engine='kaleido')
    image = Image.open(io.BytesIO(image_data))
    return image


def get_scatter_gofig(point_set, num=-1, marker_size=3, colors=None):
    assert isinstance(point_set, abc.Iterable)
    plots = []
    dim = point_set[0].shape[-1]
    assert dim in [2, 3]
    for idx, points in enumerate(point_set):
        marker = dict(size=marker_size)
        if colors is not None:
            marker.update(color=colors[idx])
        if num > 0:
            points = common.subsample_points(points, num)
        if dim == 2:
            plot = go.Scatter(x=points[:, 0],
                              y=points[:, 1],
                              mode='markers',
                              marker=marker)
        elif dim == 3:
            plot = go.Scatter3d(x=points[:, 0],
                                y=points[:, 1],
                                z=points[:, 2],
                                mode='markers',
                                marker=marker)
        plots.append(plot)
    fig = go.Figure(data=plots)
    fig.update_layout(scene=dict(
        xaxis=dict(range=[-0.55, 0.55], ),
        yaxis=dict(range=[-0.55, 0.55], ),
        zaxis=dict(range=[-0.55, 0.55], ),
    ),
                      scene_aspectmode='cube')
    return fig


def get_mesh_render_fig(mesh, return_fig=False):
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    import pyrender

    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene = pyrender.Scene(ambient_light=np.ones(3) * 0.01)
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    s = np.sqrt(2) / 2
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    Rx = trimesh.transformations.rotation_matrix(-np.pi / 8, [1, 0, 0])
    Ry = trimesh.transformations.rotation_matrix(np.pi / 4, [0, 1, 0])
    camera_pose = trimesh.transformations.concatenate_matrices(
        Ry, Rx, camera_pose)

    scene.add(camera, pose=camera_pose)

    light_pose_x = trimesh.transformations.translation_matrix([3., 0., 0.])
    light_pose_y = trimesh.transformations.translation_matrix([0., 3., 0.])
    light_pose_z = trimesh.transformations.translation_matrix([0., 0., 3.])
    light = pyrender.SpotLight(color=np.ones(3),
                               intensity=3.0,
                               innerConeAngle=np.pi / 6.0)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    scene.add(light, pose=camera_pose)
    #scene.add(light, pose=light_pose_x)
    #scene.add(light, pose=light_pose_y)
    #scene.add(light, pose=light_pose_z)

    r = pyrender.OffscreenRenderer(400, 400)
    flags = pyrender.constants.RenderFlags.OFFSCREEN
    color, depth = r.render(scene, flags=flags)
    if return_fig:
        fig = plt.figure()
        plt.axis('off')
        plt.imshow(depth, cmap=plt.cm.gray_r)
        return fig
    return color, depth


def get_pil_image_from_fig(depth):
    buf = io.BytesIO()
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(depth, cmap=plt.cm.gray_r)
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img
