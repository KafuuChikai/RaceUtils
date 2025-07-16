import numpy as np
import yaml
from typing import List, Optional, Union

import matplotlib.pyplot as plt
from race_utils.RaceGenerator.RaceTrack import RaceTrack
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D


def rpy_to_rotation_matrix(rpy) -> np.ndarray:
    """Convert roll, pitch, yaw angles to a rotation matrix.

    Parameters
    ----------
    rpy : list or np.ndarray
        Roll, pitch, yaw angles in degrees.

    Returns
    -------
    np.ndarray
        A 3x3 rotation matrix.

    """
    r, p, y = np.array(rpy) * np.pi / 180
    cos, sin = np.cos, np.sin
    R = np.eye(3)
    R[0, 0] = cos(y) * cos(p)
    R[1, 0] = sin(y) * cos(p)
    R[2, 0] = -sin(p)
    R[0, 1] = cos(y) * sin(p) * sin(r) - sin(y) * cos(r)
    R[1, 1] = sin(y) * sin(p) * sin(r) + cos(y) * cos(r)
    R[2, 1] = cos(p) * sin(r)
    R[0, 2] = cos(y) * sin(p) * cos(r) + sin(y) * sin(r)
    R[1, 2] = sin(y) * sin(p) * cos(r) - cos(y) * sin(r)
    R[2, 2] = cos(p) * cos(r)
    return R


def plot_track(
    ax: plt.Axes,
    track_file: Union[str, RaceTrack],
    radius: Optional[float] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
    margin: float = 0,
):
    """Plot the race track on a 2D axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on.
    track_file : str or RaceTrack
        The path to the track file or a RaceTrack object.
    radius : float, optional
        The radius for the waypoints. If None, use the track's radius.
    width : float, optional
        The width for the waypoints. If None, use the track's width.
    height : float, optional
        The height for the waypoints. If None, use the track's height.
    margin : float, optional
        The margin to apply to the waypoints. Default is 0.

    """
    track = (
        track_file.to_dict()
        if isinstance(track_file, RaceTrack)
        else yaml.safe_load(open(track_file).read())
    )

    for g in track["orders"]:
        g = track[g]

        if g["type"] == "FreeCorridor":
            raise NotImplementedError("Not support plotting FreeCorridor gate")

        args = {"linewidth": 3.0, "markersize": 5.0, "color": "black"}

        if g["type"] == "SingleBall":
            position = g["position"]
            r = g["radius"] - g["margin"]
            if radius is not None:
                r = radius - margin
            a = np.linspace(0, 2 * np.pi)
            ax.plot(
                position[0] + r * np.cos(a), position[1] + r * np.sin(a), "-", **args
            )
            # draw center
            ax.scatter(position[0], position[1], color="black", s=50)

        elif g["type"] == "TrianglePrisma":
            position = g["position"]
            R = rpy_to_rotation_matrix(g["rpy"])
            hw = 0.5 * (g["width"] - g["margin"])
            hh = 0.5 * (g["height"] - g["margin"])
            if width is not None:
                hw = 0.5 * (width - margin)
            if height is not None:
                hh = 0.5 * (height - margin)
            drift = 0.0
            verts = [
                [-hh, hw, drift],
                [hh, 0.0, drift],
                [-hh, -hw, drift],
                [-hh, hw, drift],
            ] @ R.T + np.array(position).reshape((1, 3))
            ax.plot(verts[:, 0], verts[:, 1], "o-", **args)

        elif g["type"] == "RectanglePrisma":
            position = g["position"]
            R = rpy_to_rotation_matrix(g["rpy"])
            hw = 0.5 * (g["width"] - g["marginW"])
            hh = 0.5 * (g["height"] - g["marginH"])
            if width is not None:
                hw = 0.5 * (width - margin)
            if height is not None:
                hh = 0.5 * (height - margin)
            drift = 0.0
            verts = [
                [-hh, hw, drift],
                [-hh, -hw, drift],
                [hh, -hw, drift],
                [hh, hw, drift],
                [-hh, hw, drift],
            ] @ R.T + np.array(position).reshape((1, 3))
            ax.plot(verts[:, 0], verts[:, 1], "o-", **args)

        elif g["type"] == "PentagonPrisma":
            position = g["position"]
            R = rpy_to_rotation_matrix(g["rpy"])
            ar = g["radius"] - g["margin"]
            if radius is not None:
                ar = radius - margin
            cos54 = np.cos(0.3 * np.pi)
            sin54 = np.sin(0.3 * np.pi)
            nd, on = ar * cos54, ar * sin54
            bc = 2 * nd
            fc = bc * sin54
            of = ar - bc * cos54
            drift = 0.0
            verts = [
                [-on, nd, drift],
                [of, fc, drift],
                [ar, 0.0, drift],
                [of, -fc, drift],
                [-on, -nd, drift],
                [-on, nd, drift],
            ] @ R.T + np.array(position).reshape((1, 3))
            ax.plot(verts[:, 0], verts[:, 1], "o-", **args)

        elif g["type"] == "HexagonPrisma":
            position = g["position"]
            R = rpy_to_rotation_matrix(g["rpy"])
            aside = g["side"] - g["margin"]
            if radius is not None:
                aside = radius - margin
            hside = 0.5 * aside
            height = hside * np.tan(np.pi / 3.0)
            drift = 0.0
            verts = [
                [-height, hside, drift],
                [0.0, aside, drift],
                [height, hside, drift],
                [height, -hside, drift],
                [0.0, -aside, drift],
                [-height, -hside, drift],
                [-height, hside, drift],
            ] @ R.T + np.array(position).reshape((1, 3))
            ax.plot(verts[:, 0], verts[:, 1], "o-", **args)

        else:
            raise ValueError("Unrecognized gate: " + g["type"])


def plot_track_3d(
    ax: Axes3D,
    track_file: Union[str, RaceTrack],
    radius: Optional[float] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
    margin: float = 0,
    gate_color: Optional[str] = None,
    gate_alpha: float = 0.1,
):
    """Plot the race track on a 3D axis.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.Axes3D
        The 3D axis to plot on.
    track_file : str or RaceTrack
        The path to the track file or a RaceTrack object.
    radius : float, optional
        The radius for the waypoints. If None, use the track's radius.
    width : float, optional
        The width for the waypoints. If None, use the track's width.
    height : float, optional
        The height for the waypoints. If None, use the track's height.
    margin : float, optional
        The margin to apply to the waypoints. Default is 0.
    gate_color : str, optional
        The color of the gates. Default is "r" (red).
    gate_alpha : float, optional
        The alpha transparency of the gates. Default is 0.1.

    """
    if gate_color is None:
        gate_color = "r"

    track = (
        track_file.to_dict()
        if isinstance(track_file, RaceTrack)
        else yaml.safe_load(open(track_file).read())
    )

    for g in track["orders"]:
        g = track[g]
        plot_gate_3d(
            ax,
            g,
            radius=radius,
            width=width,
            height=height,
            margin=margin,
            gate_color=gate_color,
            gate_alpha=gate_alpha,
        )


def plot_gate_3d(
    ax: Axes3D,
    gates: Union[dict, List[dict]],
    radius: Optional[float] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
    margin: float = 0,
    gate_color: Optional[str] = None,
    gate_alpha: float = 0.1,
):
    """Plot a single gate in 3D.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axis to plot on.
    g : dict
        The gate dictionary containing the gate type and parameters.
    radius : float, optional
        The radius for the gate, if applicable. If None, use the gate's radius.
    width : float, optional
        The width for the gate, if applicable. If None, use the gate's width.
    height : float, optional
        The height for the gate, if applicable. If None, use the gate's height.
    margin : float, optional
        The margin to apply to the gate dimensions. Default is 0.
    gate_color : str, optional
        The color of the gate. Default is "r" (red).
    gate_alpha : float, optional
        The alpha transparency of the gate. Default is 0.1.

    Returns
    -------
    artists : list
        A list of artists created during the plotting.
    """

    if gate_color is None:
        gate_color = "r"

    artists = []

    if isinstance(gates, dict):
        gates = [gates]

    for g in gates:
        if g["type"] == "FreeCorridor":
            raise NotImplementedError("Not support plotting FreeCorridor gate")

        args = {"linewidth": 9.0, "markersize": 15.0, "color": "gray"}

        if g["type"] == "SingleBall":
            position = g["position"]
            r = g["radius"] - g["margin"]
            if radius is not None:
                r = radius - margin

            # create sphere
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
            x = position[0] + r * np.cos(u) * np.sin(v)
            y = position[1] + r * np.sin(u) * np.sin(v)
            z = position[2] + r * np.cos(v)
            # draw sphere
            surface = ax.plot_surface(
                x, y, z, color=gate_color, alpha=gate_alpha, edgecolor="none"
            )
            artists.append(surface)
            # draw center
            scatter = ax.scatter(
                position[0], position[1], position[2], color="black", s=50
            )
            artists.append(scatter)

        elif g["type"] == "TrianglePrisma":
            position = g["position"]
            R = rpy_to_rotation_matrix(g["rpy"])
            hw = 0.5 * (g["width"] - g["margin"])
            hh = 0.5 * (g["height"] - g["margin"])
            if width is not None:
                hw = 0.5 * (width - margin)
            if height is not None:
                hh = 0.5 * (height - margin)
            drift = 0.0
            verts = [
                [-hh, hw, drift],
                [hh, 0.0, drift],
                [-hh, -hw, drift],
                [-hh, hw, drift],
            ] @ R.T + np.array(position).reshape((1, 3))
            line = ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], "o-", **args)
            artists.append(line)

        elif g["type"] == "RectanglePrisma":
            position = g["position"]
            R = rpy_to_rotation_matrix(g["rpy"])
            hw = 0.5 * (g["width"] - g["marginW"])
            hh = 0.5 * (g["height"] - g["marginH"])
            if width is not None:
                hw = 0.5 * (width - margin)
            if height is not None:
                hh = 0.5 * (height - margin)
            drift = 0.0
            thickness = 0.1 * min(hw, hh)

            front_outer_verts = np.array(
                [
                    [-hh, hw, -thickness / 2],
                    [-hh, -hw, -thickness / 2],
                    [hh, -hw, -thickness / 2],
                    [hh, hw, -thickness / 2],
                ]
            ) @ R.T + np.array(position).reshape((1, 3))

            inner_scale = 0.8
            front_inner_verts = np.array(
                [
                    [-hh * inner_scale, hw * inner_scale, -thickness / 2],
                    [-hh * inner_scale, -hw * inner_scale, -thickness / 2],
                    [hh * inner_scale, -hw * inner_scale, -thickness / 2],
                    [hh * inner_scale, hw * inner_scale, -thickness / 2],
                ]
            ) @ R.T + np.array(position).reshape((1, 3))

            back_outer_verts = np.array(
                [
                    [-hh, hw, thickness / 2],
                    [-hh, -hw, thickness / 2],
                    [hh, -hw, thickness / 2],
                    [hh, hw, thickness / 2],
                ]
            ) @ R.T + np.array(position).reshape((1, 3))

            back_inner_verts = np.array(
                [
                    [-hh * inner_scale, hw * inner_scale, thickness / 2],
                    [-hh * inner_scale, -hw * inner_scale, thickness / 2],
                    [hh * inner_scale, -hw * inner_scale, thickness / 2],
                    [hh * inner_scale, hw * inner_scale, thickness / 2],
                ]
            ) @ R.T + np.array(position).reshape((1, 3))

            faces = []

            front_ring = [
                [
                    front_outer_verts[0],
                    front_outer_verts[1],
                    front_inner_verts[1],
                    front_inner_verts[0],
                ],  # 左侧
                [
                    front_outer_verts[1],
                    front_outer_verts[2],
                    front_inner_verts[2],
                    front_inner_verts[1],
                ],  # 底部
                [
                    front_outer_verts[2],
                    front_outer_verts[3],
                    front_inner_verts[3],
                    front_inner_verts[2],
                ],  # 右侧
                [
                    front_outer_verts[3],
                    front_outer_verts[0],
                    front_inner_verts[0],
                    front_inner_verts[3],
                ],  # 顶部
            ]
            faces.extend(front_ring)

            back_ring = [
                [
                    back_outer_verts[0],
                    back_outer_verts[1],
                    back_inner_verts[1],
                    back_inner_verts[0],
                ],  # 左侧
                [
                    back_outer_verts[1],
                    back_outer_verts[2],
                    back_inner_verts[2],
                    back_inner_verts[1],
                ],  # 底部
                [
                    back_outer_verts[2],
                    back_outer_verts[3],
                    back_inner_verts[3],
                    back_inner_verts[2],
                ],  # 右侧
                [
                    back_outer_verts[3],
                    back_outer_verts[0],
                    back_inner_verts[0],
                    back_inner_verts[3],
                ],  # 顶部
            ]
            faces.extend(back_ring)

            outer_sides = [
                [
                    front_outer_verts[0],
                    front_outer_verts[1],
                    back_outer_verts[1],
                    back_outer_verts[0],
                ],  # 左外侧
                [
                    front_outer_verts[1],
                    front_outer_verts[2],
                    back_outer_verts[2],
                    back_outer_verts[1],
                ],  # 底外侧
                [
                    front_outer_verts[2],
                    front_outer_verts[3],
                    back_outer_verts[3],
                    back_outer_verts[2],
                ],  # 右外侧
                [
                    front_outer_verts[3],
                    front_outer_verts[0],
                    back_outer_verts[0],
                    back_outer_verts[3],
                ],  # 顶外侧
            ]
            faces.extend(outer_sides)

            inner_sides = [
                [
                    front_inner_verts[0],
                    front_inner_verts[1],
                    back_inner_verts[1],
                    back_inner_verts[0],
                ],  # 左内侧
                [
                    front_inner_verts[1],
                    front_inner_verts[2],
                    back_inner_verts[2],
                    back_inner_verts[1],
                ],  # 底内侧
                [
                    front_inner_verts[2],
                    front_inner_verts[3],
                    back_inner_verts[3],
                    back_inner_verts[2],
                ],  # 右内侧
                [
                    front_inner_verts[3],
                    front_inner_verts[0],
                    back_inner_verts[0],
                    back_inner_verts[3],
                ],  # 顶内侧
            ]
            faces.extend(inner_sides)

            poly = Poly3DCollection(
                faces,
                facecolors=gate_color,
                alpha=gate_alpha,
                edgecolors="gray",
                linewidths=1.0,
            )
            ax.add_collection3d(poly)
            artists.append(poly)

        elif g["type"] == "PentagonPrisma":
            position = g["position"]
            R = rpy_to_rotation_matrix(g["rpy"])
            ar = g["radius"] - g["margin"]
            if radius is not None:
                ar = radius - margin
            cos54 = np.cos(0.3 * np.pi)
            sin54 = np.sin(0.3 * np.pi)
            nd, on = ar * cos54, ar * sin54
            bc = 2 * nd
            fc = bc * sin54
            of = ar - bc * cos54
            drift = 0.0
            verts = [
                [-on, nd, drift],
                [of, fc, drift],
                [ar, 0.0, drift],
                [of, -fc, drift],
                [-on, -nd, drift],
                [-on, nd, drift],
            ] @ R.T + np.array(position).reshape((1, 3))
            line = ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], "o-", **args)
            artists.append(line)

        elif g["type"] == "HexagonPrisma":
            position = g["position"]
            R = rpy_to_rotation_matrix(g["rpy"])
            aside = g["side"] - g["margin"]
            if radius is not None:
                aside = radius - margin
            hside = 0.5 * aside
            height = hside * np.tan(np.pi / 3.0)
            drift = 0.0
            verts = [
                [-height, hside, drift],
                [0.0, aside, drift],
                [height, hside, drift],
                [height, -hside, drift],
                [0.0, -aside, drift],
                [-height, -hside, drift],
                [-height, hside, drift],
            ] @ R.T + np.array(position).reshape((1, 3))
            line = ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], "o-", **args)
            artists.append(line)

        else:
            raise ValueError("Unrecognized gate: " + g["type"])

    return artists
