import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, ListedColormap
import matplotlib.ticker as ticker
from matplotlib.transforms import Bbox
import matplotlib.animation as animation
from race_utils.RaceVisualizer.track import plot_track, plot_track_3d
from race_utils.RaceGenerator.RaceTrack import RaceTrack
from race_utils.RaceVisualizer.Drawers.QuadcopterDrawer import QuadcopterDrawer
from typing import Union, Optional, Tuple
from scipy.spatial.transform import Rotation
import yaml

import os
import warnings


# plot settings
matplotlib.rc("font", **{"size": 26})
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


class RacePlotter:
    def __init__(
        self,
        traj_file: Union[os.PathLike, str, np.ndarray],
        track_file: Union[os.PathLike, str, RaceTrack],
        wpt_path: Optional[Union[os.PathLike, str]] = None,
    ):
        if isinstance(traj_file, np.ndarray):
            data_ocp = traj_file
        else:
            data_ocp = np.genfromtxt(traj_file, dtype=float, delimiter=",", names=True)

        self.track_file = track_file if isinstance(track_file, RaceTrack) else os.fspath(track_file)
        if wpt_path is not None:
            self.wpt_path = os.fspath(wpt_path)

        self.t = data_ocp["t"]
        self.p_x = data_ocp["p_x"]
        self.p_y = data_ocp["p_y"]
        self.p_z = data_ocp["p_z"]
        self.q_w = data_ocp["q_w"]
        self.q_x = data_ocp["q_x"]
        self.q_y = data_ocp["q_y"]
        self.q_z = data_ocp["q_z"]
        self.v_x = data_ocp["v_x"]
        self.v_y = data_ocp["v_y"]
        self.v_z = data_ocp["v_z"]

        ts = np.linspace(self.t[0], self.t[-1], 5000)
        ps = np.array(
            [np.interp(ts, self.t, self.p_x), np.interp(ts, self.t, self.p_y), np.interp(ts, self.t, self.p_z)]
        ).T
        vs = np.array(
            [np.interp(ts, self.t, self.v_x), np.interp(ts, self.t, self.v_y), np.interp(ts, self.t, self.v_z)]
        ).T
        vs = np.linalg.norm(vs, axis=1)
        v0, v1 = (3 * np.amin(vs) + np.amax(vs)) / 4, np.amax(vs)
        vt = np.minimum(np.maximum(vs, v0), v1)

        self.ts = ts
        self.ps = ps
        self.vs = vs
        self.vt = vt

    def estimate_tangents(self, ps: np.ndarray) -> np.ndarray:
        # compute tangents
        dp_x = np.gradient(ps[:, 0])
        dp_y = np.gradient(ps[:, 1])
        dp_z = np.gradient(ps[:, 2])
        tangents = np.vstack((dp_x, dp_y, dp_z)).T
        # normalize tangents
        tangents /= np.linalg.norm(tangents, axis=1).reshape(-1, 1)
        return tangents

    def sigmoid(self, x: np.ndarray, bias: float, inner_radius: float, outer_radius: float, rate: float) -> np.ndarray:
        return inner_radius + outer_radius * (1 / (1 + np.exp(-rate * (x - bias))))

    def plot_show(self):
        plt.show()

    def get_line_tube(self, ps: np.ndarray, tube_radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # create tube parameters
        num_points = len(ps)
        theta = np.linspace(0, 2 * np.pi, 20)
        circle_x = tube_radius * np.cos(theta)
        circle_y = tube_radius * np.sin(theta)
        tangent = self.estimate_tangents(ps)

        # initialize tube coordinates array
        tube_x = np.zeros((num_points, len(theta)))
        tube_y = np.zeros((num_points, len(theta)))
        tube_z = np.zeros((num_points, len(theta)))

        # compute tangents and normals for building tube cross-section
        for i in range(num_points):
            # choose an arbitrary vector not parallel to the tangent
            arbitrary_vector = np.array([1, 0, 0]) if not np.allclose(tangent[i], [1, 0, 0]) else np.array([0, 1, 0])

            normal = np.cross(tangent[i], arbitrary_vector)
            normal /= np.linalg.norm(normal)
            binormal = np.cross(tangent[i], normal)

            # construct orthogonal basis matrix
            TNB = np.column_stack((normal, binormal, tangent[i]))

            # for each cross-section, compute points on the circle
            for j in range(len(theta)):
                local_point = np.array([circle_x[j], circle_y[j], 0])
                global_point = ps[i] + TNB @ local_point
                tube_x[i, j] = global_point[0]
                tube_y[i, j] = global_point[1]
                tube_z[i, j] = global_point[2]

        return tube_x, tube_y, tube_z

    def get_sig_tube(
        self,
        ts: np.ndarray,
        ps: np.ndarray,
        bias: float,
        inner_radius: float,
        outer_radius: float,
        rate: float,
        scale: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.wpt_path is None:
            raise ValueError("wpt_path is not provided.")
        with open(self.wpt_path, "r") as file:
            wpt_data = yaml.safe_load(file)

        wps = np.array([wpt_data["waypoints"]]).reshape(-1, 3)
        wps_t = np.array([wpt_data["timestamps"]]).flatten()

        # search for the next waypoints
        indices = np.searchsorted(wps_t[:-1], ts, side="right").astype(int)
        dist1 = np.linalg.norm(ps - wps[indices - 1], axis=1)
        dist2 = np.linalg.norm(ps - wps[indices], axis=1)
        min_distances = np.minimum(dist1, dist2)

        tube_size = self.sigmoid(min_distances, bias, inner_radius, outer_radius, rate)  # 根据距离计算 tube 半径
        tube_size = tube_size * scale  # 缩放 tube 半径

        theta = np.linspace(0, 2 * np.pi, 20)
        tangent = self.estimate_tangents(ps)

        # initialize tube coordinates array
        num_points = len(ps)
        tube_x = np.zeros((num_points, len(theta)))
        tube_y = np.zeros((num_points, len(theta)))
        tube_z = np.zeros((num_points, len(theta)))

        # compute tangents and normals for building tube cross-section
        for i in range(num_points):
            # compute circle points
            circle_x = tube_size[i] * np.cos(theta)
            circle_y = tube_size[i] * np.sin(theta)
            # choose an arbitrary vector not parallel to the tangent
            arbitrary_vector = np.array([1, 0, 0]) if not np.allclose(tangent[i], [1, 0, 0]) else np.array([0, 1, 0])

            normal = np.cross(tangent[i], arbitrary_vector)
            normal /= np.linalg.norm(normal)
            binormal = np.cross(tangent[i], normal)

            # construct orthogonal basis matrix
            TNB = np.column_stack((normal, binormal, tangent[i]))

            # for each cross-section, compute points on the circle
            for j in range(len(theta)):
                local_point = np.array([circle_x[j], circle_y[j], 0])
                global_point = ps[i] + TNB @ local_point
                tube_x[i, j] = global_point[0]
                tube_y[i, j] = global_point[1]
                tube_z[i, j] = global_point[2]

        return tube_x, tube_y, tube_z

    def plot(
        self,
        cmap: Colormap = plt.cm.winter.reversed(),
        save_fig: bool = False,
        dpi: int = 300,
        save_path: Union[os.PathLike, str] = None,
        fig_name: Optional[str] = None,
        fig_title: Optional[str] = None,
        radius: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        margin: Optional[float] = None,
        draw_tube: bool = False,
        sig_tube: bool = False,
        tube_color: Optional[str] = None,
        alpha: float = 0.01,
        tube_rate: float = 6,
    ):
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()
        self.ax_2d = ax

        ps = self.ps
        vt = self.vt

        if draw_tube:
            if not sig_tube:
                self.plot_tube(sig_tube=sig_tube, tube_color=tube_color, alpha=alpha, tube_radius=radius)
            else:
                self.plot_tube(
                    sig_tube=sig_tube,
                    tube_color=tube_color,
                    alpha=alpha,
                    bias=1.5 * radius,
                    inner_radius=radius / 2,
                    outer_radius=1.5 * radius,
                    rate=tube_rate,
                )

        plt.scatter(ps[:, 0], ps[:, 1], s=5, c=vt, cmap=cmap)
        plt.colorbar(pad=0.01).ax.set_ylabel("Speed [m/s]")

        plot_track(plt.gca(), self.track_file, set_radius=radius, set_width=width, set_height=height, set_margin=margin)

        if fig_title is not None:
            plt.title(fig_title, fontsize=26)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.axis("equal")
        plt.grid()

        if save_fig:
            if save_path is None:
                raise ValueError("save_path is not provided.")
            os.makedirs(save_path, exist_ok=True)
            fig_name = (fig_name + ".png") if fig_name is not None else "togt_traj.png"
            plt.savefig(os.path.join(save_path, fig_name), dpi=dpi, bbox_inches="tight")

    def plot_tube(
        self,
        scale: float = 1.0,
        sig_tube: bool = False,
        tube_color: Optional[str] = None,
        alpha: float = 0.01,
        tube_edge_color: Optional[str] = None,
        tube_radius: float = 1.0,
        bias: float = 1.0,
        inner_radius: float = 0.5,
        outer_radius: float = 2.0,
        rate: float = 6,
    ):

        ax = self.ax_2d
        ts = self.ts
        ps = self.ps

        # set tube color
        if tube_color is None:
            tube_color = "purple"
        if tube_edge_color is None:
            tube_edge_color = tube_color

        # compute tube coordinates
        if not sig_tube:
            tube_x, tube_y, tube_z = self.get_line_tube(ps, tube_radius)
        else:
            tube_x, tube_y, tube_z = self.get_sig_tube(
                ts, ps, bias=bias, inner_radius=inner_radius, outer_radius=outer_radius, rate=rate, scale=scale
            )

        # plot tube
        single_color_map = ListedColormap([tube_color])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ax.pcolormesh(
                tube_x,
                tube_y,
                tube_z,
                cmap=single_color_map,
                shading="auto",
                color=tube_color,
                edgecolor="none",
                alpha=alpha,
                antialiased=True,
            )

    def plot3d(
        self,
        cmap: Colormap = plt.cm.winter.reversed(),
        save_fig: bool = False,
        dpi: int = 300,
        save_path: Union[os.PathLike, str] = None,
        fig_name: Optional[str] = None,
        fig_title: Optional[str] = None,
        radius: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        margin: Optional[float] = None,
        draw_tube: bool = False,
        sig_tube: bool = False,
        gate_color: Optional[str] = None,
        tube_color: Optional[str] = None,
        alpha: float = 0.01,
        gate_alpha: float = 0.1,
        tube_rate: float = 6,
        shade: bool = True,
    ):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes([0, 0, 1, 1], projection="3d")

        # set aspect ratio
        x_range = self.ps[:, 0].max() - self.ps[:, 0].min()
        y_range = self.ps[:, 1].max() - self.ps[:, 1].min()
        z_range = self.ps[:, 2].max() - self.ps[:, 2].min()
        max_range = max(x_range, y_range, z_range)
        min_range_factor = 0.33
        x_range = max(x_range, max_range * min_range_factor)
        y_range = max(y_range, max_range * min_range_factor)
        z_range = max(z_range, max_range * min_range_factor)
        ax.set_box_aspect((x_range, y_range, z_range))

        # compute ticks
        x_ticks_count = max(min(int(x_range), 5), 3)
        y_ticks_count = max(min(int(y_range), 5), 3)
        z_ticks_count = max(min(int(z_range), 5), 3)

        # set ticks
        self.set_nice_ticks(ax, x_range, x_ticks_count, "x")
        self.set_nice_ticks(ax, y_range, y_ticks_count, "y")
        self.set_nice_ticks(ax, z_range, z_ticks_count, "z")

        self.ax_3d = ax

        ps = self.ps
        vt = self.vt

        # draw tube
        if draw_tube:
            if not sig_tube:
                self.plot3d_tube(sig_tube=sig_tube, tube_color=tube_color, alpha=alpha, tube_radius=radius, shade=shade)
            else:
                self.plot3d_tube(
                    sig_tube=sig_tube,
                    tube_color=tube_color,
                    alpha=alpha,
                    bias=1.5 * radius,
                    inner_radius=radius / 2,
                    outer_radius=1.5 * radius,
                    rate=tube_rate,
                    shade=shade,
                )

        # plot trajectory
        sc = ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], s=5, c=vt, cmap=cmap)
        shrink_factor = min(0.8, max(0.6, 0.6 * y_range / x_range))
        colorbar_aspect = 20 * shrink_factor
        cbar = plt.colorbar(sc, shrink=shrink_factor, aspect=colorbar_aspect, pad=0.1)
        cbar.ax.set_ylabel("Speed [m/s]")

        plot_track_3d(
            plt.gca(),
            self.track_file,
            set_radius=radius,
            set_width=width,
            set_height=height,
            set_margin=margin,
            color=gate_color,
            gate_alpha=gate_alpha,
        )

        if fig_title is not None:
            plt.gcf().text(0.5, 0.95, fig_title, fontsize=26, horizontalalignment="center", verticalalignment="top")
        ax.set_xlabel("x [m]", labelpad=30 * (x_range / max_range))
        ax.set_ylabel("y [m]", labelpad=30 * (y_range / max_range))
        ax.set_zlabel("z [m]", labelpad=30 * (z_range / max_range))
        plt.axis("equal")
        plt.grid()

        if save_fig:
            if save_path is None:
                raise ValueError("save_path is not provided.")
            os.makedirs(save_path, exist_ok=True)
            fig_name = (fig_name + ".png") if fig_name is not None else "togt_traj.png"
            plt.savefig(os.path.join(save_path, fig_name), dpi=dpi, bbox_inches="tight")

    def plot3d_tube(
        self,
        scale: float = 1.0,
        sig_tube: bool = False,
        tube_color: Optional[str] = None,
        alpha: float = 0.01,
        tube_edge_color: Optional[str] = None,
        tube_radius: float = 1.0,
        bias: float = 1.0,
        inner_radius: float = 0.5,
        outer_radius: float = 2.0,
        rate: float = 6,
        shade: bool = True,
    ):

        ax = self.ax_3d
        ts = self.ts
        ps = self.ps

        # set tube color
        if tube_color is None:
            tube_color = "purple"
        if tube_edge_color is None:
            tube_edge_color = tube_color

        # compute tube coordinates
        if not sig_tube:
            tube_x, tube_y, tube_z = self.get_line_tube(ps, tube_radius)
        else:
            tube_x, tube_y, tube_z = self.get_sig_tube(
                ts, ps, bias=bias, inner_radius=inner_radius, outer_radius=outer_radius, rate=rate, scale=scale
            )

        # plot tube
        ax.plot_surface(
            tube_x,
            tube_y,
            tube_z,
            color=tube_color,
            alpha=alpha,
            edgecolor=tube_edge_color,
            shade=shade,
            antialiased=True,
        )

    def set_nice_ticks(self, ax: matplotlib.axes.Axes, range_val: float, ticks_count: int, axis: str = "x"):
        ticks_interval = range_val / (ticks_count - 1)

        # select base value for major ticks
        if range_val <= 1:
            base = round(ticks_interval / 0.1) * 0.1
        elif range_val <= 5:
            base = round(ticks_interval / 0.5) * 0.5
        elif range_val <= 10:
            base = round(ticks_interval / 1.0) * 1.0
        else:
            base = int(max(1.0, round(range_val / 5)))

        # set locator
        locator = ticker.MultipleLocator(base)

        # apply locator
        if axis == "x":
            ax.xaxis.set_major_locator(locator)
        elif axis == "y":
            ax.yaxis.set_major_locator(locator)
        else:  # 'z'
            ax.zaxis.set_major_locator(locator)

    def save_2d_fig(self, save_path: Union[os.PathLike, str], fig_name: str, dpi: int = 300):
        save_path = os.fspath(save_path)
        os.makedirs(save_path, exist_ok=True)
        if not fig_name.endswith(".png"):
            fig_name = fig_name + ".png"
        self.ax_2d.figure.savefig(os.path.join(save_path, fig_name), dpi=dpi, bbox_inches="tight")

    def save_3d_fig(
        self,
        save_path: Union[os.PathLike, str],
        fig_name: str,
        dpi: int = 300,
        hide_background: bool = False,
        hide_ground: bool = False,
    ):
        save_path = os.fspath(save_path)
        os.makedirs(save_path, exist_ok=True)
        if not fig_name.endswith(".png"):
            fig_name = fig_name + ".png"

        if hide_background:
            fig = self.ax_3d.figure
            for ax in fig.axes:
                if hasattr(ax, "orientation") or ax != self.ax_3d:
                    fig.delaxes(ax)
            fig.set_size_inches(12, 8)

            # hide all axis ticks
            self.ax_3d.set_xticks([])
            self.ax_3d.set_yticks([])
            self.ax_3d.set_zticks([])
            # hide all axis labels
            self.ax_3d.set_xlabel("")
            self.ax_3d.set_ylabel("")
            self.ax_3d.set_zlabel("")

            if not hide_ground:
                # draw the ground
                x_min, x_max = self.ax_3d.get_xlim()
                y_min, y_max = self.ax_3d.get_ylim()
                z_min, z_max = self.ax_3d.get_zlim()
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                x_range = x_max - x_min
                y_range = y_max - y_min
                z_range = z_max - z_min
                # scale the ground
                scale_factor = 1.5
                x_min_ground = x_center - (x_range * scale_factor / 2)
                x_max_ground = x_center + (x_range * scale_factor / 2)
                y_min_ground = y_center - (y_range * scale_factor / 2)
                y_max_ground = y_center + (y_range * scale_factor / 2)

                xx, yy = np.meshgrid(
                    np.linspace(x_min_ground, x_max_ground, 15), np.linspace(y_min_ground, y_max_ground, 15)
                )
                zz = np.ones_like(xx) * z_min
                self.ax_3d.plot_wireframe(xx, yy, zz, color="gray", alpha=0.5, linewidth=1.0)

                self.ax_3d.set_xlim(x_min_ground, x_max_ground)
                self.ax_3d.set_ylim(y_min_ground, y_max_ground)
                self.ax_3d.set_box_aspect([x_range * scale_factor, y_range * scale_factor, z_range])

            # set background color to empty
            self.ax_3d.xaxis.pane.fill = False
            self.ax_3d.yaxis.pane.fill = False
            self.ax_3d.zaxis.pane.fill = False
            # hide pane lines
            self.ax_3d.xaxis.pane.set_edgecolor("none")
            self.ax_3d.yaxis.pane.set_edgecolor("none")
            self.ax_3d.zaxis.pane.set_edgecolor("none")

            self.ax_3d.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            self.ax_3d.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            self.ax_3d.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

            if hide_ground:
                bbox = Bbox.from_bounds(1.15, 2.05, 7.5, 3.6)
                # bbox = Bbox.from_bounds(2.2, 2.6, 6.0, 3.0)
            else:
                bbox = Bbox.from_bounds(1.05, 1.45, 8, 4)
        else:
            bbox = "tight"

        self.ax_3d.figure.savefig(os.path.join(save_path, fig_name), dpi=dpi, bbox_inches=bbox)

    def create_animation(
        self,
        save_path: Union[os.PathLike, str] = None,
        fps: int = 20,
        dpi: int = 200,
        drone_kwargs: dict = {},
        cmap: Colormap = plt.cm.winter.reversed(),
        traj_history: float = 0.0,
        track_kargs: dict = {},
        follow_drone: bool = False,
    ) -> animation.FuncAnimation:
        """Create a 3D animation of the drone trajectory.

        Parameters
        ----------
        positions : np.ndarray
            Array of shape (time_steps, 3) representing the positions.
        attitudes : np.ndarray, optional
            Array of shape (time_steps, 3) or (time_steps, 4) representing the attitudes.
            Can be Euler angles or quaternions. If None, zero attitude is used.
        save_path : str, optional
            Path to save the animation. If None, the animation is displayed.
        fps : int, optional
            Frames per second for the animation, by default 20.
        dpi : int, optional
            Dots per inch for saving the animation, by default 200.

        Returns
        -------
        animation : matplotlib.animation.FuncAnimation
            The created animation object.
        """
        # initialize the quadcopter drawer
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        self.ani_ax = ax

        # compute the plot limits
        x_min, x_max = (
            np.min(self.ps[:, 0]),
            np.max(self.ps[:, 0]),
        )
        y_min, y_max = np.min(self.ps[:, 1]), np.max(self.ps[:, 1])
        z_min, z_max = np.min(self.ps[:, 2]), np.max(self.ps[:, 2])
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        min_range_factor = 0.33
        # update the limits
        x_range = max(x_max - x_min, max_range * min_range_factor)
        y_range = max(y_max - y_min, max_range * min_range_factor)
        z_range = max(z_max - z_min, max_range * min_range_factor)
        x_max, x_min = max(x_max, (x_max + x_min + x_range) / 2), min(x_min, (x_max + x_min - x_range) / 2)
        y_max, y_min = max(y_max, (y_max + y_min + y_range) / 2), min(y_min, (y_max + y_min - y_range) / 2)
        z_max, z_min = max(z_max, (z_max + z_min + z_range) / 2), min(z_min, (z_max + z_min - z_range) / 2)

        # compute ticks
        x_ticks_count = max(min(int(x_range), 5), 3)
        y_ticks_count = max(min(int(y_range), 5), 3)
        z_ticks_count = max(min(int(z_range), 5), 3)

        # create the quadcopter drawer
        arm_length = max(max_range / 30, 0.1)
        self.quadcopter_drawer = QuadcopterDrawer(ax=ax, arm_length=arm_length, **drone_kwargs)

        # ensure positions in a right shape
        total_frames = int((self.t[-1] - self.t[0]) * fps)
        frame_history = int(fps * traj_history)
        times = np.linspace(self.t[0], self.t[-1], total_frames)
        positions = np.array(
            [np.interp(times, self.t, self.p_x), np.interp(times, self.t, self.p_y), np.interp(times, self.t, self.p_z)]
        ).T
        attitudes = np.array(
            [
                np.interp(times, self.t, self.q_x),
                np.interp(times, self.t, self.q_y),
                np.interp(times, self.t, self.q_z),
                np.interp(times, self.t, self.q_w),
            ]
        ).T
        vt = np.interp(times, self.ts, self.vt)
        vt_min, vt_max = self.vt.min(), self.vt.max()
        vt_norm = (vt - vt_min) / (vt_max - vt_min) if vt_max > vt_min else 0.5
        time_steps = positions.shape[0]

        # create lines for the drone
        quad_artists = []
        lines = []

        # plot the track
        plot_track_3d(
            plt.gca(),
            self.track_file,
            **track_kargs,
        )

        # set the figure plots
        # set the limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        # set the aspect ratio
        ax.set_box_aspect((x_range, y_range, z_range))
        # set ticks
        self.set_nice_ticks(ax, x_range, x_ticks_count, "x")
        self.set_nice_ticks(ax, y_range, y_ticks_count, "y")
        self.set_nice_ticks(ax, z_range, z_ticks_count, "z")
        # set the aspect label
        ax.set_xlabel("x [m]", labelpad=30 * (x_range / max_range))
        ax.set_ylabel("y [m]", labelpad=30 * (y_range / max_range))
        ax.set_zlabel("z [m]", labelpad=30 * (z_range / max_range))
        # set the colorbar
        norm = plt.Normalize(vt_min, vt_max)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        shrink_factor = min(0.8, max(0.6, 0.6 * y_range / x_range))
        colorbar_aspect = 20 * shrink_factor
        cbar = fig.colorbar(sm, ax=ax, shrink=shrink_factor, aspect=colorbar_aspect, pad=0.1)
        cbar.ax.set_ylabel("Speed [m/s]")

        # the initialization function
        def init():
            for line in lines:
                line.remove() if line in ax.lines else None
            lines.clear()

            # clear previous quadcopter artists
            for artist in quad_artists:
                if isinstance(artist, list):
                    for a in artist:
                        try:
                            a.remove()
                        except:
                            pass
                else:
                    try:
                        artist.remove()
                    except:
                        pass
            quad_artists.clear()
            return []

        # the update function for each frame
        def update(frame):
            # draw the trajectory
            if frame > 0:
                if frame_history > 0:
                    # clear previous lines
                    for line in lines:
                        line.remove() if line in ax.lines else None
                    lines.clear()

                    # compute the start and end index for the trajectory
                    start_idx = max(0, frame - frame_history)
                    end_idx = frame

                    # draw the trajectory
                    for i in range(start_idx, end_idx):
                        color = cmap(vt_norm[i])
                        (segment,) = ax.plot(
                            positions[i : i + 2, 0],
                            positions[i : i + 2, 1],
                            positions[i : i + 2, 2],
                            color=color,
                            linewidth=2,
                        )
                        lines.append(segment)
                else:
                    # only update the last segment
                    if frame > 1 and hasattr(update, "last_frame"):
                        start_idx = update.last_frame
                        end_idx = frame

                        # compute the average velocity for the segment
                        avg_velocity = np.mean(vt_norm[start_idx : end_idx + 1])
                        color = cmap(avg_velocity)

                        # draw the segment
                        (segment,) = ax.plot(
                            positions[start_idx : end_idx + 1, 0],
                            positions[start_idx : end_idx + 1, 1],
                            positions[start_idx : end_idx + 1, 2],
                            color=color,
                            linewidth=2,
                        )
                        lines.append(segment)
                    else:
                        # draw the first segment
                        color = cmap(vt_norm[0])
                        (segment,) = ax.plot(
                            positions[0 : frame + 1, 0],
                            positions[0 : frame + 1, 1],
                            positions[0 : frame + 1, 2],
                            color=color,
                            linewidth=2,
                        )
                        lines.append(segment)

                    # update the last frame
                    update.last_frame = frame

            # clear previous quadcopter artists
            for artist in quad_artists:
                try:
                    artist.remove()
                except:
                    pass
            quad_artists.clear()

            # draw the quadcopter
            position = positions[frame]
            attitude = attitudes[frame]

            # draw the quadcopter
            artists = self.quadcopter_drawer.draw(position=position, attitude=attitude)
            quad_artists.extend(artists)

            # support follow camera
            if follow_drone:
                R = Rotation.from_quat(attitude).as_matrix()

                # compute forward vector
                forward_vector = R @ np.array([1, 0, 0])  # x axis forward

                # set the view range
                x_view_range = arm_length * 8
                y_view_range = arm_length * 8
                z_view_range = arm_length * 8

                # compute the target azimuth and elevation angles
                target_azim = (-np.degrees(np.arctan2(forward_vector[1], forward_vector[0])) + 90) % 360
                horizontal_distance = np.sqrt(forward_vector[0] ** 2 + forward_vector[1] ** 2)
                target_elev = -np.degrees(np.arctan2(forward_vector[2], horizontal_distance)) / 3 + 30

                # smooth the camera angles
                if not hasattr(update, "current_azim"):
                    update.current_azim = target_azim
                    update.current_elev = target_elev
                else:
                    smooth_factor = 0.15  # smooth factor, 0 < smooth_factor < 1, smaller value means smoother

                    # prevent azimuth angle from jumping
                    azim_diff = target_azim - update.current_azim
                    if azim_diff > 180:
                        azim_diff -= 360
                    elif azim_diff < -180:
                        azim_diff += 360

                    update.current_azim = (update.current_azim + smooth_factor * azim_diff) % 360
                    update.current_elev = update.current_elev + smooth_factor * (target_elev - update.current_elev)

                # set the smooth azimuth and elevation angles
                ax.view_init(elev=update.current_elev, azim=update.current_azim)

                # set the camera limits
                ax.set_xlim(position[0] - x_view_range / 2, position[0] + x_view_range / 2)
                ax.set_ylim(position[1] - y_view_range / 2, position[1] + y_view_range / 2)
                ax.set_zlim(position[2] - z_view_range / 2, position[2] + z_view_range / 2)
                # set the aspect ratio
                ax.set_box_aspect((x_view_range, y_view_range, z_view_range))
                # set ticks
                x_view_ticks_count = max(min(int(x_view_range), 5), 3)
                y_view_ticks_count = max(min(int(y_view_range), 5), 3)
                z_view_ticks_count = max(min(int(z_view_range), 5), 3)
                self.set_nice_ticks(ax, x_view_range, x_view_ticks_count, "x")
                self.set_nice_ticks(ax, y_view_range, y_view_ticks_count, "y")
                self.set_nice_ticks(ax, z_view_range, z_view_ticks_count, "z")

            # collect all artists
            all_artists = lines.copy()
            all_artists.extend(artists)
            return all_artists

        # create the animation
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=time_steps,
            init_func=init,
            blit=False,
            interval=1000 / fps,
        )

        # save the animation if a path is provided
        if save_path is not None:
            ani.save(save_path, writer="ffmpeg", fps=fps, dpi=dpi)
            print(f"The animation is saved at {save_path}")

        return ani
