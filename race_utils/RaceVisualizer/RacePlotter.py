import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, ListedColormap
import matplotlib.ticker as ticker
from matplotlib.transforms import Bbox
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from race_utils.RaceVisualizer.track import plot_track, plot_track_3d, plot_gate_3d
from race_utils.RaceGenerator.RaceTrack import RaceTrack
from race_utils.RaceVisualizer.Drawers.QuadcopterDrawer import QuadcopterDrawer
from typing import Union, Optional, Tuple, List
from scipy.spatial.transform import Rotation
import yaml

import os
import warnings


# plot settings
matplotlib.rc("font", **{"size": 26})
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


class BasePlotter:
    def __init__(
        self,
        traj_file: Union[os.PathLike, str, np.ndarray],
        track_file: Union[os.PathLike, str, RaceTrack] = None,
        wpt_path: Optional[Union[os.PathLike, str]] = None,
        end_time: Optional[int] = None,
        moving_gate_data: Optional[np.ndarray] = None,
    ):
        """Initialize the BasePlotter class.

        Parameters
        ----------
        traj_file : Union[os.PathLike, str, np.ndarray]
            The trajectory file to load, which can be a path to a CSV file or a numpy array.
        track_file : Union[os.PathLike, str, RaceTrack], optional
            The track file to load, which can be a path to a YAML file or a RaceTrack object.
        wpt_path : Optional[Union[os.PathLike, str]], optional
            The path to the waypoints file, if any.
        end_time : Optional[int], optional
            The end time of the trajectory, if known.
        moving_gate_data : Optional[np.ndarray], optional
            The moving gate data, if any, as a numpy array with columns for time and position.

        """
        if isinstance(traj_file, np.ndarray):
            data_ocp = traj_file
        else:
            data_ocp = np.genfromtxt(traj_file, dtype=float, delimiter=",", names=True)

        if track_file is not None:
            self.track_file = (
                track_file
                if isinstance(track_file, RaceTrack)
                else os.fspath(track_file)
            )
        else:
            self.track_file = None
        if wpt_path is not None:
            self.wpt_path = os.fspath(wpt_path)

        self.end_time = end_time

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
            [
                np.interp(ts, self.t, self.p_x),
                np.interp(ts, self.t, self.p_y),
                np.interp(ts, self.t, self.p_z),
            ]
        ).T
        vs = np.array(
            [
                np.interp(ts, self.t, self.v_x),
                np.interp(ts, self.t, self.v_y),
                np.interp(ts, self.t, self.v_z),
            ]
        ).T
        vs = np.linalg.norm(vs, axis=1)
        v0, v1 = (3 * np.amin(vs) + np.amax(vs)) / 4, np.amax(vs)
        vt = np.minimum(np.maximum(vs, v0), v1)

        self.ts = ts
        self.ps = ps
        self.vs = vs
        self.vt = vt

        if moving_gate_data is not None:
            self.gate_t = moving_gate_data["t"][: len(data_ocp["t"])]
            self.gate_p_x = moving_gate_data["p_x"][: len(data_ocp["t"])]
            self.gate_p_y = moving_gate_data["p_y"][: len(data_ocp["t"])]
            self.gate_p_z = moving_gate_data["p_z"][: len(data_ocp["t"])]

            # Interpolate gate positions to match the high-resolution time steps 'ts'
            if self.gate_p_x.ndim == 2:
                num_gates = self.gate_p_x.shape[1]
                num_ts = len(ts)

                # Initialize the final array with shape (num_ts, num_gates, 3)
                self.moving_gate_ps = np.zeros((num_ts, num_gates, 3))

                for i in range(num_gates):
                    # Interpolate each coordinate for the i-th gate
                    self.moving_gate_ps[:, i, 0] = np.interp(
                        ts, self.gate_t, self.gate_p_x[:, i]
                    )
                    self.moving_gate_ps[:, i, 1] = np.interp(
                        ts, self.gate_t, self.gate_p_y[:, i]
                    )
                    self.moving_gate_ps[:, i, 2] = np.interp(
                        ts, self.gate_t, self.gate_p_z[:, i]
                    )
            else:
                # Fallback for 1D data (single gate case)
                self.moving_gate_ps = np.array(
                    [
                        np.interp(ts, self.gate_t, self.gate_p_x),
                        np.interp(ts, self.gate_t, self.gate_p_y),
                        np.interp(ts, self.gate_t, self.gate_p_z),
                    ]
                ).T
        else:
            self.moving_gate_ps = None

        # initialize the plot
        self._fig_2d = None
        self.ax_2d = None
        self._fig_3d = None
        self.ax_3d = None
        self._fig_ani = None
        self.ani_ax = None

    def load_track(self, track_file: Union[os.PathLike, str, RaceTrack]) -> None:
        """Load a track file into the plotter.

        Parameters
        ----------
        track_file : Union[os.PathLike, str, RaceTrack]
            The track file to load, which can be a path to a YAML file or a RaceTrack object.

        """
        self.track_file = (
            track_file if isinstance(track_file, RaceTrack) else os.fspath(track_file)
        )

    def _ensure_2d_fig_exists(self) -> Tuple[plt.Figure, plt.Axes]:
        """Ensure that the 2D figure and axes are created.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            the figure and axes for 2D plotting.

        """
        if self._fig_2d is None:
            self._fig_2d = plt.figure(figsize=(8, 6))
            self.ax_2d = self._fig_2d.add_subplot(111)
        return self._fig_2d, self.ax_2d

    def _ensure_3d_fig_exists(self) -> Tuple[plt.Figure, plt.Axes]:
        """Ensure that the 3D figure and axes are created.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            the figure and axes for 3D plotting.

        """
        if self._fig_3d is None:
            self._fig_3d = plt.figure(figsize=(12, 8))
            self.ax_3d = self._fig_3d.add_axes([0, 0, 1, 1], projection="3d")
        return self._fig_3d, self.ax_3d

    def _ensure_ani_fig_exists(self) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
        """Ensure that the animation figure and axes are created.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes, plt.Axes]
            the figure and axes for animation plotting, containing the main 3D plot and a colorbar.

        """
        if self._fig_ani is None:
            # Use a wider figure to comfortably fit the colorbar
            self._fig_ani = plt.figure(figsize=(14, 10))
            # Main 3D plot axes, leaving space on the right for the colorbar
            self.ani_ax = self._fig_ani.add_axes(
                [0.05, 0.05, 0.8, 0.9], projection="3d"
            )
            # Dedicated axes for the colorbar
            self.ani_cax = self._fig_ani.add_axes([0.88, 0.15, 0.03, 0.7])
        return self._fig_ani, self.ani_ax

    def plot_show(self) -> None:
        """Display the plot."""
        plt.show()

    def plot(self) -> None:
        """must be implemented in subclasses"""
        raise NotImplementedError("plot() must be implemented in subclasses")

    def plot3d(self) -> None:
        """must be implemented in subclasses"""
        raise NotImplementedError("plot3d() must be implemented in subclasses")

    def create_animation(self) -> None:
        """must be implemented in subclasses"""
        raise NotImplementedError(
            "create_animation() must be implemented in subclasses"
        )

    def save_2d_fig(
        self, save_path: Union[os.PathLike, str], fig_name: str, dpi: int = 300
    ) -> None:
        """Save the 2D figure to a file.

        Parameters
        ----------
        save_path : Union[os.PathLike, str]
            The directory where the figure will be saved.
        fig_name : str
            The name of the figure file.
        dpi : int, optional
            The resolution of the saved figure in dots per inch (default is 300).

        """
        save_path = os.fspath(save_path)
        os.makedirs(save_path, exist_ok=True)
        if not fig_name.endswith(".png"):
            fig_name = fig_name + ".png"
        self.ax_2d.figure.savefig(
            os.path.join(save_path, fig_name), dpi=dpi, bbox_inches="tight"
        )

    def save_3d_fig(
        self,
        save_path: Union[os.PathLike, str],
        fig_name: str,
        dpi: int = 300,
        hide_background: bool = False,
        hide_ground: bool = False,
    ) -> None:
        """Save the 3D figure to a file.

        Parameters
        ----------
        save_path : Union[os.PathLike, str]
            The directory where the figure will be saved.
        fig_name : str
            The name of the figure file.
        dpi : int, optional
            The resolution of the saved figure in dots per inch (default is 300).
        hide_background : bool, optional
            If True, hides the background of the 3D plot (default is False).
        hide_ground : bool, optional
            If True, hides the ground plane in the 3D plot (default is False).

        """
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
                    np.linspace(x_min_ground, x_max_ground, 15),
                    np.linspace(y_min_ground, y_max_ground, 15),
                )
                zz = np.ones_like(xx) * z_min
                self.ax_3d.plot_wireframe(
                    xx, yy, zz, color="gray", alpha=0.5, linewidth=1.0
                )

                self.ax_3d.set_xlim(x_min_ground, x_max_ground)
                self.ax_3d.set_ylim(y_min_ground, y_max_ground)
                self.ax_3d.set_box_aspect(
                    [x_range * scale_factor, y_range * scale_factor, z_range]
                )

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

        self.ax_3d.figure.savefig(
            os.path.join(save_path, fig_name), dpi=dpi, bbox_inches="tight"
        )

    def set_nice_ticks(
        self,
        ax: matplotlib.axes.Axes,
        range_val: float,
        ticks_count: int,
        axis: str = "x",
    ) -> None:
        """Set nice ticks on the specified axis of the plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to set the ticks.
        range_val : float
            The range of the axis to determine the tick intervals.
        ticks_count : int
            The number of ticks to display on the axis.
        axis : str, optional
            The axis to set the ticks on, can be 'x', 'y', or 'z' (default is 'x').

        """
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
        base = max(0.1, base)  # ensure base is not zero
        locator = ticker.MultipleLocator(base)

        # apply locator
        if axis == "x":
            ax.xaxis.set_major_locator(locator)
        elif axis == "y":
            ax.yaxis.set_major_locator(locator)
        else:  # 'z'
            ax.zaxis.set_major_locator(locator)


class BasePlotterList:
    """
    Class for chaining BasePlotter objects together.

    Parameters
    ----------
    plotters : list
        List of plotters to be chained together.

    """

    def __init__(self, plotters: List[BasePlotter]):
        """Initialize the BasePlotterList with a list of plotters.

        Parameters
        ----------
        plotters : List[BasePlotter]
            List of BasePlotter objects to be managed by this class.

        """
        assert isinstance(plotters, list)
        self.plotters = plotters
        self.num_plotters = len(plotters)
        self._fig_ani = None
        self.ani_ax = None
        self.ani_cax = None

    def _ensure_ani_fig_exists(self) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
        """Ensure that the animation figure and axes are created.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes, plt.Axes]
            the figure and axes for animation plotting, containing the main 3D plot and a colorbar.

        """
        if self._fig_ani is None:
            # Use a wider figure to comfortably fit the colorbar
            self._fig_ani = plt.figure(figsize=(14, 10))
            # Main 3D plot axes, leaving space on the right for the colorbar
            self.ani_ax = self._fig_ani.add_axes(
                [0.05, 0.05, 0.8, 0.9], projection="3d"
            )
            # Dedicated axes for the colorbar
            self.ani_cax = self._fig_ani.add_axes([0.88, 0.15, 0.03, 0.7])
        return self._fig_ani, self.ani_ax, self.ani_cax

    def load_track(
        self,
        track_file: Union[os.PathLike, str, RaceTrack],
        index: Optional[list] = None,
        plot_track_once: bool = False,
    ) -> None:
        """Load a track file into the plotters.

        Parameters
        ----------
        track_file : Union[os.PathLike, str, RaceTrack]
            The track file to load, which can be a path to a YAML file or a RaceTrack object.
        index : Optional[list], optional
            List of indices of the plotters to load the track file into. If None, all plotters will load the track file.

        """
        if index is None:
            index = list(range(self.num_plotters))
        if plot_track_once:
            self.plotters[0].load_track(track_file)
        else:
            for i in index:
                self.plotters[i].load_track(track_file)

    def plot(self, **input_kwargs) -> None:
        """Plot the trajectories of all plotters in 2D.

        Parameters
        ----------
        input_kwargs : dict
            Additional keyword arguments for the plot function, such as `fig_name`, `fig_title`,
            `cmap`, etc.

        """
        input_kwargs["fig_name"] = input_kwargs.get("fig_name", "racetrack")
        input_kwargs["fig_title"] = input_kwargs.get("fig_title", "racetrack")
        fig_name = input_kwargs["fig_name"]
        fig_title = input_kwargs["fig_title"]
        for i, plotter in enumerate(self.plotters):
            kwargs = input_kwargs.copy()
            if isinstance(kwargs.get("cmap", None), list):
                kwargs["cmap"] = kwargs["cmap"][i % len(kwargs["cmap"])]
            kwargs["fig_name"] = f"{fig_name}_drone{i + 1}_2d"
            kwargs["fig_title"] = f"{fig_title} (drone {i + 1})"
            plotter.plot(**kwargs)

    def plot3d(self, **input_kwargs) -> None:
        """Plot the trajectories of all plotters in 3D.

        Parameters
        ----------
        input_kwargs : dict
            Additional keyword arguments for the plot3d function, such as `fig_name`, `fig_title`,
            `cmap`, etc.

        """
        input_kwargs["fig_name"] = input_kwargs.get("fig_name", "racetrack")
        input_kwargs["fig_title"] = input_kwargs.get("fig_title", "racetrack")
        fig_name = input_kwargs["fig_name"]
        fig_title = input_kwargs["fig_title"]
        for i, plotter in enumerate(self.plotters):
            kwargs = input_kwargs.copy()
            if isinstance(kwargs.get("cmap", None), list):
                kwargs["cmap"] = kwargs["cmap"][i % len(kwargs["cmap"])]
            kwargs["fig_name"] = f"{fig_name}_drone{i + 1}_3d"
            kwargs["fig_title"] = f"{fig_title} (drone {i + 1})"
            plotter.plot3d(**kwargs)

    def create_animation(self, **input_kwargs) -> List[animation.FuncAnimation]:
        """Create animations for all plotters.

        Parameters
        ----------
        input_kwargs : dict
            Additional keyword arguments for the create_animation function, such as `video_name`, `cmap`, `plot_colorbar`, etc.

        Returns
        -------
        List[animation.FuncAnimation]
            A list of FuncAnimation objects for each plotter.

        """
        ani_list = []
        fig, ax, cax_container = self._ensure_ani_fig_exists()

        # --- Create dedicated sub-axes for each colorbar within the container ---
        if self.num_plotters > 0 and input_kwargs.get("plot_colorbar", True):
            # Get the position of the container axes [left, bottom, width, height]
            bbox = cax_container.get_position()
            cax_container.set_visible(False)  # Hide the container itself

            # Define horizontal spacing between colorbars
            padding = 0.1  # 10% of the container width as padding
            total_width = bbox.width
            sub_cax_width = (
                total_width * (1 - padding * (self.num_plotters - 1))
            ) / self.num_plotters

            sub_caxes = []
            for i in range(self.num_plotters):
                # Calculate left position for each sub-cax, arranging from left to right
                left = bbox.x0 + i * (sub_cax_width + total_width * padding)
                bottom = bbox.y0
                width = sub_cax_width
                height = bbox.height

                sub_cax = fig.add_axes([left, bottom, width, height])
                sub_caxes.append(sub_cax)
        else:
            cax_container.set_visible(False)
            sub_caxes = [None] * self.num_plotters

        # --- Loop through plotters and create animations ---
        input_kwargs["plot_colorbar"] = input_kwargs.get("plot_colorbar", True)
        if self.num_plotters > 1:
            input_kwargs["adjest_colorbar"] = False

        for i, plotter in enumerate(self.plotters):
            kwargs = input_kwargs.copy()
            if isinstance(kwargs.get("cmap", None), list):
                kwargs["cmap"] = kwargs["cmap"][i % len(kwargs["cmap"])]

            video_full_name = kwargs.get("video_name", "racetrack")
            video_name, ext = os.path.splitext(video_full_name)
            if not ext:
                ext = ".mp4"
            kwargs["video_name"] = f"{video_name}{ext}"
            if i < self.num_plotters - 1:
                kwargs["show_bar_info"] = False
                kwargs["show_title"] = False
                kwargs["show_time"] = False
                kwargs["save_path"] = None

            # Pass the dedicated colorbar axes to the individual plotter
            plotter.ani_cax = sub_caxes[i]

            # Ensure the plotter uses the shared figure and main axes
            plotter._fig_ani = fig
            plotter.ani_ax = ax

            ani = plotter.create_animation(**kwargs)
            ani_list.append(ani)

        return ani_list

    def plot_show(self) -> None:
        """Display the plot for all plotters."""
        plt.show()


class RacePlotter(BasePlotter):
    def __init__(
        self,
        traj_file: Union[os.PathLike, str, np.ndarray],
        track_file: Union[os.PathLike, str, RaceTrack] = None,
        wpt_path: Optional[Union[os.PathLike, str]] = None,
        end_time: Optional[int] = None,
        crash_effect: Optional[bool] = False,
        crash_kwargs: dict = {},
        moving_gate_data: Optional[np.ndarray] = None,
    ):
        """Initialize the RacePlotter class.

        Parameters
        ----------
        traj_file : Union[os.PathLike, str, np.ndarray]
            The trajectory file to load, which can be a path to a CSV file or a numpy array.
        track_file : Union[os.PathLike, str, RaceTrack], optional
            The track file to load, which can be a path to a YAML file or a RaceTrack object.
        wpt_path : Optional[Union[os.PathLike, str]], optional
            The path to the waypoints file, if any.
        end_time : Optional[int], optional
            The end time of the trajectory, if known.
        crash_effect : Optional[bool], optional
            If True, enables crash effects in the visualization (default is False).
        crash_kwargs : dict, optional
            Additional keyword arguments for crash effects, such as `crash_radius`, `crash_color`, etc.
        moving_gate_data : Optional[np.ndarray], optional
            The moving gate data, if any, as a numpy array with columns for time and position.

        """
        super().__init__(
            traj_file=traj_file,
            track_file=track_file,
            wpt_path=wpt_path,
            end_time=end_time,
            moving_gate_data=moving_gate_data,
        )
        self.crash_effect = crash_effect
        self.crash_kwargs = crash_kwargs

    def estimate_tangents(self, ps: np.ndarray) -> np.ndarray:
        """Estimate the tangents of the trajectory points.

        Parameters
        ----------
        ps : np.ndarray
            An array of shape (N, 3) representing the trajectory points in 3D space.

        Returns
        -------
        np.ndarray
            An array of shape (N, 3) representing the normalized tangents at each trajectory point.

        """
        # compute tangents
        dp_x = np.gradient(ps[:, 0])
        dp_y = np.gradient(ps[:, 1])
        dp_z = np.gradient(ps[:, 2])
        tangents = np.vstack((dp_x, dp_y, dp_z)).T
        # normalize tangents
        tangents /= np.linalg.norm(tangents, axis=1).reshape(-1, 1)
        return tangents

    def sigmoid(
        self,
        x: np.ndarray,
        bias: float,
        inner_radius: float,
        outer_radius: float,
        rate: float,
    ) -> np.ndarray:
        """Compute a sigmoid function to determine the tube radius based on distance from waypoints.

        Parameters
        ----------
        x : np.ndarray
            An array of distances from waypoints.
        bias : float
            The bias value for the sigmoid function, which shifts the curve along the x-axis.
        inner_radius : float
            The inner radius of the tube.
        outer_radius : float
            The outer radius of the tube.
        rate : float
            The rate of change for the sigmoid function, controlling how quickly it transitions from inner to outer radius.

        Returns
        -------
        np.ndarray
            An array of tube radii computed using the sigmoid function.

        """
        return inner_radius + outer_radius * (1 / (1 + np.exp(-rate * (x - bias))))

    def get_line_tube(
        self, ps: np.ndarray, tube_radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a tube around the trajectory points.

        Parameters
        ----------
        ps : np.ndarray
            An array of shape (N, 3) representing the trajectory points in 3D space.
        tube_radius : float
            The radius of the tube to be created around the trajectory.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Three arrays representing the x, y, and z coordinates of the tube points.

        """
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
            arbitrary_vector = (
                np.array([1, 0, 0])
                if not np.allclose(tangent[i], [1, 0, 0])
                else np.array([0, 1, 0])
            )

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
        """Create a tube around the trajectory points using a sigmoid function to determine the radius.

        Parameters
        ----------
        ts : np.ndarray
            An array of timestamps corresponding to the trajectory points.
        ps : np.ndarray
            An array of shape (N, 3) representing the trajectory points in 3D space.
        bias : float
            The bias value for the sigmoid function, which shifts the curve along the x-axis.
        inner_radius : float
            The inner radius of the tube.
        outer_radius : float
            The outer radius of the tube.
        rate : float
            The rate of change for the sigmoid function, controlling how quickly it transitions from inner to outer radius.
        scale : float, optional
            A scaling factor for the tube radius (default is 1.0).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Three arrays representing the x, y, and z coordinates of the tube points.

        """
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

        tube_size = self.sigmoid(
            min_distances, bias, inner_radius, outer_radius, rate
        )  # 根据距离计算 tube 半径
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
            arbitrary_vector = (
                np.array([1, 0, 0])
                if not np.allclose(tangent[i], [1, 0, 0])
                else np.array([0, 1, 0])
            )

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
    ) -> None:
        """Plot the trajectory in 2D.

        Parameters
        ----------
        cmap : Colormap, optional
            The colormap to use for the speed visualization (default is plt.cm.winter.reversed()).
        save_fig : bool, optional
            If True, saves the figure to a file (default is False).
        dpi : int, optional
            The resolution of the saved figure in dots per inch (default is 300).
        save_path : Union[os.PathLike, str], optional
            The directory where the figure will be saved (default is None).
        fig_name : Optional[str], optional
            The name of the figure file (default is None, which will use "togt_traj.png").
        fig_title : Optional[str], optional
            The title of the figure (default is None).
        radius : Optional[float], optional
            The radius of the tube to be plotted (default is None, which will use the default radius).
        width : Optional[float], optional
            The width of the plot (default is None, which will use the default width).
        height : Optional[float], optional
            The height of the plot (default is None, which will use the default height).
        margin : Optional[float], optional
            The margin around the plot (default is None, which will use the default margin).
        draw_tube : bool, optional
            If True, draws a tube around the trajectory (default is False).
        sig_tube : bool, optional
            If True, uses a sigmoid function to determine the tube radius (default is False).
        tube_color : Optional[str], optional
            The color of the tube (default is None, which will use "purple").
        alpha : float, optional
            The transparency of the tube (default is 0.01).
        tube_rate : float, optional
            The rate of change for the sigmoid function used to determine the tube radius (default is 6).

        """
        self._ensure_2d_fig_exists()
        fig = self._fig_2d
        ax = self.ax_2d
        ps = self.ps
        vt = self.vt

        if draw_tube:
            if not sig_tube:
                self.plot_tube(
                    sig_tube=sig_tube,
                    tube_color=tube_color,
                    alpha=alpha,
                    tube_radius=radius,
                )
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

        sc = ax.scatter(ps[:, 0], ps[:, 1], s=5, c=vt, cmap=cmap)
        fig.colorbar(sc, ax=ax, pad=0.01).ax.set_ylabel("Speed [m/s]")

        if self.track_file is not None:
            plot_track(
                ax,
                self.track_file,
                radius=radius,
                width=width,
                height=height,
                margin=margin,
            )

        if fig_title is not None:
            ax.set_title(fig_title, fontsize=26)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        ax.grid()

        if save_fig:
            if save_path is None:
                raise ValueError("save_path is not provided.")
            os.makedirs(save_path, exist_ok=True)
            fig_name = (fig_name + ".png") if fig_name is not None else "togt_traj.png"
            fig.savefig(os.path.join(save_path, fig_name), dpi=dpi, bbox_inches="tight")

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
    ) -> None:
        """Plot a tube around the trajectory in 2D.

        Parameters
        ----------
        scale : float, optional
            A scaling factor for the tube radius (default is 1.0).
        sig_tube : bool, optional
            If True, uses a sigmoid function to determine the tube radius (default is False).
        tube_color : Optional[str], optional
            The color of the tube (default is None, which will use "purple").
        alpha : float, optional
            The transparency of the tube (default is 0.01).
        tube_edge_color : Optional[str], optional
            The edge color of the tube (default is None, which will use the same color as `tube_color`).
        tube_radius : float, optional
            The radius of the tube to be plotted (default is 1.0).
        bias : float, optional
            The bias value for the sigmoid function, which shifts the curve along the x-axis (default is 1.0).
        inner_radius : float, optional
            The inner radius of the tube (default is 0.5).
        outer_radius : float, optional
            The outer radius of the tube (default is 2.0).
        rate : float, optional
            The rate of change for the sigmoid function, controlling how quickly it transitions from inner to outer radius (default is 6).

        """
        self._ensure_2d_fig_exists()
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
                ts,
                ps,
                bias=bias,
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                rate=rate,
                scale=scale,
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
    ) -> None:
        """Plot the trajectory in 3D.

        Parameters
        ----------
        cmap : Colormap, optional
            The colormap to use for the speed visualization (default is plt.cm.winter.reversed()).
        save_fig : bool, optional
            If True, saves the figure to a file (default is False).
        dpi : int, optional
            The resolution of the saved figure in dots per inch (default is 300).
        save_path : Union[os.PathLike, str], optional
            The directory where the figure will be saved (default is None).
        fig_name : Optional[str], optional
            The name of the figure file (default is None, which will use "togt_traj.png").
        fig_title : Optional[str], optional
            The title of the figure (default is None).
        radius : Optional[float], optional
            The radius of the tube to be plotted (default is None, which will use the default radius).
        width : Optional[float], optional
            The width of the plot (default is None, which will use the default width).
        height : Optional[float], optional
            The height of the plot (default is None, which will use the default height).
        margin : Optional[float], optional
            The margin around the plot (default is None, which will use the default margin).
        draw_tube : bool, optional
            If True, draws a tube around the trajectory (default is False).
        sig_tube : bool, optional
            If True, uses a sigmoid function to determine the tube radius (default is False).
        gate_color : Optional[str], optional
            The color of the gate (default is None, which will use "purple").
        tube_color : Optional[str], optional
            The color of the tube (default is None, which will use "purple").
        alpha : float, optional
            The transparency of the tube (default is 0.01).
        gate_alpha : float, optional
            The transparency of the gate (default is 0.1).
        tube_rate : float, optional
            The rate of change for the sigmoid function used to determine the tube radius (default is 6).
        shade : bool, optional
            If True, applies shading to the tube surface (default is True).

        """
        self._ensure_3d_fig_exists()
        fig = self._fig_3d
        ax = self.ax_3d

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

        ps = self.ps
        vt = self.vt

        # draw tube
        if draw_tube:
            if not sig_tube:
                self.plot3d_tube(
                    sig_tube=sig_tube,
                    tube_color=tube_color,
                    alpha=alpha,
                    tube_radius=radius,
                    shade=shade,
                )
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
        fig.colorbar(
            sc, ax=ax, shrink=shrink_factor, aspect=colorbar_aspect, pad=0.1
        ).ax.set_ylabel("Speed [m/s]")

        if self.track_file is not None:
            plot_track_3d(
                ax,
                self.track_file,
                radius=radius,
                width=width,
                height=height,
                margin=margin,
                gate_color=gate_color,
                gate_alpha=gate_alpha,
            )

        if fig_title is not None:
            fig.text(
                0.5,
                0.95,
                fig_title,
                fontsize=26,
                horizontalalignment="center",
                verticalalignment="top",
            )
        ax.set_xlabel("x [m]", labelpad=30 * (x_range / max_range))
        ax.set_ylabel("y [m]", labelpad=30 * (y_range / max_range))
        ax.set_zlabel("z [m]", labelpad=30 * (z_range / max_range))
        ax.axis("equal")
        ax.grid()

        if save_fig:
            if save_path is None:
                raise ValueError("save_path is not provided.")
            os.makedirs(save_path, exist_ok=True)
            fig_name = (fig_name + ".png") if fig_name is not None else "togt_traj.png"
            fig.savefig(os.path.join(save_path, fig_name), dpi=dpi, bbox_inches="tight")

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
    ) -> None:
        """Plot a tube around the trajectory in 3D.

        Parameters
        ----------
        scale : float, optional
            A scaling factor for the tube radius (default is 1.0).
        sig_tube : bool, optional
            If True, uses a sigmoid function to determine the tube radius (default is False).
        tube_color : Optional[str], optional
            The color of the tube (default is None, which will use "purple").
        alpha : float, optional
            The transparency of the tube (default is 0.01).
        tube_edge_color : Optional[str], optional
            The edge color of the tube (default is None, which will use the same color as `tube_color`).
        tube_radius : float, optional
            The radius of the tube to be plotted (default is 1.0).
        bias : float, optional
            The bias value for the sigmoid function, which shifts the curve along the x-axis (default is 1.0).
        inner_radius : float, optional
            The inner radius of the tube (default is 0.5).
        outer_radius : float, optional
            The outer radius of the tube (default is 2.0).
        rate : float, optional
            The rate of change for the sigmoid function, controlling how quickly it transitions from inner to outer radius (default is 6).
        shade : bool, optional
            If True, applies shading to the tube surface (default is True).

        """
        self._ensure_3d_fig_exists()
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
                ts,
                ps,
                bias=bias,
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                rate=rate,
                scale=scale,
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

    def create_animation(
        self,
        save_path: Union[os.PathLike, str] = None,
        video_name: str = "drone_animation.mp4",
        fps: int = 20,
        dpi: int = 200,
        drone_kwargs: dict = {},
        cmap: Colormap = plt.cm.winter.reversed(),
        traj_history: float = 0.0,
        track_kwargs: dict = {},
        follow_drone: bool = False,
        hide_background: bool = False,
        hide_ground: bool = False,
        flash_gate: bool = False,
        plot_colorbar: bool = True,
        adjest_colorbar: bool = True,
        show_bar_info: bool = True,
        show_title: bool = True,
        show_time: bool = True,
    ) -> animation.FuncAnimation:
        """Create a 3D animation of the drone trajectory.

        Parameters
        ----------
        save_path : Union[os.PathLike, str], optional
            The directory where the animation will be saved (default is None).
        video_name : str, optional
            The name of the video file (default is "drone_animation.mp4").
        fps : int, optional
            The frames per second for the animation (default is 20).
        dpi : int, optional
            The resolution of the saved video in dots per inch (default is 200).
        drone_kwargs : dict, optional
            Additional keyword arguments for the drone visualization, such as `arm_length`, `color`, etc.
        cmap : Colormap, optional
            The colormap to use for the speed visualization (default is plt.cm.winter.reversed()).
        traj_history : float, optional
            The history of the trajectory to show in seconds (default is 0.0, which means no history).
        track_kwargs : dict, optional
            Additional keyword arguments for the track visualization, such as `gate_color`, `gate_alpha`, etc.
        follow_drone : bool, optional
            If True, the camera will follow the drone (default is False).
        hide_background : bool, optional
            If True, hides the background of the plot (default is False).
        hide_ground : bool, optional
            If True, hides the ground plane in the plot (default is False).
        flash_gate : bool, optional
            If True, flashes the gate during the animation (default is False).
        plot_colorbar : bool, optional
            If True, plots the colorbar for speed visualization (default is True).
        adjest_colorbar : bool, optional
            If True, adjusts the colorbar to fit the speed range (default is True).
        show_bar_info : bool, optional
            If True, shows the colorbar information (default is True).
        show_title : bool, optional
            If True, shows the title of the figure (default is True).
        show_time : bool, optional
            If True, shows the dynamic time text in the figure (default is True).

        Returns
        -------
        animation.FuncAnimation
            The created animation object.

        """
        self._ensure_ani_fig_exists()
        fig = self._fig_ani
        ax = self.ani_ax
        cax = self.ani_cax

        # Add the main title to the figure if provided
        if show_title:
            fig_title, ext = os.path.splitext(video_name)
            fig.suptitle(fig_title, fontsize=20)

        # Initialize the dynamic time text object in the top-left corner
        if show_time:
            time_text = fig.text(
                0.92,  # x-position: 95% from the left
                0.08,  # y-position: 5% from the bottom
                "",  # Initial text is empty
                fontsize=20,
                transform=fig.transFigure,
                ha="right",  # Horizontal alignment: right
                va="bottom",  # Vertical alignment: bottom
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
            )

        # --- Crash Effect Parameters ---
        crash_n_debris = self.crash_kwargs.get("n_debris", 60)
        crash_duration = self.crash_kwargs.get("duration", 1.2)
        crash_color = self.crash_kwargs.get("color", "orange")

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
        x_max, x_min = max(x_max, (x_max + x_min + x_range) / 2), min(
            x_min, (x_max + x_min - x_range) / 2
        )
        y_max, y_min = max(y_max, (y_max + y_min + y_range) / 2), min(
            y_min, (y_max + y_min - y_range) / 2
        )
        z_max, z_min = max(z_max, (z_max + z_min + z_range) / 2), min(
            z_min, (z_max + z_min - z_range) / 2
        )

        # compute ticks
        x_ticks_count = max(min(int(x_range), 5), 3)
        y_ticks_count = max(min(int(y_range), 5), 3)
        z_ticks_count = max(min(int(z_range), 5), 3)

        # create the quadcopter drawer
        arm_length = drone_kwargs.get("arm_length", 0.2)
        if not follow_drone and "arm_length" not in drone_kwargs:
            arm_length = max(max_range / 30, 0.2)  # automatically set arm length
        drawer_kwargs = drone_kwargs.copy()
        drawer_kwargs.pop("arm_length", None)
        self.quadcopter_drawer = QuadcopterDrawer(
            ax=ax, arm_length=arm_length, **drawer_kwargs
        )

        # ensure positions in a right shape
        if self.end_time is not None:
            max_frame = int(self.end_time * fps)
        else:
            max_frame = 0
        total_frames = int((self.t[-1] - self.t[0]) * fps)
        frame_history = int(fps * traj_history)
        times = np.linspace(self.t[0], self.t[-1], total_frames)
        positions = np.array(
            [
                np.interp(times, self.t, self.p_x),
                np.interp(times, self.t, self.p_y),
                np.interp(times, self.t, self.p_z),
            ]
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
        time_steps = (
            positions.shape[0] if max_frame <= 0 else max(max_frame, positions.shape[0])
        )
        if self.moving_gate_ps is not None:
            if self.gate_p_x.ndim == 2:
                num_gates = self.gate_p_x.shape[1]
                num_ts = len(times)

                # Initialize the final array with shape (num_ts, num_gates, 3)
                moving_gate_ps = np.zeros((num_ts, num_gates, 3))

                for i in range(num_gates):
                    # Interpolate each coordinate for the i-th gate
                    moving_gate_ps[:, i, 0] = np.interp(
                        times, self.gate_t, self.gate_p_x[:, i]
                    )
                    moving_gate_ps[:, i, 1] = np.interp(
                        times, self.gate_t, self.gate_p_y[:, i]
                    )
                    moving_gate_ps[:, i, 2] = np.interp(
                        times, self.gate_t, self.gate_p_z[:, i]
                    )
            else:
                # Fallback for 1D data (single gate case)
                moving_gate_ps = np.array(
                    [
                        np.interp(times, self.gate_t, self.gate_p_x),
                        np.interp(times, self.gate_t, self.gate_p_y),
                        np.interp(times, self.gate_t, self.gate_p_z),
                    ]
                ).T
        else:
            moving_gate_ps = None

        # create lines for the drone
        quad_artists = []
        lines = []
        gate_artists = []
        crash_artists = []  # Artists for the crash effect

        # plot the track
        if self.track_file is not None:
            if moving_gate_ps is not None:
                if isinstance(self.track_file, RaceTrack):
                    track = self.track_file.to_dict()
                else:
                    track = yaml.safe_load(open(self.track_file).read())
                gates = []
                gate_counter = 0
                for gate_id in track["orders"]:
                    gate_temp = track[gate_id]
                    gate_temp["position"] = moving_gate_ps[0][gate_counter].tolist()
                    gates.append(gate_temp)
                    gate_counter += 1
                track_len = len(gates)
            elif not flash_gate:
                if isinstance(self.track_file, RaceTrack):
                    # If the track is a RaceTrack object, use its to_dict method
                    track = self.track_file.to_dict()
                    gates = []
                    # Populate the gates list from the track data
                    for gate_id in track["orders"]:
                        gates.append(track[gate_id])
                    track_len = len(gates)

                    # Filter out gates that are in overlapping positions
                    gates_to_plot = []
                    unique_positions = []
                    # A small distance threshold to consider positions as identical
                    overlap_threshold = 1e-2  # 1 cm

                    for gate in gates:
                        current_pos = np.array(gate["position"])
                        is_unique = True
                        # Check if the current gate's position is too close to any already added unique positions
                        for unique_pos in unique_positions:
                            if (
                                np.linalg.norm(current_pos - unique_pos)
                                < overlap_threshold
                            ):
                                is_unique = False
                                break

                        if is_unique:
                            gates_to_plot.append(gate)
                            unique_positions.append(current_pos)

                    # draw the gates
                    if gates_to_plot:
                        artists = plot_gate_3d(
                            ax,
                            gates_to_plot,  # Pass the filtered list
                            **track_kwargs,
                        )
                else:
                    plot_track_3d(
                        ax,
                        self.track_file,
                        **track_kwargs,
                    )
            else:
                track = self.track_file.to_dict()
                gates = []
                for gate_id in track["orders"]:
                    gates.append(track[gate_id])
                track_len = len(gates)

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
        if not hide_background:
            norm = plt.Normalize(vt_min, vt_max)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            shrink_factor = min(0.8, max(0.6, 0.6 * y_range / x_range))
            colorbar_aspect = 20 * shrink_factor
            if plot_colorbar:
                if adjest_colorbar:
                    cbar = fig.colorbar(
                        sm,
                        cax=cax,
                        shrink=shrink_factor,
                        aspect=colorbar_aspect,
                        pad=0.1,
                    )
                else:
                    cbar = fig.colorbar(sm, cax=cax)

                if show_bar_info:
                    cbar.set_label("Velocity [m/s]")
                else:
                    cbar.ax.set_yticks([])
        elif hide_background:
            # set background color to empty
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            # hide pane lines
            ax.xaxis.pane.set_edgecolor("none")
            ax.yaxis.pane.set_edgecolor("none")
            ax.zaxis.pane.set_edgecolor("none")

            ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

            if not hide_ground:
                # scale the ground
                scale_factor = 1.5
                x_min_ground = (x_min + x_max) / 2 - (x_range * scale_factor / 2)
                x_max_ground = (x_min + x_max) / 2 + (x_range * scale_factor / 2)
                y_min_ground = (y_min + y_max) / 2 - (y_range * scale_factor / 2)
                y_max_ground = (y_min + y_max) / 2 + (y_range * scale_factor / 2)

                xx, yy = np.meshgrid(
                    np.linspace(x_min_ground, x_max_ground, 15),
                    np.linspace(y_min_ground, y_max_ground, 15),
                )
                zz = np.ones_like(xx) * z_min
                # draw the ground
                ground_surface = ax.plot_surface(
                    xx, yy, zz, color="gray", alpha=0.05, shade=True, edgecolor="none"
                )
                # draw the wireframe
                ground_wireframe = ax.plot_wireframe(
                    xx, yy, zz, color="gray", alpha=0.5, linewidth=0.5
                )
                ax.set_xlim(x_min_ground, x_max_ground)
                ax.set_ylim(y_min_ground, y_max_ground)
                ax.set_box_aspect(
                    [x_range * scale_factor, y_range * scale_factor, z_range]
                )

        # the initialization function
        def init():
            """Initialize the animation.

            Returns
            -------
            List[Artist]
                A list of artists to be drawn in the initial frame.

            """
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

            # clear previous gate artists
            for artist in gate_artists:
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
            gate_artists.clear()

            # clear previous crash artists
            for artist in crash_artists:
                artist.remove()
            crash_artists.clear()

            # Reset time text
            if show_time:
                time_text.set_text("")

            return []

        # the update function for each frame
        def update(frame):
            """Update the animation for each frame.

            Parameters
            ----------
            frame : int
                The current frame number.

            Returns
            -------
            List[Artist]
                A list of artists to be drawn in the current frame.

            """
            # Update the time text with the current simulation time
            if show_time:
                time_text.set_text(f"Time: {(frame / fps):.2f}s")

            # --- Crash / End of Data Logic ---
            if self.crash_effect and frame >= len(positions):
                # If this is the first frame after data runs out, create the crash effect.
                if not hasattr(update, "crashed") or not update.crashed:
                    update.crashed = True
                    update.crash_frame = frame
                    last_position = positions[-1]
                    last_velocity = np.linalg.norm(vt[-1])

                    # 1. Clear the drone model
                    for artist in quad_artists:
                        artist.remove()
                    quad_artists.clear()

                    # 2. Create the scatter plot object for debris, initially at the crash point
                    debris_radius = arm_length * 2.0
                    debris_positions = (
                        last_position
                        + (np.random.rand(crash_n_debris, 3) - 0.5) * debris_radius
                    )
                    debris = ax.scatter(
                        debris_positions[:, 0],
                        debris_positions[:, 1],
                        debris_positions[:, 2],
                        color=crash_color,
                        marker=".",
                        s=20,
                        alpha=0.7,
                    )
                    crash_artists.append(debris)
                    update.debris_artist = debris

                    # 3. Initialize debris velocities (random directions, varied speeds)
                    velocities = debris_positions - last_position
                    velocities /= np.linalg.norm(velocities, axis=1)[:, np.newaxis]
                    velocities *= (
                        last_velocity
                        * (0.5 + 0.5 * np.random.rand(crash_n_debris, 1))
                        / 10
                    )
                    update.debris_positions = debris_positions
                    update.debris_velocities = velocities

                elif update.crashed:
                    # If already crashed, just update the crash artists
                    # --- Update the diffusion animation on subsequent frames ---
                    time_since_crash = (frame - update.crash_frame) / fps
                    if time_since_crash < crash_duration:
                        # Calculate new positions
                        new_positions = (
                            update.debris_positions
                            + update.debris_velocities * time_since_crash
                        )
                        update.debris_artist._offsets3d = (
                            new_positions[:, 0],
                            new_positions[:, 1],
                            new_positions[:, 2],
                        )

                        # Calculate fade-out alpha
                        new_alpha = max(0, 1 - (time_since_crash / crash_duration))
                        update.debris_artist.set_alpha(new_alpha)
                    else:
                        # Hide debris after duration
                        update.debris_artist.set_alpha(0)

                # Return only the persistent crash artists
                return crash_artists

            # If not crashed, reset the flag
            update.crashed = False
            display_frame = frame

            # draw the trajectory
            if display_frame > 0:
                if frame_history > 0:
                    # 1. Add the newest trajectory segment
                    new_segment_color = cmap(vt_norm[display_frame])
                    (new_segment,) = ax.plot(
                        positions[display_frame - 1 : display_frame + 1, 0],
                        positions[display_frame - 1 : display_frame + 1, 1],
                        positions[display_frame - 1 : display_frame + 1, 2],
                        color=new_segment_color,
                        linewidth=2,
                    )
                    lines.append(new_segment)

                    # 2. If the history buffer is full, remove the oldest segment
                    if len(lines) > frame_history:
                        oldest_segment = lines.pop(
                            0
                        )  # Get the oldest line from the list
                        try:
                            oldest_segment.remove()  # Remove it from the plot
                        except ValueError:
                            pass  # Already removed
                else:
                    # only update the last segment
                    if display_frame > 1 and hasattr(update, "last_frame"):
                        start_idx = update.last_frame
                        end_idx = display_frame

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
                            positions[0 : display_frame + 1, 0],
                            positions[0 : display_frame + 1, 1],
                            positions[0 : display_frame + 1, 2],
                            color=color,
                            linewidth=2,
                        )
                        lines.append(segment)

                    # update the last frame
                    update.last_frame = display_frame
            else:
                update.next_id = 0

            # clear previous quadcopter artists
            for artist in quad_artists:
                try:
                    artist.remove()
                except:
                    pass
            quad_artists.clear()

            # draw the quadcopter
            position = positions[display_frame]
            attitude = attitudes[display_frame]

            # draw the quadcopter
            artists = self.quadcopter_drawer.draw(position=position, attitude=attitude)
            quad_artists.extend(artists)

            # draw the gate
            if flash_gate:
                if update.next_id < track_len:
                    # clear previous gate artists
                    for artist in gate_artists:
                        try:
                            artist.remove()
                        except:
                            pass
                    gate_artists.clear()

                    if moving_gate_ps is not None:
                        gates[update.next_id]["position"] = moving_gate_ps[
                            display_frame, update.next_id
                        ]

                    # draw the next gate
                    artists = plot_gate_3d(
                        ax,
                        gates[update.next_id],
                        **track_kwargs,
                    )
                    gate_artists.extend(artists)
                    if (
                        np.linalg.norm(
                            gates[update.next_id]["position"] - positions[display_frame]
                        )
                        < gates[update.next_id]["radius"]
                    ):
                        update.next_id += 1
            elif moving_gate_ps is not None:
                # clear previous gate artists
                for artist in gate_artists:
                    try:
                        artist.remove()
                    except:
                        pass
                gate_artists.clear()

                for i in range(len(gates)):
                    gates[i]["position"] = moving_gate_ps[display_frame, i]

                # Filter out gates that are in overlapping positions
                gates_to_plot = []
                unique_positions = []
                # A small distance threshold to consider positions as identical
                overlap_threshold = 1e-2  # 1 cm

                for gate in gates:
                    current_pos = np.array(gate["position"])
                    is_unique = True
                    # Check if the current gate's position is too close to any already added unique positions
                    for unique_pos in unique_positions:
                        if np.linalg.norm(current_pos - unique_pos) < overlap_threshold:
                            is_unique = False
                            break

                    if is_unique:
                        gates_to_plot.append(gate)
                        unique_positions.append(current_pos)

                # draw the gates
                if gates_to_plot:
                    artists = plot_gate_3d(
                        ax,
                        gates_to_plot,  # Pass the filtered list
                        **track_kwargs,
                    )
                    gate_artists.extend(artists)

            # support follow camera
            if follow_drone:
                R = Rotation.from_quat(attitude).as_matrix()

                # compute forward vector
                forward_vector = R @ np.array([1, 0, 0])  # x axis forward

                # set the view range
                x_view_range = arm_length * 12
                y_view_range = arm_length * 12
                z_view_range = arm_length * 12

                # compute the target azimuth and elevation angles
                target_azim = (
                    -np.degrees(np.arctan2(forward_vector[1], forward_vector[0])) + 180
                ) % 360
                horizontal_distance = np.sqrt(
                    forward_vector[0] ** 2 + forward_vector[1] ** 2
                )
                target_elev = (
                    -np.degrees(np.arctan2(forward_vector[2], horizontal_distance)) / 3
                    + 30
                )

                # smooth the camera angles
                if not hasattr(update, "current_azim"):
                    update.current_azim = target_azim
                    update.current_elev = target_elev
                else:
                    smooth_factor = 0.10  # smooth factor, 0 < smooth_factor < 1, smaller value means smoother

                    # prevent azimuth angle from jumping
                    azim_diff = target_azim - update.current_azim
                    if azim_diff > 180:
                        azim_diff -= 360
                    elif azim_diff < -180:
                        azim_diff += 360

                    update.current_azim = (
                        update.current_azim + smooth_factor * azim_diff
                    ) % 360
                    update.current_elev = update.current_elev + smooth_factor * (
                        target_elev - update.current_elev
                    )

                # set the smooth azimuth and elevation angles
                ax.view_init(elev=update.current_elev, azim=update.current_azim)

                # set the camera limits
                ax.set_xlim(
                    position[0] - x_view_range / 2, position[0] + x_view_range / 2
                )
                ax.set_ylim(
                    position[1] - y_view_range / 2, position[1] + y_view_range / 2
                )
                ax.set_zlim(
                    position[2] - z_view_range / 2, position[2] + z_view_range / 2
                )
                # set the aspect ratio
                ax.set_box_aspect((x_view_range, y_view_range, z_view_range))
                # set ticks
                x_view_ticks_count = max(min(int(x_view_range), 5), 3)
                y_view_ticks_count = max(min(int(y_view_range), 5), 3)
                z_view_ticks_count = max(min(int(z_view_range), 5), 3)
                self.set_nice_ticks(ax, x_view_range, x_view_ticks_count, "x")
                self.set_nice_ticks(ax, y_view_range, y_view_ticks_count, "y")
                self.set_nice_ticks(ax, z_view_range, z_view_ticks_count, "z")

            if hide_background:
                for ax_temp in fig.axes:
                    if hasattr(ax_temp, "orientation") or ax_temp != ax:
                        fig.delaxes(ax_temp)
                # fig.set_size_inches(12, 8)

                # hide all axis ticks
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                # hide all axis labels
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_zlabel("")

            # collect all artists
            all_artists = lines.copy()
            all_artists.extend(artists)
            all_artists.extend(quad_artists)
            all_artists.extend(gate_artists)
            all_artists.extend(crash_artists)
            return all_artists

        # save the animation if a path is provided
        if save_path is not None:
            writer = FFMpegWriter(fps=fps)
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, video_name)

            # print the file path
            print(f"Saving animation to {file_path}...")

            # start the animation writer
            with writer.saving(fig, file_path, dpi=dpi):
                for i in range(time_steps):
                    update(i)  # update the frame
                    writer.grab_frame()

                    # show progress every 10% of the total time steps
                    if (i + 1) % max(1, time_steps // 10) == 0:
                        print(
                            f"Progress: {(i+1)/time_steps*100:.1f}% ({i+1}/{time_steps})"
                        )

                    plt.pause(0.001)

            print(f"Animation saved to {file_path}")

        # return the animation object
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=time_steps,
            init_func=init,
            blit=False,
            interval=1000 / fps,
        )

        return ani
