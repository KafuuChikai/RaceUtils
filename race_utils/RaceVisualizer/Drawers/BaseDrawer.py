import numpy as np
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backend_bases import RendererBase


class Arrow3D(FancyArrowPatch):
    """Class to draw 3D arrows in matplotlib"""

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        dx: float,
        dy: float,
        dz: float,
        *args: tuple,
        **kwargs: dict,
    ):
        """Initialize the 3D arrow.

        Parameters
        ----------
        x : float
            The x-coordinate of the arrow's start point.
        y : float
            The y-coordinate of the arrow's start point.
        z : float
            The z-coordinate of the arrow's start point.
        dx : float
            The change in x-coordinate (length of the arrow in x).
        dy : float
            The change in y-coordinate (length of the arrow in y).
        dz : float
            The change in z-coordinate (length of the arrow in z).
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.

        """
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer: RendererBase) -> None:
        """Draw the arrow.

        Parameters
        ----------
        renderer : RendererBase
            The renderer to use for drawing.

        """
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self) -> float:
        """Project the arrow in 3D space.

        Returns
        -------
        float
            The minimum z-coordinate of the arrow.

        """
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


class BaseDrawer:
    """Base class for all drawers in the RaceVisualizer."""

    def __init__(self, ax: Axes3D, show_body_axis: bool = False):
        """Initialize the BaseDrawer.

        Parameters
        ----------
        ax : Axes3D
            The 3D axes to draw on.
        show_body_axis : bool
            Whether to show the body axis of the drone.

        """
        self.ax = ax
        self.show_body_axis = show_body_axis
        self.total_artists = []

    def draw(self) -> None:
        """Draw the object."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _draw_cylinder(
        self,
        center: np.ndarray,
        height: float,
        radius: float,
        rotation_matrix: np.ndarray,
        color: str = "white",
        alpha: float = 1.0,
        closed: bool = False,
    ) -> None:
        """Draw a cylinder in 3D space.

        Parameters
        ----------
        center : list
            Center of the cylinder in 3D space.
        height : float
            Height of the cylinder.
        radius : float
            Radius of the cylinder.
        rotation_matrix : np.ndarray
            Rotation matrix to rotate the cylinder.
        color : str
            Color of the cylinder.
        alpha : float
            Transparency of the cylinder.
        closed : bool
            Whether to draw the top and bottom circles of the cylinder.

        """
        artists = []

        # compute the points for the cylinder
        theta = np.linspace(0, 2 * np.pi, 30)
        z = np.linspace(-height / 2, height / 2, 2)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)
        cylinder_points = np.array([x_grid, y_grid, z_grid])

        # apply rotation
        rotated_points = np.einsum("ij,jkl->ikl", rotation_matrix, cylinder_points)

        # move to center
        x_cyl = rotated_points[0] + center[0]
        y_cyl = rotated_points[1] + center[1]
        z_cyl = rotated_points[2] + center[2]

        # draw the surface of the cylinder
        surf = self.ax.plot_surface(x_cyl, y_cyl, z_cyl, color=color, alpha=alpha)
        artists.append(surf)

        # draw the top and bottom circles if closed
        if closed:
            for z_pos in [-height / 2, height / 2]:
                circle_x = radius * np.cos(theta)
                circle_y = radius * np.sin(theta)
                circle_z = np.ones_like(theta) * z_pos

                # apply rotation
                circle_points = np.array([circle_x, circle_y, circle_z])
                rotated_circle = np.einsum("ij,jk->ik", rotation_matrix, circle_points)

                # move to center
                circle_x_rot = rotated_circle[0] + center[0]
                circle_y_rot = rotated_circle[1] + center[1]
                circle_z_rot = rotated_circle[2] + center[2]

                # create the vertices for the circle
                verts = [list(zip(circle_x_rot, circle_y_rot, circle_z_rot))]

                # create the Poly3DCollection
                circle_color = to_rgba(color, alpha)
                circle_poly = Poly3DCollection(
                    verts, facecolors=circle_color, edgecolors=circle_color
                )

                # add the circle to the plot
                self.ax.add_collection3d(circle_poly)
                artists.append(circle_poly)

                # draw the edge of the circle
                circ = self.ax.plot(
                    circle_x_rot, circle_y_rot, circle_z_rot, color=color
                )
                artists.append(circ[0])

        self.total_artists.extend(artists)

    def _draw_sphere(
        self,
        center: np.ndarray,
        radius: float,
        color: str = "white",
        alpha: float = 1.0,
    ) -> None:
        """Draw a sphere in 3D space.

        Parameters
        ----------
        center : np.ndarray
            Center of the sphere in 3D space.
        radius : float
            Radius of the sphere.
        color : str
            Color of the sphere.
        alpha : float
            Transparency of the sphere.

        """
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

        # draw the surface of the sphere
        surf = self.ax.plot_surface(x, y, z, color=color, alpha=alpha)
        self.total_artists.append(surf)

    def _draw_cylinder_ring(
        self,
        center: np.ndarray,
        height: float,
        outer_radius: float,
        inner_radius: float,
        rotation_matrix: np.ndarray,
        color: str = "white",
        alpha: float = 1.0,
    ) -> None:
        """Draw a ring in 3D space.

        Parameters
        ----------
        center : np.ndarray
            Center of the ring in 3D space.
        height : float
            Height of the ring.
        outer_radius : float
            The outer radius of the ring.
        inner_radius : float
            The inner radius of the ring.
        rotation_matrix : np.ndarray
            Rotation matrix to rotate the ring.
        color : str
            Color of the ring.
        alpha : float
            Transparency of the ring.

        """
        artists = []

        # draw the outer cylinder
        self._draw_cylinder(
            center=center,
            height=height,
            radius=outer_radius,
            rotation_matrix=rotation_matrix,
            color=color,
            alpha=alpha,
            closed=False,
        )
        # draw the inner cylinder
        self._draw_cylinder(
            center=center,
            height=height,
            radius=inner_radius,
            rotation_matrix=rotation_matrix,
            color="white",
            alpha=alpha,
            closed=False,
        )

        # draw the outer and inner circles
        for z_pos in [-height / 2, height / 2]:
            # create the points for the ring
            theta = np.linspace(0, 2 * np.pi, 40)

            # outer circle points
            outer_x = outer_radius * np.cos(theta)
            outer_y = outer_radius * np.sin(theta)
            outer_z = np.ones_like(theta) * z_pos

            # inner circle points (reversed order)
            inner_x = inner_radius * np.cos(theta[::-1])
            inner_y = inner_radius * np.sin(theta[::-1])
            inner_z = np.ones_like(theta) * z_pos

            # concatenate the outer and inner points
            ring_x = np.concatenate([outer_x, inner_x])
            ring_y = np.concatenate([outer_y, inner_y])
            ring_z = np.concatenate([outer_z, inner_z])

            # apply rotation
            ring_points = np.array([ring_x, ring_y, ring_z])
            rotated_ring = np.einsum("ij,jk->ik", rotation_matrix, ring_points)

            # move to center
            ring_x_rot = rotated_ring[0] + center[0]
            ring_y_rot = rotated_ring[1] + center[1]
            ring_z_rot = rotated_ring[2] + center[2]

            # create the vertices for the ring
            verts = [list(zip(ring_x_rot, ring_y_rot, ring_z_rot))]

            # create the Poly3DCollection
            ring_color = to_rgba(color, alpha)
            ring_poly = Poly3DCollection(
                verts, facecolors=ring_color, edgecolors=ring_color
            )

            # add the ring to the plot
            self.ax.add_collection3d(ring_poly)
            artists.append(ring_poly)

        self.total_artists.extend(artists)

    def _draw_body_axis(
        self, position: np.ndarray, rotation_matrix: np.ndarray
    ) -> None:
        """Draw an arrow in 3D space.

        Parameters
        ----------
        position : np.ndarray
            Position of the arrow in 3D space.
        rotation_matrix : np.ndarray
            Rotation matrix to rotate the arrow.

        """
        x, y, z = position

        # x-axis
        body_x = np.dot(rotation_matrix, 3 * self.arm_length * np.array([1, 0, 0]))
        arrow_x = Arrow3D(
            x,
            y,
            z,
            body_x[0],
            body_x[1],
            body_x[2],
            mutation_scale=10,
            lw=1,
            arrowstyle="-|>",
            color="red",
        )
        self.ax.add_artist(arrow_x)
        self.total_artists.append(arrow_x)

        # y-axis
        body_y = np.dot(rotation_matrix, 3 * self.arm_length * np.array([0, 1, 0]))
        arrow_y = Arrow3D(
            x,
            y,
            z,
            body_y[0],
            body_y[1],
            body_y[2],
            mutation_scale=10,
            lw=1,
            arrowstyle="-|>",
            color="blue",
        )
        self.ax.add_artist(arrow_y)
        self.total_artists.append(arrow_y)

        # z-axis
        body_z = np.dot(rotation_matrix, 3 * self.arm_length * np.array([0, 0, 1]))
        arrow_z = Arrow3D(
            x,
            y,
            z,
            body_z[0],
            body_z[1],
            body_z[2],
            mutation_scale=10,
            lw=1,
            arrowstyle="-|>",
            color="green",
        )
        self.ax.add_artist(arrow_z)
        self.total_artists.append(arrow_z)
