import numpy as np
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.patches import FancyArrowPatch


class Arrow3D(FancyArrowPatch):
    """Class to draw 3D arrows in matplotlib"""

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


class BaseDrawer:
    """Base class for all drawers in the RaceVisualizer."""

    def __init__(self, ax):
        self.ax = ax
        self.total_artists = []

    def draw(self):
        """Draw the object."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def draw_cylinder(
        self, center, height, radius, rotation_matrix, color="white", alpha=0.8, closed=False
    ):
        """Draw a cylinder in 3D space.

        Parameters
        ----------
        total_artists : list
            List of artists to be drawn.
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

        Returns
        -------
        total_artists : list
            Updated list of artists to be drawn.

        """
        ax = self.ax
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
        surf = ax.plot_surface(x_cyl, y_cyl, z_cyl, color=color, alpha=alpha)
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

                # draw the circle
                circ = ax.plot(circle_x_rot, circle_y_rot, circle_z_rot, color=color)
                artists.append(circ[0])

        self.total_artists.extend(artists)
