import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from scipy.spatial.transform import Rotation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D


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


class QuadcopterDrawer:
    """Class to draw a quadcopter in 3D space using matplotlib"""

    def __init__(self, arm_length: float = 0.2, quad_type: str = "X", color: str = "blue"):
        """Initialize the QuadcopterDrawer

        Parameters
        ---------
        arm_length: float
            Length of the quadcopter arms (default is 0.2)
        quad_type: str
            Type of quadcopter ('X' or 'H', default is 'X')
        color: str
            Color of the quadcopter body (default is 'blue')

        """
        self.arm_length = arm_length
        self.quad_type = quad_type
        self.color = color

    def draw_quadcopter(self, ax: Axes3D, position: list, attitude: list) -> list:
        """Draw a quadcopter in 3D space

        Parameters
        ---------
        ax: matplotlib 3D axis object
            The axis on which to draw the quadcopter
        position: list
            [x, y, z] position of the quadcopter
        attitude: list
            [roll, pitch, yaw] Euler angles (in radians) or quaternion [qx, qy, qz, qw]

        Returns:
        --------
        artists: list
            List of artists created for the quadcopter, used for later updates or deletions

        """
        x, y, z = position
        artists = []
        quad_type = self.quad_type
        color = self.color

        # convert attitude to rotation matrix
        if len(attitude) == 3:  # use [roll, pitch, yaw] for euler angles
            R = Rotation.from_euler("xyz", attitude, degrees=False).as_matrix()
        elif len(attitude) == 4:  # use [qx, qy, qz, qw] for quaternion
            R = Rotation.from_quat(attitude).as_matrix()
        else:
            raise ValueError("Attitude must be either Euler angles [roll, pitch, yaw] or quaternion [qx, qy, qz, qw]")

        # Define the quadcopter arms in the body frame
        if quad_type == "X":
            arms = np.array(
                [
                    [-self.arm_length / np.sqrt(2), self.arm_length / np.sqrt(2), 0],  # front-left: id 0
                    [self.arm_length / np.sqrt(2), self.arm_length / np.sqrt(2), 0],  # front-right: id 1
                    [-self.arm_length / np.sqrt(2), -self.arm_length / np.sqrt(2), 0],  # back-left: id 2
                    [self.arm_length / np.sqrt(2), -self.arm_length / np.sqrt(2), 0],  # back-right: id 3
                ]
            )
        elif quad_type == "H":
            arms = np.array(
                [
                    [self.arm_length, 0, 0],  # front
                    [0, self.arm_length, 0],  # right
                    [-self.arm_length, 0, 0],  # back
                    [0, -self.arm_length, 0],  # left
                ]
            )
        else:
            raise ValueError("Quadcopter type must be either 'X' or 'H'")

        # Convert arms to world coordinates
        arms_world = np.array([np.dot(R, arm) for arm in arms])

        # Draw the quadcopter arms and propellers
        for i, arm in enumerate(arms_world):
            # calculate the end point of the arm
            end_point = position + arm

            # draw the arm
            (line,) = ax.plot([x, end_point[0]], [y, end_point[1]], [z, end_point[2]], color=color, linewidth=2)
            artists.append(line)

            # draw the propeller
            prop_color = "red" if i % 2 == 0 else "green"  # alternate colors
            circle_radius = self.arm_length / 4
            theta = np.linspace(0, 2 * np.pi, 20)

            # choose the rotation axis based on the arm index
            if quad_type == "X":
                if i == 0 or i == 3:  # front-left and back-right
                    v1 = np.array([0, 0, 1])
                    v2 = np.array([0, 1, 0])
                else:  # front-right and back-left
                    v1 = np.array([0, 0, 1])
                    v2 = np.array([1, 0, 0])
            elif quad_type == "H":
                if i == 0 or i == 2:  # front and back
                    v1 = np.array([0, 1, 0])
                    v2 = np.array([0, 0, 1])
                else:  # left and right
                    v1 = np.array([1, 0, 0])
                    v2 = np.array([0, 0, 1])
            else:
                raise ValueError("Quadcopter type must be either 'X' or 'H'")

            # rotate the vectors to match the arm orientation
            v1 = np.dot(R, v1)
            v2 = np.dot(R, v2)

            # draw the propeller circle
            circle_points = np.array([end_point + circle_radius * (np.cos(t) * v1 + np.sin(t) * v2) for t in theta])
            (prop_line,) = ax.plot(
                circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color=prop_color, linewidth=1.5
            )
            artists.append(prop_line)

        # draw the body of the quadcopter
        center_point = ax.scatter([x], [y], [z], color="black", s=30)
        artists.append(center_point)

        # draw the front vector
        front_vector = np.dot(R, np.array([self.arm_length * 1.2, 0, 0]))
        arrow = Arrow3D(
            x,
            y,
            z,
            front_vector[0],
            front_vector[1],
            front_vector[2],
            mutation_scale=10,
            lw=1,
            arrowstyle="-|>",
            color="red",
        )
        ax.add_artist(arrow)
        artists.append(arrow)

        return artists
