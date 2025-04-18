import numpy as np
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from race_utils.RaceVisualizer.Drawers.BaseDrawer import Arrow3D, BaseDrawer


class QuadcopterDrawer(BaseDrawer):
    """Class to draw a quadcopter in 3D space using matplotlib"""

    def __init__(
        self,
        ax: Axes3D,
        arm_length: float = 0.2,
        quad_type: str = "X",
        color: str = "black",
        front_color: str = "chartreuse",
        back_color: str = "deepskyblue",
    ):
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
        super().__init__(ax=ax)
        self.arm_length = arm_length
        self.quad_type = quad_type
        self.color = color
        self.front_color = front_color
        self.back_color = back_color

    def draw(self, position: np.ndarray, attitude: np.ndarray, show_body_axis: bool = False) -> list:
        """Draw a quadcopter in 3D space

        Parameters
        ---------
        ax: matplotlib 3D axis object
            The axis on which to draw the quadcopter
        position: np.ndarray
            [x, y, z] position of the quadcopter
        attitude: np.ndarray
            [roll, pitch, yaw] Euler angles (in radians) or quaternion [qx, qy, qz, qw]

        Returns:
        --------
        artists: list
            List of artists created for the quadcopter, used for later updates or deletions

        """
        x, y, z = position
        artists = []
        ax = self.ax

        # convert attitude to rotation matrix
        if len(attitude) == 3:  # use [roll, pitch, yaw] for euler angles
            R = Rotation.from_euler("xyz", attitude, degrees=False).as_matrix()
        elif len(attitude) == 4:  # use [qx, qy, qz, qw] for quaternion
            R = Rotation.from_quat(attitude).as_matrix()
        else:
            raise ValueError("Attitude must be either Euler angles [roll, pitch, yaw] or quaternion [qx, qy, qz, qw]")

        # draw the quadcopter arms and propellers
        artists = self.draw_propeller(ax=ax, total_artists=artists, position=position, rotation_matrix=R)

        # draw the body of the quadcopter
        artists = self.draw_body(
            ax=ax,
            total_artists=artists,
            position=position,
            rotation_matrix=R,
            top_color="black",
            side_color="gray",
            edge_color="white",
            alpha=0.5,
        )

        base_height = 0.05 * self.arm_length
        body_height = 0.5 * self.arm_length
        cylinder_radius = 0.1 * self.arm_length
        cylinder_height_h = 0.25 * self.arm_length
        cylinder_height_l = 0.2 * self.arm_length
        cylinder_h_center = position + np.dot(R, np.array([0, 0, base_height + body_height + cylinder_height_h / 2]))
        cylinder_l_centers_1 = position + np.dot(R, np.array([0, 0, base_height]))
        cylinder_l_centers_2 = position + np.dot(R, np.array([0, 0, base_height]))
        cylinder_l_centers_2 = position + np.dot(R, np.array([0, 0, base_height]))
        artists = self.draw_cylinder(
            total_artists=artists,
            center=cylinder_h_center,
            height=cylinder_height_h,
            radius=cylinder_radius,
            rotation_matrix=R,
            color="white",
            alpha=1,
        )
        # artists = self.draw_cylinder(ax=ax, total_artists=artists, center=cylinder_h_center, height=cylinder_height_l, radius=cylinder_radius, rotation_matrix=R, color='white', alpha=1)

        ball_radius = 0.15 * self.arm_length
        ball_h_center = position + np.dot(R, np.array([0, 0, base_height + body_height + cylinder_height_h]))
        ball_l_centers_1 = position + np.dot(R, np.array([0, 0, base_height]))
        ball_l_centers_2 = position + np.dot(R, np.array([0, 0, base_height]))
        ball_l_centers_2 = position + np.dot(R, np.array([0, 0, base_height]))
        theta = np.linspace(0, np.pi, 20)
        phi = np.linspace(0, 2 * np.pi, 20)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        x_ball = ball_radius * np.sin(theta_grid) * np.cos(phi_grid)
        y_ball = ball_radius * np.sin(theta_grid) * np.sin(phi_grid)
        z_ball = ball_radius * np.cos(theta_grid)
        x_ball += ball_h_center[0]
        y_ball += ball_h_center[1]
        z_ball += ball_h_center[2]
        ball = ax.plot_surface(x_ball, y_ball, z_ball, color="gray", alpha=1)
        artists.append(ball)

        # draw the body axis vector
        if show_body_axis:
            body_x = np.dot(R, 3 * self.arm_length * np.array([1, 0, 0]))  # x-axis
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
            ax.add_artist(arrow_x)
            artists.append(arrow_x)

            body_y = np.dot(R, 3 * self.arm_length * np.array([0, 1, 0]))  # y-axis
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
            ax.add_artist(arrow_y)
            artists.append(arrow_y)

            body_z = np.dot(R, 3 * self.arm_length * np.array([0, 0, 1]))  # z-axis
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
            ax.add_artist(arrow_z)
            artists.append(arrow_z)

        return artists

    def draw_propeller(
        self, ax: Axes3D, total_artists: list, position: np.ndarray, rotation_matrix: np.ndarray
    ) -> list:
        """Draw the propellers of the quadcopter

        Parameters
        ----------
        ax: matplotlib 3D axis object
            The axis on which to draw the quadcopter
        total_artists: list
            List of artists created for the quadcopter, used for later updates or deletions
        position: np.ndarray
            [x, y, z] position of the quadcopter
        rotation_matrix: np.ndarray
            3x3 rotation matrix for the quadcopter orientation

        Returns:
        -------
        artists: list
            List of artists created for the quadcopter, used for later updates or deletions

        """
        x, y, z = position
        quad_type = self.quad_type
        artists = []
        # Define the quadcopter arms in the body frame
        if quad_type == "X":
            arms = np.array(
                [
                    [self.arm_length / np.sqrt(2), -self.arm_length / np.sqrt(2), 0],  # front-left: id 0
                    [self.arm_length / np.sqrt(2), self.arm_length / np.sqrt(2), 0],  # front-right: id 1
                    [-self.arm_length / np.sqrt(2), -self.arm_length / np.sqrt(2), 0],  # back-left: id 2
                    [-self.arm_length / np.sqrt(2), self.arm_length / np.sqrt(2), 0],  # back-right: id 3
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
        arms_world = np.array([np.dot(rotation_matrix, arm) for arm in arms])
        # Draw the quadcopter arms and propellers
        for i, arm in enumerate(arms_world):
            # calculate the end point of the arm
            end_point = position + arm

            # draw the arm
            (line,) = ax.plot([x, end_point[0]], [y, end_point[1]], [z, end_point[2]], color=self.color, linewidth=2)
            artists.append(line)

            # draw the propeller
            if quad_type == "X":
                prop_color = self.front_color if i < 2 else self.back_color  # alternate colors
            else:
                prop_color = self.color
            circle_radius = self.arm_length * 1.1 / 2
            theta = np.linspace(0, 2 * np.pi, 20)

            # rotate the vectors to match the arm orientation
            v1 = np.dot(rotation_matrix, np.array([1, 0, 0]))
            v2 = np.dot(rotation_matrix, np.array([0, 1, 0]))

            # draw the propeller circle with thinkness
            circle_points = np.array([end_point + circle_radius * (np.cos(t) * v1 + np.sin(t) * v2) for t in theta])
            (prop_line,) = ax.plot(
                circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color=prop_color, linewidth=2.5
            )
            artists.append(prop_line)
        total_artists.extend(artists)
        return total_artists

    def draw_body(
        self,
        ax,
        total_artists,
        position,
        rotation_matrix,
        top_color="black",
        side_color="gray",
        edge_color="white",
        alpha=0.5,
    ):
        base_height = 0.05 * self.arm_length
        body_height = 0.5 * self.arm_length
        half_size_l = self.arm_length * 1.3 / 2
        half_size_w = self.arm_length * 0.8 / 2

        # define the vertices of the cube in the body frame
        cube_vertices = np.array(
            [
                [-half_size_w, -half_size_l, base_height],  # 0: left bottom back
                [half_size_w, -half_size_l, base_height],  # 1: right bottom back
                [half_size_w, half_size_l, base_height],  # 2: right bottom front
                [-half_size_w, half_size_l, base_height],  # 3: left bottom front
                [-half_size_w, -half_size_l, base_height + body_height],  # 4: left top back
                [half_size_w, -half_size_l, base_height + body_height],  # 5: right top back
                [half_size_w, half_size_l, base_height + body_height],  # 6: right top front
                [-half_size_w, half_size_l, base_height + body_height],  # 7: left top front
            ]
        )

        # Convert cube vertices to world coordinates
        cube_vertices_world = np.array([np.dot(rotation_matrix, vertex) + position for vertex in cube_vertices])

        # define the faces of the cube
        faces = [
            [0, 1, 2, 3],  # 0: bottom face
            [4, 5, 6, 7],  # 1: top face
            [0, 1, 5, 4],  # 2: back face
            [2, 3, 7, 6],  # 3: front face
            [0, 3, 7, 4],  # 4: left face
            [1, 2, 6, 5],  # 5: right face
        ]

        # 为每个面创建多边形
        for face_id, face in enumerate(faces):
            # 获取这个面的顶点
            face_vertices = [cube_vertices_world[idx] for idx in face]

            # 创建3D多边形集合
            poly = Poly3DCollection([face_vertices], alpha=alpha)

            # 设置颜色，可以根据朝向不同设置不同颜色
            if face_id < 2:
                poly.set_color(top_color)
                poly.set_edgecolor(edge_color)
            else:
                poly.set_color(side_color)
                poly.set_edgecolor(edge_color)

            ax.add_collection3d(poly)
            total_artists.append(poly)
        return total_artists
