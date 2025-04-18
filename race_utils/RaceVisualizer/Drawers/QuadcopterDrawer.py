import numpy as np
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from race_utils.RaceVisualizer.Drawers.BaseDrawer import Arrow3D, BaseDrawer


class QuadcopterDrawer(BaseDrawer):
    """Class to draw a quadcopter in 3D space using matplotlib

    The quadcopter is the same as the real world quadcopter of NeSC, ZJU
    """

    def __init__(
        self,
        ax: Axes3D,
        show_body_axis: bool = False,
        arm_length: float = 0.2,
        arm_angle: float = np.pi * 65 / 180,
        quad_type: str = "X",
        arm_color: str = "black",
        front_color: str = "chartreuse",
        back_color: str = "deepskyblue",
        body_top_color: str = "gray",
        body_side_color: str = "black",
        body_edge_color: str = "white",
        body_alpha: float = 0.5,
    ):
        """Initialize the QuadcopterDrawer

        Parameters
        ---------
        ax: matplotlib 3D axis object
            The axis on which to draw the quadcopter
        show_body_axis: bool
            Whether to show the body axis of the quadcopter (default is False)
        arm_length: float
            Length of the quadcopter arms (default is 0.2)
        arm_angle: float
            Angle of the quadcopter arms (default is np.pi / 3)
        quad_type: str
            Type of quadcopter ('X' or 'H', default is 'X')
        arm_color: str
            Color of the quadcopter arms (default is 'black')
        front_color: str
            Color of the front propellers (default is 'chartreuse')
        back_color: str
            Color of the back propellers (default is 'deepskyblue')
        body_top_color: str
            Color of the top face of the quadcopter body (default is 'black')
        body_side_color: str
            Color of the side faces of the quadcopter body (default is 'gray')
        body_edge_color: str
            Color of the edges of the quadcopter body (default is 'white')
        body_alpha: float
            Alpha value for the quadcopter body (default is 0.5)

        """
        super().__init__(ax=ax, show_body_axis=show_body_axis)
        self.arm_length = arm_length
        self.arm_angle = arm_angle
        self.quad_type = quad_type
        self.arm_color = arm_color
        self.front_color = front_color
        self.back_color = back_color
        self.body_top_color = body_top_color
        self.body_side_color = body_side_color
        self.body_edge_color = body_edge_color
        self.body_alpha = body_alpha
        self._compute_body_data()

    def _compute_body_data(self) -> None:
        """Compute the body data for the quadcopter
        This method computes the size and position of the quadcopter body
        and arms based on the arm length and angle.

        """
        self.base_height = 0.02 * self.arm_length
        self.body_height = 0.4 * self.arm_length
        self.half_size_l = self.arm_length * 0.8 / 2
        self.half_size_w = self.arm_length * 0.5 / 2

    def draw(self, position: np.ndarray, attitude: np.ndarray) -> list:
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
        # clear the previous artists
        self.total_artists = []

        # convert attitude to rotation matrix
        if len(attitude) == 3:  # use [roll, pitch, yaw] for euler angles
            R = Rotation.from_euler("xyz", attitude, degrees=False).as_matrix()
        elif len(attitude) == 4:  # use [qx, qy, qz, qw] for quaternion
            R = Rotation.from_quat(attitude).as_matrix()
        else:
            raise ValueError("Attitude must be either Euler angles [roll, pitch, yaw] or quaternion [qx, qy, qz, qw]")

        # draw the quadcopter arms and propellers
        self._draw_propeller(position=position, rotation_matrix=R)

        # draw the body of the quadcopter
        self._draw_body(
            position=position,
            rotation_matrix=R,
        )

        # draw the motion balls
        self._draw_motion_balls(
            position=position,
            rotation_matrix=R,
        )

        # draw the body axis vector
        if self.show_body_axis:
            self._draw_body_axis(position=position, rotation_matrix=R)

        return self.total_artists

    def _draw_propeller(self, position: np.ndarray, rotation_matrix: np.ndarray) -> None:
        """Draw the propellers of the quadcopter

        Parameters
        ----------
        position: np.ndarray
            [x, y, z] position of the quadcopter
        rotation_matrix: np.ndarray
            3x3 rotation matrix for the quadcopter orientation

        """
        ax = self.ax
        x, y, z = position
        quad_type = self.quad_type
        artists = []
        # Define the quadcopter arms in the body frame
        if quad_type == "X":
            arms = np.array(
                [
                    [
                        self.arm_length * np.cos(self.arm_angle),
                        -self.arm_length * np.cos(self.arm_angle),
                        0,
                    ],  # front-left: id 0
                    [
                        self.arm_length * np.cos(self.arm_angle),
                        self.arm_length * np.cos(self.arm_angle),
                        0,
                    ],  # front-right: id 1
                    [
                        -self.arm_length * np.cos(self.arm_angle),
                        -self.arm_length * np.cos(self.arm_angle),
                        0,
                    ],  # back-left: id 2
                    [
                        -self.arm_length * np.cos(self.arm_angle),
                        self.arm_length * np.cos(self.arm_angle),
                        0,
                    ],  # back-right: id 3
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
            (line,) = ax.plot(
                [x, end_point[0]], [y, end_point[1]], [z, end_point[2]], color=self.arm_color, linewidth=2
            )
            artists.append(line)
            end_point_rotor = end_point + np.dot(rotation_matrix, np.array([0, 0, -0.06 * self.arm_length]))
            # draw the rotor
            self._draw_cylinder(
                center=end_point_rotor,
                height=0.12 * self.arm_length,
                radius=0.06 * self.arm_length,
                rotation_matrix=rotation_matrix,
                color=self.arm_color,
                alpha=1.0,
                closed=True,
            )

            # draw the propeller
            end_point_prop = end_point + np.dot(rotation_matrix, np.array([0, 0, -0.2 * self.arm_length]))
            if quad_type == "X":
                prop_color = self.front_color if i < 2 else self.back_color  # alternate colors
            else:
                prop_color = self.front_color
            circle_radius = 0.4 * self.arm_length
            circle_height = 0.1 * self.arm_length
            self._draw_cylinder_ring(
                center=end_point_prop,
                height=circle_height,
                outer_radius=1.0 * circle_radius,
                inner_radius=0.8 * circle_radius,
                rotation_matrix=rotation_matrix,
                color=prop_color,
                alpha=0.3,
            )
            self._draw_cylinder(
                center=end_point_prop,
                height=circle_height,
                radius=0.8 * circle_radius,
                rotation_matrix=rotation_matrix,
                color=prop_color,
                alpha=0.1,
                closed=True,
            )

        self.total_artists.extend(artists)

    def _draw_body(
        self,
        position: np.ndarray,
        rotation_matrix: np.ndarray,
    ) -> None:
        """Draw the body of the quadcopter

        Parameters
        ----------
        position: np.ndarray
            [x, y, z] position of the quadcopter
        rotation_matrix: np.ndarray
            3x3 rotation matrix for the quadcopter orientation

        """
        ax = self.ax
        base_height = self.base_height
        body_height = self.body_height
        half_size_l = self.half_size_l
        half_size_w = self.half_size_w

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

        # create a 3D polygon for each face of the cube
        for face_id, face in enumerate(faces):
            # get the vertices of the face in world coordinates
            face_vertices = [cube_vertices_world[idx] for idx in face]

            # create a Poly3DCollection for the face
            poly = Poly3DCollection([face_vertices], alpha=self.body_alpha)

            # set the color and edge color of the face
            if face_id < 2:
                poly.set_color(self.body_top_color)
                poly.set_edgecolor(self.body_edge_color)
            else:
                poly.set_color(self.body_side_color)
                poly.set_edgecolor(self.body_edge_color)

            ax.add_collection3d(poly)
            self.total_artists.append(poly)

    def _draw_motion_balls(
        self,
        position: np.ndarray,
        rotation_matrix: np.ndarray,
    ) -> None:
        """Draw the motion balls of the quadcopter

        Parameters
        ----------
        position: np.ndarray
            [x, y, z] position of the quadcopter
        rotation_matrix: np.ndarray
            3x3 rotation matrix for the quadcopter orientation

        """
        # compute the size and position of the base
        base_height = self.base_height
        body_height = self.body_height
        half_size_l = self.half_size_l
        half_size_w = self.half_size_w

        # compute the size and position of the motion cylinders
        cylinder_radius = 0.03 * self.arm_length
        cylinder_height_h = 0.25 * self.arm_length
        cylinder_height_l = 0.1 * self.arm_length
        cylinder_h_center = position + np.dot(
            rotation_matrix, np.array([0, 0, base_height + body_height + cylinder_height_h / 2])
        )
        cylinder_l_centers_1 = position + np.dot(
            rotation_matrix, np.array([0.8 * half_size_w, 0, base_height + body_height + cylinder_height_l / 2])
        )
        cylinder_l_centers_2 = position + np.dot(
            rotation_matrix,
            np.array([-0.5 * half_size_w, 0.8 * half_size_l, base_height + body_height + cylinder_height_l / 2]),
        )
        cylinder_l_centers_3 = position + np.dot(
            rotation_matrix,
            np.array([-0.5 * half_size_w, -0.8 * half_size_l, base_height + body_height + cylinder_height_l / 2]),
        )

        # compute the size and position of the motion balls
        ball_radius = 0.05 * self.arm_length
        ball_h_center = cylinder_h_center + np.dot(rotation_matrix, np.array([0, 0, cylinder_height_h / 2]))
        ball_l_centers_1 = cylinder_l_centers_1 + np.dot(rotation_matrix, np.array([0, 0, cylinder_height_l / 2]))
        ball_l_centers_2 = cylinder_l_centers_2 + np.dot(rotation_matrix, np.array([0, 0, cylinder_height_l / 2]))
        ball_l_centers_3 = cylinder_l_centers_3 + np.dot(rotation_matrix, np.array([0, 0, cylinder_height_l / 2]))

        # draw the motion cylinders high
        self._draw_cylinder(
            center=cylinder_h_center,
            height=cylinder_height_h,
            radius=cylinder_radius,
            rotation_matrix=rotation_matrix,
            color="white",
            alpha=1,
            closed=False,
        )

        # draw the motion cylinders low 1
        self._draw_cylinder(
            center=cylinder_l_centers_1,
            height=cylinder_height_l,
            radius=cylinder_radius,
            rotation_matrix=rotation_matrix,
            color="white",
            alpha=1,
            closed=False,
        )

        # draw the motion cylinders low 2
        self._draw_cylinder(
            center=cylinder_l_centers_2,
            height=cylinder_height_l,
            radius=cylinder_radius,
            rotation_matrix=rotation_matrix,
            color="white",
            alpha=1,
            closed=False,
        )

        # draw the motion cylinders low 3
        self._draw_cylinder(
            center=cylinder_l_centers_3,
            height=cylinder_height_l,
            radius=cylinder_radius,
            rotation_matrix=rotation_matrix,
            color="white",
            alpha=1,
            closed=False,
        )

        # draw the motion balls
        self._draw_sphere(center=ball_h_center, radius=ball_radius, color="gray", alpha=1)
        self._draw_sphere(center=ball_l_centers_1, radius=ball_radius, color="gray", alpha=1)
        self._draw_sphere(center=ball_l_centers_2, radius=ball_radius, color="gray", alpha=1)
        self._draw_sphere(center=ball_l_centers_3, radius=ball_radius, color="gray", alpha=1)
