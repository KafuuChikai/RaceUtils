import numpy as np
from typing import List, Union

########################################
###### BASE CLASS FOR GATE SHAPES ######
########################################


class BaseShape:
    def __init__(self, gate_type: str):
        """Initialize the base shape class.

        Parameters
        ----------
        gate_type : str
            The type of the gate shape.
        
        """
        self.type = gate_type

    def get_shape_info(self) -> dict:
        """Get the shape information as a dictionary.
        
        Returns
        -------
        dict
            A dictionary containing the shape type and its properties.
        
        """
        return vars(self)


class BasePrisma(BaseShape):
    def __init__(
        self,
        gate_type: str,
        rpy: Union[List[float], np.ndarray],
        length: float,
        midpoints: int,
    ):
        """Initialize the base prisma class.

        Parameters
        ----------
        gate_type : str
            The type of the gate shape.
        rpy : Union[List[float], np.ndarray]
            The roll, pitch, and yaw of the gate.
        length : float
            The length of the gate.
        midpoints : int
            The number of midpoints in the gate.
        
        """
        super().__init__(gate_type)
        self.rpy = rpy if isinstance(rpy, list) else rpy.tolist()
        self.length = length
        self.midpoints = midpoints


########################################
###### GATE SHAPES IMPLEMENTATION ######
########################################


class SingleBall(BaseShape):
    def __init__(self, radius: float, margin: float):
        """Initialize the single ball gate shape.

        Parameters
        ----------
        radius : float
            The radius of the ball.
        margin : float
            The margin around the ball.

        """
        super().__init__("SingleBall")
        self.radius = radius
        self.margin = margin


class TrianglePrisma(BasePrisma):
    def __init__(
        self,
        rpy: Union[List[float], np.ndarray],
        length: float,
        midpoints: int,
        width: float,
        height: float,
        margin: float,
    ):
        """Initialize the triangle prisma gate shape.

        Parameters
        ----------
        rpy : Union[List[float], np.ndarray]
            The roll, pitch, and yaw of the gate.
        length : float
            The length of the gate.
        midpoints : int
            The number of midpoints in the gate.
        width : float
            The width of the triangle.
        height : float
            The height of the triangle.
        margin : float
            The margin around the triangle.

        """
        super().__init__("TrianglePrisma", rpy, length, midpoints)
        self.width = width
        self.height = height
        self.margin = margin


class RectanglePrisma(BasePrisma):
    def __init__(
        self,
        rpy: Union[List[float], np.ndarray],
        length: float,
        midpoints: int,
        width: float,
        height: float,
        marginW: float,
        marginH: float,
    ):
        """Initialize the rectangle prisma gate shape.

        Parameters
        ----------
        rpy : Union[List[float], np.ndarray]
            The roll, pitch, and yaw of the gate.
        length : float
            The length of the gate.
        midpoints : int
            The number of midpoints in the gate.
        width : float
            The width of the rectangle.
        height : float
            The height of the rectangle.
        marginW : float
            The margin around the width of the rectangle.
        marginH : float
            The margin around the height of the rectangle.

        """
        super().__init__("RectanglePrisma", rpy, length, midpoints)
        self.width = width
        self.height = height
        self.marginW = marginW
        self.marginH = marginH


class PentagonPrisma(BasePrisma):
    def __init__(
        self,
        rpy: Union[List[float], np.ndarray],
        length: float,
        midpoints: int,
        radius: float,
        margin: float,
    ):
        """Initialize the pentagon prisma gate shape.

        Parameters
        ----------
        rpy : Union[List[float], np.ndarray]
            The roll, pitch, and yaw of the gate.
        length : float
            The length of the gate.
        midpoints : int
            The number of midpoints in the gate.
        radius : float
            The radius of the pentagon.
        margin : float
            The margin around the pentagon.
            
        """
        super().__init__("PentagonPrisma", rpy, length, midpoints)
        self.radius = radius
        self.margin = margin


class HexagonPrisma(BasePrisma):
    def __init__(
        self,
        rpy: Union[List[float], np.ndarray],
        length: float,
        midpoints: int,
        side: float,
        margin: float,
    ):
        """Initialize the hexagon prisma gate shape.

        Parameters
        ----------
        rpy : Union[List[float], np.ndarray]
            The roll, pitch, and yaw of the gate.
        length : float
            The length of the gate.
        midpoints : int
            The number of midpoints in the gate.
        side : float
            The length of each side of the hexagon.
        margin : float
            The margin around the hexagon.
            
        """
        super().__init__("HexagonPrisma", rpy, length, midpoints)
        self.side = side
        self.margin = margin
