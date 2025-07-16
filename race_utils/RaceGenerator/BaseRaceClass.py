import numpy as np
from typing import List, Optional, Union
from race_utils.RaceGenerator.GateShape import BaseShape

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)
yaml.default_flow_style = False


class BaseRaceClass:
    def __init__(self):
        pass

    def to_ordered_dict(self, ordered_keys: List[str], dict: dict) -> CommentedMap:
        """Convert a dictionary to an ordered dictionary.

        Parameters
        ----------
        ordered_keys : List[str]
            The keys in the order they should appear.
        dict : dict
            The dictionary to convert.

        Returns
        -------
        CommentedMap
            The ordered dictionary.

        """
        data = CommentedMap()
        for key in ordered_keys:
            value = dict[key]
            if isinstance(dict[key], list):
                commented_seq = CommentedSeq(value)
                commented_seq.fa.set_flow_style()
                data[key] = commented_seq
            else:
                data[key] = value
        return data


class State(BaseRaceClass):
    def __init__(
        self,
        pos: Union[List[float], np.ndarray],  # (3,) - position [x, y, z]
        vel: Optional[
            Union[List[float], np.ndarray]
        ] = None,  # (3,) - velocity [vx, vy, vz]
        acc: Optional[
            Union[List[float], np.ndarray]
        ] = None,  # (3,) - acceleration [ax, ay, az]
        jer: Optional[
            Union[List[float], np.ndarray]
        ] = None,  # (3,) - jerk [jx, jy, jz]
        rot: Optional[
            Union[List[float], np.ndarray]
        ] = None,  # (4,) - quaternion [w, x, y, z]
        cthrustmass: Optional[float] = None,  # (1,) - collective thrust
        euler: Optional[
            Union[List[float], np.ndarray]
        ] = None,  # (3,) - euler angles [roll, pitch, yaw]
    ):
        """Initialize a state with position, velocity, acceleration, jerk, rotation, collective thrust mass, and euler angles.

        Parameters
        ----------
        pos : Union[List[float], np.ndarray]
            Position of the drone in 3D space.
        vel : Optional[Union[List[float], np.ndarray]], default=None
            Velocity of the drone in 3D space.
        acc : Optional[Union[List[float], np.ndarray]], default=None
            Acceleration of the drone in 3D space.
        jer : Optional[Union[List[float], np.ndarray]], default=None
            Jerk of the drone in 3D space.
        rot : Optional[Union[List[float], np.ndarray]], default=None
            Quaternion representing the rotation of the drone.
        cthrustmass : Optional[float], default=None
            Collective thrust mass of the drone.
        euler : Optional[Union[List[float], np.ndarray]], default=None
            Euler angles representing the orientation of the drone.

        """
        self.pos = pos if isinstance(pos, list) else pos.tolist()
        self.vel = (
            [0.0, 0.0, 0.0]
            if vel is None
            else (vel if isinstance(vel, list) else vel.tolist())
        )
        self.acc = (
            [0.0, 0.0, 0.0]
            if acc is None
            else (acc if isinstance(acc, list) else acc.tolist())
        )
        self.jer = (
            [0.0, 0.0, 0.0]
            if jer is None
            else (jer if isinstance(jer, list) else jer.tolist())
        )
        self.rot = (
            [0.0, 0.0, 0.0, 0.0]
            if rot is None
            else (rot if isinstance(rot, list) else rot.tolist())
        )
        self.cthrustmass = 9.8066 if cthrustmass is None else cthrustmass
        self.euler = (
            [0.0, 0.0, 0.0]
            if euler is None
            else (euler if isinstance(euler, list) else euler.tolist())
        )

    def to_dict(self) -> dict:
        """Convert the state to a dictionary.

        Returns
        -------
        dict
            The dictionary representation of the state.

        """
        return vars(self)

    def to_ordered_dict(self) -> CommentedMap:
        """Convert the state to an ordered dictionary.

        Returns
        -------
        CommentedMap
            The ordered dictionary representation of the state.

        """
        ordered_keys = ["pos", "vel", "acc", "jer", "rot", "cthrustmass", "euler"]
        return super().to_ordered_dict(ordered_keys=ordered_keys, dict=self.to_dict())


class Gate(BaseRaceClass):
    def __init__(
        self,
        gate_shape: BaseShape,
        position: Union[List[float], np.ndarray],
        stationary: bool,
        name: Optional[str] = None,
    ):
        """Initialize a gate with a shape, position, and whether it is stationary.

        Parameters
        ----------
        gate_shape : BaseShape
            The shape of the gate.
        position : Union[List[float], np.ndarray]
            The position of the gate in 3D space.
        stationary : bool
            Whether the gate is stationary or not.
        name : Optional[str], default=None
            The name of the gate. If None, a default name will be generated.

        """
        self.shape = gate_shape
        self.name = name
        self.position = position if isinstance(position, list) else position.tolist()
        self.stationary = stationary
        self.SHAPE_ORDER_KEYS = {
            "SingleBall": [
                "type",
                "name",
                "position",
                "radius",
                "margin",
                "stationary",
            ],
            "TrianglePrisma": [
                "type",
                "name",
                "position",
                "rpy",
                "width",
                "height",
                "margin",
                "length",
                "midpoints",
                "stationary",
            ],
            "RectanglePrisma": [
                "type",
                "name",
                "position",
                "rpy",
                "width",
                "height",
                "marginW",
                "marginH",
                "length",
                "midpoints",
                "stationary",
            ],
            "PentagonPrisma": [
                "type",
                "name",
                "position",
                "rpy",
                "radius",
                "margin",
                "length",
                "midpoints",
                "stationary",
            ],
            "HexagonPrisma": [
                "type",
                "name",
                "position",
                "rpy",
                "side",
                "margin",
                "length",
                "midpoints",
                "stationary",
            ],
        }

    def to_dict(self) -> dict:
        """Convert the gate to a dictionary.

        Returns
        -------
        dict
            The dictionary representation of the gate.

        """
        data = {
            **self.shape.get_shape_info(),
            "position": self.position,
            "stationary": self.stationary,
        }
        if self.name is not None:
            data["name"] = self.name
        return data

    def to_ordered_dict(self) -> CommentedMap:
        """Convert the gate to an ordered dictionary.

        Returns
        -------
        CommentedMap
            An ordered dictionary representation of the gate.

        """
        ordered_keys = self.SHAPE_ORDER_KEYS[self.shape.type]
        if self.name is None:
            ordered_keys.remove("name")
        return super().to_ordered_dict(ordered_keys=ordered_keys, dict=self.to_dict())

    def set_position(self, position: Union[List[float], np.ndarray]) -> None:
        """Set the position of the gate.

        Parameters
        ----------
        position : Union[List[float], np.ndarray]
            The new position of the gate.

        """
        if self.stationary:
            raise ValueError("Cannot set position for a stationary gate.")
        self.position = position if isinstance(position, list) else position.tolist()
