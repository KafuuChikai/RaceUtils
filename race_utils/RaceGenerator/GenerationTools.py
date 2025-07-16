import numpy as np
from typing import List, Optional, Union, Type
from race_utils.RaceGenerator.GateShape import (
    BaseShape,
    SingleBall,
    TrianglePrisma,
    RectanglePrisma,
    PentagonPrisma,
    HexagonPrisma,
)
from race_utils.RaceGenerator.BaseRaceClass import State, Gate
from ruamel.yaml.scalarstring import SingleQuotedScalarString

KEYS_TO_QUOTE = ["type", "name"]


def quote_specific_keys(
    data: Union[dict, List[dict], str], keys_to_quote: List[str] = KEYS_TO_QUOTE
) -> Union[dict, List[dict], str]:
    """Recursively quote specific keys in a dictionary or list of dictionaries.

    Parameters
    ----------
    data : Union[dict, List[dict], str]
        The data to process, which can be a dictionary, a list of dictionaries, or a string.
    keys_to_quote : List[str]
        The keys that should be quoted in the dictionary.

    Returns
    -------
    Union[dict, List[dict], str]
        The processed data with specified keys quoted as SingleQuotedScalarString.

    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key in keys_to_quote and isinstance(value, str):
                data[key] = SingleQuotedScalarString(value)
            elif isinstance(value, (dict, list)):
                quote_specific_keys(value, keys_to_quote)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, str):
                data[idx] = SingleQuotedScalarString(item)
            elif isinstance(item, (dict, list)):
                quote_specific_keys(item, keys_to_quote)
    return data


def get_shape_class(gate_shape: str) -> BaseShape:
    """Get the class of the gate shape based on its type.

    Parameters
    ----------
    gate_shape : str
        The type of the gate shape as a string.

    Returns
    -------
    BaseShape
        The class corresponding to the gate shape type.

    """
    shape_classes = {
        "SingleBall": SingleBall,
        "TrianglePrisma": TrianglePrisma,
        "RectanglePrisma": RectanglePrisma,
        "PentagonPrisma": PentagonPrisma,
        "HexagonPrisma": HexagonPrisma,
    }
    return shape_classes[gate_shape]


def create_state(state_kwargs: dict) -> State:
    """Create a State object from the given parameters.

    Parameters
    ----------
    state_kwargs : dict
        A dictionary containing the parameters for the state.

    Returns
    -------
    State
        An instance of the State class initialized with the provided parameters.

    """
    missing_pos = True if "pos" not in state_kwargs else False
    if missing_pos:
        raise ValueError("Missing parameters for State: pos")

    return State(**state_kwargs)


def create_gate(
    gate_type: Union[Type[BaseShape], str],
    position: Union[List[float], np.ndarray],
    stationary: bool,
    shape_kwargs: dict,
    name: Optional[str] = None,
) -> Gate:
    """Create a Gate object with the specified parameters.

    Parameters
    ----------
    gate_type : Union[Type[BaseShape], str]
        The type of the gate shape, either as a class or a string.
    position : Union[List[float], np.ndarray]
        The position of the gate in 3D space.
    stationary : bool
        Whether the gate is stationary or not.
    shape_kwargs : dict
        A dictionary containing the parameters for the gate shape.
    name : Optional[str]
        The name of the gate. If not provided, it will be generated automatically.

    Returns
    -------
    Gate
        An instance of the Gate class initialized with the provided parameters.

    """
    shape_params = {
        "SingleBall": ["radius", "margin"],
        "TrianglePrisma": ["rpy", "length", "midpoints", "width", "height", "margin"],
        "RectanglePrisma": [
            "rpy",
            "length",
            "midpoints",
            "width",
            "height",
            "marginW",
            "marginH",
        ],
        "PentagonPrisma": ["rpy", "length", "midpoints", "radius", "margin"],
        "HexagonPrisma": ["rpy", "length", "midpoints", "side", "margin"],
    }

    if isinstance(gate_type, str):
        gate_type = get_shape_class(gate_type)
    gate_type_name = gate_type.__name__

    missing_params = [
        param for param in shape_params[gate_type_name] if param not in shape_kwargs
    ]
    if missing_params:
        raise ValueError(
            f"Missing parameters for {gate_type_name}: {', '.join(missing_params)}"
        )

    return Gate(
        gate_shape=gate_type(**shape_kwargs),
        position=position,
        stationary=stationary,
        name=name,
    )
