import numpy as np
from typing import Callable, Any

def validate_objective(objective: Callable[[Any], Any]) -> Callable[[Any], float]:
    """
    Quick and basic validation of the user-provided objective.
    """
    def wrapper(input, *objective_args: Any):
        out = objective(input, *objective_args)
        if not isinstance(out, float):
            try:
                out = float(out)
            except:
                raise Exception(f"Was not able to convert the objective's output to a float. Objective type is {type(out)}")
        return out
    
    return wrapper

def normalize_objective(
        objective: Callable[[Any], Any],
        mean: float,
        stdev: float,
        epsilon: float = 1e-8
    )-> Callable[[Any, float, float, float], float]:
    """
    Use the mean and standard deviation of seeds to normalize the objective.
    """
    def wrapper(input, *objective_args: Any) -> float:
        valid_obj = validate_objective(objective)
        obj = valid_obj(input, *objective_args)
        return (obj - mean) / (stdev + epsilon)
    
    return wrapper