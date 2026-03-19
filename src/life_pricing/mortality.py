from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class GompertzConfig:
    """
    Configuration for Gompertz mortality.

    Force of mortality:
        mu_x = B * c^x

    Parameters
    ----------
    B : float
        Scale parameter (must be > 0).
    c : float
        Growth parameter (must be > 1).
    """
    B: float
    c: float

    def __post_init__(self) -> None:
        """Validate parameters after initialisation."""
        if self.B <= 0:
            raise ValueError("B must be > 0.")
        if self.c <= 1:
            raise ValueError("c must be > 1.")

def mu(age, config: GompertzConfig):
    """
    Force of mortality μ_x at age x.
    Works for scalars or numpy arrays.
    """
    age = np.asarray(age)
    return config.B * (config.c ** age)

def survival_prob(age, t, config: GompertzConfig):
    """
    Survival probability from age x to x+t:

    _t p_x = exp( - (B / ln(c)) * (c^(x+t) - c^x) )
    """
    age = np.asarray(age)
    t = np.asarray(t)

    integral = (config.B / np.log(config.c)) * (
        config.c ** (age + t) - config.c ** age
    )
    return np.exp(-integral)


def death_prob(age, config: GompertzConfig, t=1.0):
    """
    Probability of death between age x and x+t.
    """
    return 1.0 - survival_prob(age, t, config)