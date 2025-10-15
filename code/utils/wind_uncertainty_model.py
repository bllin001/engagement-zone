import numpy as np


def wrap_angle_deg(angle: float) -> float:
    """Wrap an angle in degrees to the [-180, 180] range, including -180 and 180 only once."""
    wrapped = ((angle + 180.0) % 360.0) - 180.0
    # If input is exactly 180 or -180, preserve sign
    if np.isclose(np.abs(angle) % 360, 180.0):
        return np.sign(angle) * 180.0
    return wrapped

def wind_effect(wind_speed, v_ground, entity_parameters, tol=1e-8):
    """
    Compute the effective ground speed given a fixed eastward wind.
    v_ground: ground speed (without wind effect)
    wind_speed: magnitude of the wind speed (eastward)
    entity_parameters: dictionary with 'heading' (radians)
    tol: tolerance for angle comparison
    Returns (v_ground_effect, wind_type, relative_angle)
    """
    
    heading = entity_parameters.get('heading', 0.0)
    
    # Wind Direction: Fixed eastward (deterministic)
    wind_direction = 0.0  # radians, eastward
    
    relative_angle = heading - wind_direction
    
    # Convert radians → degrees → wrap → back to radians
    relative_angle = np.deg2rad(wrap_angle_deg(np.rad2deg(relative_angle)))
    
    # Check alignment
    if abs(np.sin(relative_angle)) < tol:
        # Colinear (east or west)
        if np.cos(relative_angle) > 0:
            v_ground_effect = v_ground + wind_speed
            wind_type = "tailwind"
        else:
            v_ground_effect = max(0.0, v_ground - wind_speed)
            wind_type = "headwind"
    else:
        # Oblique (crosswind)
        v_ground_effect = np.sqrt(v_ground**2 + wind_speed**2 + 2*v_ground*wind_speed*np.cos(relative_angle))
        wind_type = "oblique"

    return v_ground_effect, wind_type, relative_angle

def ground_speed(entity_parameters, wind_speed):
    """
    Compute the ground speed magnitude given an entity's airspeed and a scalar eastward wind.

    Parameters
    ----------
    entity_parameters : dict
        Must contain 'heading' (radians) and 'speed' (scalar airspeed).
    wind_speed : float
        Scalar wind speed (eastward, fixed direction).

    Returns
    -------
    v_ground : float
        Ground speed magnitude (Euclidean norm).
    """
    heading = float(entity_parameters.get('heading', 0.0))
    speed = float(entity_parameters.get('speed', 0.0))

    # Wind Direction: Fixed eastward (deterministic)
    wind_speed_x, wind_speed_y = wind_speed, 0.0

    # Ground velocity components (east-west, north-south)
    v_ground_x = speed * np.cos(heading) + wind_speed_x
    v_ground_y = speed * np.sin(heading) + wind_speed_y

    return np.hypot(v_ground_x, v_ground_y)

def ground_speed_uncertainty(entity_parameters,  wind_speed_1 = 10, wind_speed_2 = 20 , prob_w1 = 0.6):
    """
    Compute the expected ground speed under two wind-speed scenarios
    with fixed probabilities (0.6 and 0.4).

    Parameters
    ----------
    entity_parameters : dict
        Contains 'heading' and 'speed'.
    wind_speed_1, wind_speed_2 : float
        Scalar eastward wind speeds.

    Returns
    -------
    v_ground_expected : float
        Expected ground speed (probability-weighted).
    v_ground_1 : float
        Ground speed under wind_speed_1.
    v_ground_2 : float
        Ground speed under wind_speed_2.
    """
    if not (0.0 <= prob_w1 <= 1.0):
        raise ValueError("prob_w1 must be in [0, 1]")

    prob_w2 = 1 - prob_w1

    v_ground_1 = ground_speed(entity_parameters, wind_speed_1)
    v_ground_2 = ground_speed(entity_parameters, wind_speed_2)

    # Expected (probability-weighted) ground speed
    v_ground_expected = prob_w1 * v_ground_1 + prob_w2 * v_ground_2
    return v_ground_expected, v_ground_1, v_ground_2

def wind_model(agent_parameters, intruder_parameters, t_fuel,
               method, wind_speed_1=10.0, wind_speed_2=20.0,
               prob_w1=0.6, R_baseline=None):
    """
    Compute μ (speed ratio) and R (range) under wind uncertainty for four methods.
    
    Assumptions:
    - Intruder = Pursuer (P) → uses intruder_parameters
    - Agent = Agent (A) → uses agent_parameters
    - Pursuer heading = 0 rad (east)
    - Wind direction = eastward (0 rad)
    - If intruder 'heading' is missing, it defaults to 0 rad via ground_speed()
    """

    if t_fuel < 0:
        raise ValueError("t_fuel must be non-negative")
    if not (0.0 <= prob_w1 <= 1.0):
        raise ValueError("prob_w1 must be in [0, 1]")

    prob_w2 = 1.0 - prob_w1
    eps = 1e-9
    
    agent_speed = agent_parameters.get('speed')
    pursuer_speed = intruder_parameters.get('speed')

    # ---------------------------------------------------------
    # Step 1: Compute ground-speed uncertainty using your function
    # ---------------------------------------------------------

    pursuer_ground_speed_expected, pursuer_ground_speed_1, pursuer_ground_speed_2 = ground_speed_uncertainty(intruder_parameters, wind_speed_1, wind_speed_2, prob_w1)

    # Agent: can have arbitrary heading → use your helper
    agent_ground_speed_expected, agent_ground_speed_1, agent_ground_speed_2 = ground_speed_uncertainty(agent_parameters, wind_speed_1, wind_speed_2, prob_w1)
    

    # ---------------------------------------------------------
    # Step 2: Apply four wind-uncertainty methods
    # ---------------------------------------------------------
    
    if method == "baseline":
        # Baseline (no wind uncertainty)
        mu = agent_parameters['speed'] / intruder_parameters['speed']
        R  = R_baseline

    elif method == "conservative":
        # Method 1 – Conservative (Tailwind Max)
        pursuer_effective_speed = max(pursuer_ground_speed_1, pursuer_ground_speed_2)
        mu = agent_ground_speed_expected / pursuer_effective_speed
        R  = pursuer_effective_speed / pursuer_speed

    elif method == "optimistic":
        # Method 2 – Optimistic (Headwind Min)
        pursuer_effective_speed = min(pursuer_ground_speed_1 - wind_speed_1*2, pursuer_ground_speed_2 - wind_speed_2*2)

        mu = agent_ground_speed_expected / pursuer_effective_speed
        R  = pursuer_effective_speed / pursuer_speed

    elif method == "cons_expected":
        # Method 3 – Conservative Expected (Expected of ratios)
        mu = (
            prob_w1 * (agent_ground_speed_1 / pursuer_ground_speed_1) +
            prob_w2 * (agent_ground_speed_2 / pursuer_ground_speed_2)
        )
        R = (
            prob_w1 * (pursuer_ground_speed_1) / pursuer_speed +
            prob_w2 * (pursuer_ground_speed_2) / pursuer_speed
        )

    elif method == "opt_expected":
        # Method 4 – Optimistic Expected (Ratio of expectations)
        mu = (
            prob_w1 * (agent_ground_speed_1 / (pursuer_ground_speed_1 - wind_speed_1*2)) +
            prob_w2 * (agent_ground_speed_2 / (pursuer_ground_speed_2 - wind_speed_2*2))
        )
        R = (
            prob_w1 * (pursuer_ground_speed_1 - wind_speed_1*2)/pursuer_speed +
            prob_w2 * (pursuer_ground_speed_2 - wind_speed_2*2)/pursuer_speed
        )

    else:
        raise ValueError("method must be 'conservative', 'optimistic', 'cons_expected', or 'opt_expected'")

    if R_baseline is not None:
        R = min(R, R_baseline)

    # ---------------------------------------------------------
    # Step 3: Return explicit results
    # ---------------------------------------------------------
    return {
        "method": method,
        "mu": mu,
        "R": R,
        "pursuer_ground_speeds": {
            "expected": pursuer_ground_speed_expected,
            "wind_1": pursuer_ground_speed_1,
            "wind_2": pursuer_ground_speed_2,
        },
        "agent_ground_speeds": {
            "expected": agent_ground_speed_expected,
            "wind_1": agent_ground_speed_1,
            "wind_2": agent_ground_speed_2,
        }
    }