import numpy as np


def wrap_angle_deg(angle: float) -> float:
    """Wrap an angle in degrees to the [-180, 180] range, including -180 and 180 only once."""
    wrapped = ((angle + 180.0) % 360.0) - 180.0
    # If input is exactly 180 or -180, preserve sign
    if np.isclose(np.abs(angle) % 360, 180.0):
        return np.sign(angle) * 180.0
    return wrapped

def wind_effect(wind_direction=0.0):
    """Classify wind relative to +x (east) for labeling purposes (radians).

    0 → east (tailwind for easdimetbound pursuer), π → west (headwind), others crosswind.
    """
    two_pi = 2 * np.pi
    wd = float(wind_direction) % two_pi
    if np.isclose(wd, 0.0) or np.isclose(wd, two_pi):
        return 'tailwind'
    if np.isclose(wd, np.pi):
        return 'headwind'
    return 'crosswind'

def ground_speed(entity_parameters, wind_speed, wind_direction=0.0):
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
    
    wind_type = wind_effect(wind_direction)

    # # Wind Direction: Fixed eastward (deterministic)
    # wind_speed_x, wind_speed_y = wind_speed, 0.0

    # Ground velocity components (east-west, north-south)
    v_ground_x = speed * np.cos(heading) + wind_speed * np.cos(wind_direction)
    v_ground_y = speed * np.sin(heading) + wind_speed * np.sin(wind_direction)

    return np.hypot(v_ground_x, v_ground_y), wind_type

def compute_ground_speed(air_speed, wind_speed, heading_deg, wind_direction_deg):
    """
    Convenience wrapper that accepts heading and wind direction in degrees.

    Parameters
    ----------
    air_speed : float
        Air-referenced speed of the platform (m/s).
    wind_speed : float
        Magnitude of the wind (m/s).
    heading_deg : float
        Heading of the platform in degrees (0° = East, positive counter-clockwise).
    wind_direction_deg : float
        Direction the wind is blowing toward in degrees.

    Returns
    -------
    float
        Ground speed magnitude after accounting for wind.
    """
    entity_parameters = {
        'speed': float(air_speed),
        'heading': np.deg2rad(heading_deg),
    }
    ground_speed_value, _ = ground_speed(entity_parameters, wind_speed, np.deg2rad(wind_direction_deg))
    return ground_speed_value

def ground_speed_uncertainty(entity_parameters,  wind_speed_1 = 10, wind_speed_2 = 20 , prob_w1 = 0.6, wind_direction=0.0):
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

    v_ground_1, wind_type_1 = ground_speed(entity_parameters, wind_speed_1, wind_direction)
    v_ground_2, wind_type_2 = ground_speed(entity_parameters, wind_speed_2, wind_direction)

    # Expected (probability-weighted) ground speed
    v_ground_expected = prob_w1 * v_ground_1 + prob_w2 * v_ground_2
    return v_ground_expected, v_ground_1, v_ground_2

def wind_model(agent_parameters, intruder_parameters, t_fuel,
               method, wind_speed_1=10.0, wind_speed_2=20.0,
               prob_w1=0.6, R_baseline=None, wind_direction=0.0):
    """
    Compute μ (speed ratio) and R (range) under wind uncertainty for four methods.
    
    Assumptions:
    - Intruder = Pursuer (P) → uses intruder_parameters
    - Agent = Agent (A) → uses agent_parameters
    - Pursuer heading = 0 rad (east)
    - Wind direction = eastward (0 degrees) or westward (180 degrees)
    - If intruder 'heading' is missing, it defaults to 0 rad via ground_speed()
    """

    if t_fuel < 0:
        raise ValueError("t_fuel must be non-negative")
    if not (0.0 <= prob_w1 <= 1.0):
        raise ValueError("prob_w1 must be in [0, 1]")
    # Allow any wind_direction (radians). Ground speed uses vector sum and classification uses wind_effect().

    prob_w2 = 1.0 - prob_w1
    eps = 1e-9

    agent_speed = float(agent_parameters.get('speed'))
    pursuer_speed = float(intruder_parameters.get('speed'))

    # Helper: compute expected ground speeds for agent (uses given wind_direction)
    agent_g_exp, agent_g1, agent_g2 = ground_speed_uncertainty(
        agent_parameters,
        wind_speed_1,
        wind_speed_2,
        prob_w1,
        wind_direction=wind_direction,
    )

    # Choose pursuer wind direction per method
    # Document conservative assumption (Eq. 24): V_P = v_intruder + w
    # -> intruder always benefits from wind, no heading, no vector decomposition.
    method_lower = str(method).lower()
    valid = ("baseline", "conservative", "optimistic", "cons_expected", "opt_expected")
    if method_lower not in valid:
        raise ValueError("method must be 'baseline', 'conservative', 'optimistic', 'cons_expected', or 'opt_expected'")

    if method_lower == "baseline":
        pursuer_g1 = pursuer_speed
        pursuer_g2 = pursuer_speed
        pursuer_g_exp = pursuer_speed

    else:
        if method_lower in ("conservative", "cons_expected"):
            # Conservative: intruder benefits (tailwind best-case)
            pursuer_g1 = pursuer_speed + float(wind_speed_1)
            pursuer_g2 = pursuer_speed + float(wind_speed_2)

        elif method_lower in ("optimistic", "opt_expected"):
            # Optimistic (for the agent): intruder is penalized by wind (headwind worst-case)
            pursuer_g1 = max(eps, pursuer_speed - float(wind_speed_1))
            pursuer_g2 = max(eps, pursuer_speed - float(wind_speed_2))

        pursuer_g_exp = prob_w1 * pursuer_g1 + prob_w2 * pursuer_g2

    # Apply methods
    if method_lower == "baseline":
        mu = agent_speed / pursuer_speed
        R = R_baseline if R_baseline is not None else pursuer_speed / pursuer_speed

    elif method_lower == "conservative":
        # Ratio of expectations: E[V_A] / E[V_P] with pursuer tailwind
        mu = agent_g_exp / pursuer_g_exp
        R = R_baseline if R_baseline is not None else (pursuer_g_exp / pursuer_speed)

    elif method_lower == "optimistic":
        # Ratio of expectations: E[V_A] / E[V_P] with pursuer headwind
        mu = agent_g_exp / pursuer_g_exp
        R = R_baseline if R_baseline is not None else (pursuer_g_exp / pursuer_speed)

    elif method_lower == "cons_expected":
        # Expectation of ratios with pursuer tailwind
        mu = prob_w1 * (agent_g1 / pursuer_g1) + prob_w2 * (agent_g2 / pursuer_g2)
        R = R_baseline if R_baseline is not None else (prob_w1 * pursuer_g1 + prob_w2 * pursuer_g2) / pursuer_speed

    elif method_lower == "opt_expected":
        # Expectation of ratios with pursuer headwind
        mu = prob_w1 * (agent_g1 / pursuer_g1) + prob_w2 * (agent_g2 / pursuer_g2)
        R = R_baseline if R_baseline is not None else (prob_w1 * pursuer_g1 + prob_w2 * pursuer_g2) / pursuer_speed

    # ---------------------------------------------------------
    # Step 3: Return explicit results
    # ---------------------------------------------------------
    return {
        "method": method,
        "mu": mu,
        "R": R,
        "pursuer_ground_speeds": {
            "expected": pursuer_g_exp if method_lower != "baseline" else pursuer_speed,
            "wind_1": pursuer_g1 if method_lower != "baseline" else pursuer_speed,
            "wind_2": pursuer_g2 if method_lower != "baseline" else pursuer_speed,
        },
        "agent_ground_speeds": {
            "expected": agent_g_exp,
            "wind_1": agent_g1,
            "wind_2": agent_g2,
        }
    }
