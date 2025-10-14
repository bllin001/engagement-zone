import numpy as np
import os
import pandas as pd

def wind_model():
    
    pass

def wrap_angle_deg(angle: float) -> float:
    """Wrap an angle in degrees to the [-180, 180] range, including -180 and 180 only once."""
    wrapped = ((angle + 180.0) % 360.0) - 180.0
    # If input is exactly 180 or -180, preserve sign
    if np.isclose(np.abs(angle) % 360, 180.0):
        return np.sign(angle) * 180.0
    return wrapped

def intruder(x, y):
    return x, y

def agent(x, y, heading, speed=None):
    """
    Create agent parameters.
    heading: in degrees (converted to radians internally)
    speed: optional, not used in current calculations
    """
    agent_heading = np.deg2rad(heading)
    return x, y, agent_heading

def rho_values(mu, R, r, save_csv=None, save_txt=None):
    """
    Calculate engagement zone boundary points for all aspect angles.
    
    Aspect angle (ξ) convention: Clockwise angle from LOS direction to agent heading.
    Range: -180° to +180° (ξ ∈ [-π, π] radians)
    """
    # rho values is a list of aspect angles where is between -180 and 180 degrees
    intruder_xi_values = np.linspace(-np.pi, np.pi, 500)  # radians

    cos_xi = np.cos(intruder_xi_values)
    cos_squared = cos_xi**2
    under_sqrt = cos_squared - 1 + ((R + r)**2) / (mu**2 * R**2)
    sqrt_term = np.sqrt(under_sqrt)
    ez_points = mu * R * (cos_xi + sqrt_term)

    # Build DataFrame
    xi_deg = np.rad2deg(intruder_xi_values)
    xi_deg_wrapped = np.array([wrap_angle_deg(deg) for deg in xi_deg])
    x_boundary = ez_points * np.cos(intruder_xi_values)
    y_boundary = ez_points * np.sin(intruder_xi_values)
    df = pd.DataFrame({
        'ξ_rad': intruder_xi_values,
        'ξ_deg': xi_deg_wrapped,
        'ρ': ez_points,
        'x_boundary': x_boundary,
        'y_boundary': y_boundary,
        'cos(ξ)': np.round(cos_xi, 4),
        'cos²(ξ)': np.round(cos_squared, 4),
        'under_sqrt': np.round(under_sqrt, 4),
        'sqrt_term': np.round(sqrt_term, 4)
    })

    if save_csv is not None:
        df.to_csv(save_csv, index=False)
    if save_txt is not None:
        with open(save_txt, 'w') as f:
            f.write(df.to_string(index=False))

    return intruder_xi_values, ez_points, df

def agent_safety(intruder_parameters, agent_parameters, mu=0.7, R=1.0, r=0.25):
    """
    Evaluate agent safety relative to engagement zone.
    
    Aspect angle convention: Clockwise angle from LOS direction to agent heading.
    - Positive aspect angle: Agent heading is clockwise from LOS
    - Negative aspect angle: Agent heading is counter-clockwise from LOS (heading "above" LOS)
    """
    
# Step 1: Centralized geometric calculations and summary output
    agent_x, agent_y, agent_heading = agent_parameters
    intruder_x, intruder_y = intruder_parameters

    # Compute agent/intruder positions
    intruder_pos = np.array([intruder_x, intruder_y])
    agent_pos = np.array([agent_x, agent_y])

    # Agent heading in radians; convert to degrees for calculations
    heading_deg = wrap_angle_deg(np.degrees(agent_heading))


    # LOS from intruder to agent (geometric, signed)
    theta_LOS_intruder = wrap_angle_deg(np.degrees(np.arctan2(agent_y - intruder_y, agent_x - intruder_x)))
    
    # LOS from agent to intruder (geometric, signed)
    theta_LOS_agent = wrap_angle_deg(np.degrees(np.arctan2(intruder_y - agent_y, intruder_x - agent_x)))
    
    theta_LOS = theta_LOS_intruder
    los_rad = np.deg2rad(theta_LOS)


    # Distance from agent to intruder
    distance = np.linalg.norm(agent_pos - intruder_pos)

    # Get EZ boundary for LOS angle
    xi_values, ez_points, _ = rho_values(mu, R, r)

    # Aspect angle: clockwise angle from LOS direction to agent heading
    # Positive when heading is clockwise from LOS, negative when counter-clockwise
    aspect_deg = wrap_angle_deg(heading_deg - theta_LOS)
    aspect_rad = np.deg2rad(aspect_deg)
    # Estimate rho and boundary using aspect angle
    idx = (np.abs(xi_values - aspect_rad)).argmin()
    ez_radius = ez_points[idx]
    boundary_x = ez_radius * np.cos(aspect_rad)
    boundary_y = ez_radius * np.sin(aspect_rad)


    # Collect all relevant outputs
    angles = {
        'los': float(distance),
        'rho': float(ez_radius),
        'los_deg': float(theta_LOS),
        'los_intruder': float(theta_LOS_intruder),
        'heading_deg': float(heading_deg),
        'aspect_deg': float(aspect_deg),
        'agent_x': float(agent_x),
        'agent_y': float(agent_y),
        'intruder_x': float(intruder_x),
        'intruder_y': float(intruder_y),
        'boundary_x': float(boundary_x),
        'boundary_y': float(boundary_y),
        'mu': float(mu),
        'R': float(R),
        'r': float(r),
    }
    summary_df = pd.DataFrame(angles.items(), columns=['Parameter', 'Value'])

    # Safety check
    if distance <= ez_radius:
        return False, "Agent is inside the engagement zone (unsafe).", summary_df
    else:
        return True, "Agent is outside the engagement zone (safe).", summary_df