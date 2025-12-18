import streamlit as st
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
sys.path.append('code/utils')
from utils.engament_zone_model import *
from utils.engament_zone_viz import *

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import io
##---Setup---##
color = st.get_option("theme.primaryColor")
os.makedirs('../data', exist_ok=True)

st.set_page_config(page_title="Engagement Zone Interactive", layout="wide", initial_sidebar_state="expanded", page_icon="🛩️")
st.sidebar.header("Navigation")
st.sidebar.info("Select **1_Engament_Zone** to define the scenario, then **2_Path_Planner** to generate an EZ-aware path.")

st.title("🛩️ Engagement Zone (EZ) + RRT* Interactive")

st.markdown(
    """
This Streamlit app supports **decision-making and path planning** for an agent operating in the presence of an intruder, using the **Engagement Zone (EZ)** concept and an **RRT\***-based planner.

### Purpose of the app
- **Strategic layer (Page 1 — Engagement Zone):** help you **choose and validate** scenario settings (agent/intruder coordinates and EZ parameters) before planning.
- **Tactical layer (Page 2 — Path Planner):** compute an **EZ-aware path** using **RRT\*** under the selected scenario conditions.

### How to use this app
**1) Page 1 — `1_Engament_Zone` (Strategic decision-making)**
- Set **agent/intruder positions** and **goal coordinates**.
- Configure EZ parameters (**μ**, **R_ez**, **r**) and inspect the resulting geometry.
- Assess **safety/feasibility** of candidate configurations before running the planner.

**2) Page 2 — `2_Path_Planner` (Tactical path planning)**
- Run the **RRT\*** planner using the selected agent/intruder/goal/EZ configuration.
- Visualize the planned path and review run outputs.

### Development notes
Page 2 is still under development. Upcoming additions include:
- **Wind conditions** (environmental uncertainty / disturbance modeling)
- **Offline vs. Online modes** (hybrid operation where the plan can be computed offline and adapted online)

---
Use the **sidebar** to navigate between pages.
    """
)


