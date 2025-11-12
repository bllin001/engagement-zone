# RRT* Path Planner with Engagement Zone considering wind uncertainty Tasks

- [x] Add EZ to RRT* algorithm 
  - Start developing Phase 1 (offline mode) by adding the Engagement Zone (EZ) to the RRT* algorithm without the wind. Reference the PowerPoint ([Progress_Report_6_1_Fall2025](docs/Progress_Report_6_1_Fall2025%20.pdf)) for details.
- [ ] Validation and Cost (Next Week: Nov 17 to 19)
  - [ ] Step 1: Validate Edge (EZ Constraint)
    - Sample $M = 15$ points along edge (branches for the next node). For each point $j$:
      - Calculate $d_j = \text{distance to intruder}$
      - Calculate $\xi_j = \text{aspect angle}$
      - Calculate $\rho_j = \rho(\xi_j, \mu, R, r)$
      - If $d_j \leq \rho_j$: REJECT edge, return cost $= \infty$
      - Store $\text{clearance}_j = ||d_j - \rho_j||$
    - Find $D_{\text{max}} = \max(\text{clearance}_j)$ across all points.
  - [ ] Step 2: Calculate Distance Cost ($C_{\text{distance}}$) based on       
    - Calculate $L = ||\text{new\_node} - \text{parent\_node}||$
    - Calculate $C_{\text{distance}} = \frac{L}{L_{\text{max}}}$.
  - [ ] Step 3: Calculate Safety Cost
    - Calculate $C_{\text{safety}} = \frac{\text{Store clearance}_j}{D_{\text{max}}}$.
  - [ ] Step 4: Combine Costs
    - Combine costs using $C_{\text{edge}} = \lambda \times C_{\text{distance}} + (1 - \lambda) \times C_{\text{safety}}$.
- [ ] Incorporate wind into RRT* (Nov 24 to Dec 3)
  - After applying the cost function, include the wind effects in the RRT* algorithm.
- [ ] Add controller to RRT* (Dec 8 to 18)
  - Once the wind is incorporated, add the controller to the RRT* algorithm.
- [ ] Run experiments to validate method
  - After applying the cost function, incorporating wind, and adding the controller, run experiments to demonstrate the method’s effectiveness.

# Timeline for Engagement Zone (EZ) + RRT* Development

<table>
  <tr>
    <th>Task</th>
    <th>Timeline</th>
    <th>Description</th>
    <th>Status</th>
  </tr>
  <tr style="background-color: #d4edda;">
    <td>Add EZ to RRT* algorithm</td>
    <td>Nov 10 to Nov 11</td>
    <td>Completed: Added the Engagement Zone (EZ) to the RRT* algorithm.</td>
    <td>
      <select>
        <option value="complete" selected>Complete</option>
        <option value="pending">Pending</option>
        <option value="in-progress">In Progress</option>
      </select>
    </td>
  </tr>
  <tr style="background-color: #fff3cd;">
    <td>Validation and Cost</td>
    <td>Nov 17 to Nov 19</td>
    <td>Validate edge constraints, calculate costs, and combine them.</td>
    <td>
      <select>
        <option value="complete">Complete</option>
        <option value="pending" selected>Pending</option>
        <option value="in-progress">In Progress</option>
      </select>
    </td>
  </tr>
  <tr style="background-color: #f8d7da;">
    <td>Incorporate wind into RRT*</td>
    <td>Nov 24 to Dec 3</td>
    <td>Add wind effects to the RRT* algorithm.</td>
    <td>
      <select>
        <option value="complete">Complete</option>
        <option value="pending" selected>Pending</option>
        <option value="in-progress">In Progress</option>
      </select>
    </td>
  </tr>
  <tr style="background-color: #f8d7da;">
    <td>Add controller to RRT*</td>
    <td>Dec 8 to Dec 18</td>
    <td>Integrate the controller into the RRT* algorithm.</td>
    <td>
      <select>
        <option value="complete">Complete</option>
        <option value="pending" selected>Pending</option>
        <option value="in-progress">In Progress</option>
      </select>
    </td>
  </tr>
  <tr style="background-color: #f8d7da;">
    <td>Run experiments to validate</td>
    <td>Dec 19 to Dec 22</td>
    <td>Test the method and demonstrate its effectiveness.</td>
    <td>
      <select>
        <option value="complete">Complete</option>
        <option value="pending" selected>Pending</option>
        <option value="in-progress">In Progress</option>
      </select>
    </td>
  </tr>
</table>