# Basic Engagement Zones

## Introduction

The central problem is *navigation in threat-laden environments*.  

> For example, a supply airplane may need to reach its destination quickly, even if that requires flying near unfriendly areas containing ground-based defensive assets, which could fire upon the airplane if it comes too close.

A gap arises when the “obstacles” are not only dynamic but also *adversarial*.  

---

## How can this be addressed?

One way to capture such interactions is through **modeling adversarial engagements**. A well-established paradigm for this is **differential games**, and in particular, **zero-sum differential games**.

If a scenario can be modeled as two players with opposing objectives over a single scalar payoff, it can be cast as a zero-sum differential game. The solution typically consists of:

- The **value function**, which gives the equilibrium cost or reward as a function of the initial condition.  
- The **equilibrium control policies** for both players.  

In many cases, the relevant solution concept is the **Nash equilibrium**, a stable outcome in which neither player can improve their payoff by unilaterally changing strategy.

Within differential games, two classes are especially relevant:

- **Pursuit–evasion:** the Pursuer seeks to capture the Evader, usually in *minimum time* or with *minimum control effort*.  
  *Example:* an enemy missile (Pursuer) attempting to intercept an aircraft (Evader) before it escapes.  

- **Target guarding (reach–avoid):** the Defender seeks to capture the Attacker before the Attacker can reach a protected target region.  
  *Example:* a guard drone (Defender) intercepting an intruding drone (Attacker) before it reaches a restricted facility.  

---

## Method 1: Differential Games (Background)
#### Previous Work

Despite their theoretical appeal, most existing solutions are not well-suited to the **navigation problem**.  

- **Pursuit–evasion solutions:**  
  - Focus mainly on the Evader’s goal of *avoiding capture*.  
  - Hard to include additional objectives (e.g., reaching a destination or minimizing fuel) without reformulating the problem.  

- **Target guarding solutions:**  
  - Typically delineate regions where capture or breach can be guaranteed.  
  - Often neglect practical constraints on the Defender (e.g., fuel or time limits).  

**Notable exceptions:**  
- [5] Target-guarding game with a **fixed time limit** (interpretable as a fuel constraint).  
  - *Limitation:* only provides the outcome (breach or capture) and equilibrium headings at initial engagement; no guidance on strategies before or after.  

- [6] **Zero-sum stochastic differential game** with a “safe set vs. unsafe set” formulation over a fixed duration.  
  - *Limitation:* describes agent behavior once the game starts but does not guide the Evader on how to navigate **a priori** to avoid entering a losing situation.  

**In summary:** these approaches contribute important insights but fall short of informing practical navigation strategies in adversarial environments.

> **REFERENCES**  
[5] Chen, X., Yu, J., Yang, D., and Niu, K., “A geometric approach to reach-avoid games with time limits,” *IET Control Theory & Applications*, 2022. https://doi.org/10.1049/cth2.12374.  
[6] Patil, A., Zhou, Y., Fridovich-Keil, D., and Tanaka, T., “Risk-Minimizing Two-Player Zero-Sum Stochastic Differential Game via Path Integral Control,” 2023.  

---

## Method 2: Engagement Zone (EZ)–Based Navigation

#### EZ Definition

**Definition 1 (Engagement Zone).**  
: Given a Mobile Agent $A$, a Threat $T$, their respective dynamic models $\dot{x}_A$, $\dot{x}_T$, and an Agent strategy $u_A(t)$, an **Engagement Zone (EZ)** is a region of the state space in which it is possible for the Threat to neutralize the Mobile Agent if the latter does not deviate from its current strategy.

**Where:**
- $A$: the **Mobile Agent** (e.g., an aircraft trying to reach a destination).  
- $T$: the **Threat** (e.g., a missile, turret, or pursuing vehicle).  
- $\dot{x}_A$: the **state dynamics of the Agent**, i.e., the equations that describe how its state variables (e.g., position, velocity, heading) evolve over time when it follows a chosen strategy.  
- $\dot{x}_T$: the **state dynamics of the Threat**, i.e., how the Threat moves or acts over time.  
- $u_A(t)$: the **control strategy of the Agent**, assumed fixed (for example, “keep flying straight at current speed and heading”).  
- **Neutralization:** usually defined as the Threat being able to reach the Agent within its effective range $R$, and within a capture radius $r$.

---

##### Interpretation

- **Outside the EZ:**  
  The Threat cannot guarantee capture within its effective range $R$. The Agent is safe to continue on its current course without evasive action.  

- **Inside the EZ:**  
  The Threat can guarantee capture if it engages, given its dynamics and range. To survive, the Agent must change course or maneuver.  

- **Why this matters:**  
  - The EZ offers a *conservative safety margin*: staying outside ensures no last-second evasive maneuvers are needed.  
  - In contrast, **win/lose regions** in classical differential games may require aggressive action at the boundary if the Threat chooses to engage. The EZ removes this risk by providing a buffer zone.  

---

#### Previous Work

- **[7]** Introduced the **single-Agent, single-EZ navigation problem**.  
  - Developed path plans where:
    - The Agent avoided entering the EZ entirely.  
    - Some penetration into the EZ was allowed to reduce overall travel time or meet a specified arrival time.  

- **[8]** Extended the problem to navigation of a **single vehicle around (or through) two EZs**.  

- Both works used a **notional EZ model** (cardioid shape):  
  - The EZ was not tied to a specific Threat model.  
  - In practice, for a given Agent and Threat, EZs can be computed via **simulation**.  
  - This leads to EZs that may be **data-based** rather than analytic, which can make path planning more computationally expensive.  

**In summary:** these works advanced EZ-based navigation, but the reliance on notional models limited realism and increased computational burden when extending to data-driven EZs.

*These limitations motivate the contributions of the present paper.*

> **REFERENCES**  
[7] Weintraub, I. E., Von Moll, A., Carrizales, C., Hanlon, N., and Fuchs, Z., “An Optimal Engagement Zone Avoidance Scenario in 2-D,” *AIAA SciTech*, San Diego, 2022. https://doi.org/10.2514/6.2022-1587.  
[8] Dillon, P. M., Zollars, M. D., Weintraub, I. E., and Von Moll, A., “Optimal Trajectories for Aircraft Avoidance of Multiple Weapon Engagement Zones,” *Journal of Aerospace Information Systems*, Vol. 20, 2023, pp. 520–525. https://doi.org/10.2514/1.I011224.  

## Contribution of This Paper
This paper introduces two main contributions:  
1. A set of EZs derived from **first-principle models** of the Agent and Threat (not merely notional shapes).  
2. A direct comparison of **EZ-based navigation and path-planning** with the more traditional approach of **circumnavigation**.  

### Where the limitations apply

| Problem area            | Prior work focus                          | Main limitation for navigation                                                                       |
| ----------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Differential games**  | Pursuit–evasion and target-guarding games | Do not capture broader navigation objectives (e.g., reaching destination, constraints on fuel/time). |
| **EZ-based navigation** | Path planning around notional EZs         | EZs not derived from real Agent/Threat models; often only notional or simulation-based.              |

## Problem Definition
### Step 1: Basic Pursuit–Evasion Model

#### Variables
- $A$: Mobile Agent (Evader) position in $\mathbb{R}^2$ (e.g., $A=(x_A,y_A)=(2,3)$ km)
- $P$: Pursuer (Threat) position in $\mathbb{R}^2$ (e.g., $P=(x_P,y_P)=(0,0)$ km at the origin)
- $r$: Capture radius (neutralization occurs if $\|A - P\| \leq r$) (e.g., $r=0.1$ km)
- $R$: Maximum range of the Pursuer (e.g., $R=10$ km)
- $\mathcal{R}_P$: Reachability region of the Pursuer  
  - $\mathcal{R}_P = \{ (x, y) \;\mid\; (x - x_{P0})^2 + (y - y_{P0})^2 \leq R^2 \}$  
  - Represents the disk of radius $R$ centered at the Pursuer’s initial position $(x_{P0}, y_{P0})$ (e.g., a circle of radius 10 km centered at $(0,0)$).
- $v_P$: Speed of the Pursuer (e.g., $v_P = 500$ m/s)
- $v_A$: Speed of the Agent (e.g., $v_A = 400$ m/s)
- $\mu$: Speed ratio  
  - $\mu = \dfrac{v_A}{v_P}$ (e.g., $\mu=0.8$ if $v_A=400$ m/s and $v_P=500$ m/s)

#### Assumptions
- Both $A$ and $P$ move in the plane ($\mathbb{R}^2$) with **simple motion** and constant speeds.  
- The Pursuer $P$ is **range-limited** with maximum range $R$ [9].  
- $P$ can reach any point within the disk of radius $R$ centered at its initial position.  
- For this basic model, all parameters ($A$, $P$, $R$, $r$, $v_A$, $v_P$) are assumed known and deterministic.

#### Restrictions
- $r \geq 0$  
- $\mu > 0$  

#### Example

**Given:**
- Agent $A$ is at position $A = (x_A, y_A) = (2,3)$ km.
- Pursuer $P$ is at position $P = (x_P, y_P) = (0,0)$ km (the origin).
- Pursuer's maximum range: $R = 10$ km.
- Capture radius: $r = 0.1$ km.
- Agent speed: $v_A = 400$ m/s.
- Pursuer speed: $v_P = 500$ m/s.

**To Estimate:**
- The distance between $A$ and $P$.
- Whether the Agent is within the capture radius (neutralization condition).
- Whether the Agent is within the Pursuer's maximum range.
- The speed ratio $\mu = v_A / v_P$.
- Interpretation: Is the Agent at risk? What does the speed ratio imply for capture?

Suppose an aircraft (Agent $A$) is flying at position  
\[
A = (x_A, y_A) = (2,3) \; \text{km},
\]  
while a missile battery (Pursuer $P$) is located at the origin  
\[
P = (x_P, y_P) = (0,0) \; \text{km}.
\]

---

**Step 1. Reachability region of $P$.**  
The maximum range of the Pursuer is $R=10$ km. Its reachability region is the disk:  
\[
\mathcal{R}_P = \{ (x,y) \in \mathbb{R}^2 \;|\; (x-x_{P0})^2 + (y-y_{P0})^2 \leq R^2 \}.
\]  
Since $P_0=(0,0)$ and $R=10$, this becomes:  
\[
\mathcal{R}_P = \{ (x,y) \;|\; x^2 + y^2 \leq 100 \}.
\]

---

**Step 2. Capture condition.**  
Neutralization occurs if:  
\[
\|A - P\| \leq r,
\]  
where $r=0.1$ km is the capture radius.  

The Euclidean distance between $A$ and $P$ is:  
\[
\|A-P\| = \sqrt{(x_A - x_P)^2 + (y_A - y_P)^2}.
\]  
Substituting values:  
\[
\|A-P\| = \sqrt{(2-0)^2 + (3-0)^2} = \sqrt{4+9} = \sqrt{13}.
\]  
Numerically:  
\[
\|A-P\| \approx 3.606 \; \text{km}.
\]  

Since $3.606 > r=0.1$, the Agent is not yet captured.

---

**Step 3. Range check.**  
We compare the distance with the maximum range:  
\[
\|A-P\| = \sqrt{13} \approx 3.606 \; \text{km} < R=10 \; \text{km}.
\]  
Thus, the Agent lies **inside the Pursuer’s reachability region**.

---

**Step 4. Speed ratio.**  
The Pursuer’s speed is $v_P=500$ m/s and the Agent’s speed is $v_A=400$ m/s.  
The speed ratio is:  
\[
\mu = \frac{v_A}{v_P} = \frac{400}{500}.
\]  
Carrying out the division:  
\[
\mu = 0.8.
\]

---

**Interpretation.**  
- The Agent is within the missile’s reachability region ($3.606 < R=10$).  
- The Agent is not within the capture radius ($3.606 > r=0.1$), so it is safe for now.  
- Because $\mu=0.8<1$, the Pursuer is faster than the Agent. This means that if the Agent keeps its course, the Pursuer can guarantee capture by maneuvering within its range.
- In this basic example, $\mathcal{R}_P$ is not directly used in the calculations. However, it formalizes the Pursuer’s reachability region, which becomes important later when defining the Engagement Zone (EZ).

### Step 2: Pursuit–Evasion Engagement Zone (EZ)

#### Variables
- $\rho(\xi)$: Radial distance of the EZ boundary at aspect angle $\xi$.
  - Formula: $$\rho(\xi) \approx (1 + \mu \cos \xi)R + r$$
- $\xi$: Aspect angle between the Agent’s heading and the line of sight from $P$ to $A$.
  - Definition: $$\cos \xi = \frac{v_A}{\|v_A\|} \cdot \frac{A - P}{\|A - P\|}$$
  - (See domain in **Restrictions**.)
- $\rho_{\max}$: Maximum EZ boundary distance (head-on).
  - Formula: $$\rho_{\max} \approx (1+\mu)R + r$$
- $\rho_{\min}$: Minimum EZ boundary distance (tail-chase).
  - Formula: $$\rho_{\min} \approx (1-\mu)R + r$$

#### Assumptions
- The Agent $A$ maintains a fixed heading (static EZ assumption).
- Pursuer $P$ has maximum range $R$ and capture radius $r$.
- Both move with constant speeds $v_A$, $v_P$ in $\mathbb{R}^2$.

#### Restrictions
- $0 \leq \xi \leq \pi$ (aspect angle domain).
- $\mu = v_A / v_P$, with $0 < \mu < 1$ (Pursuer faster than Agent).

#### Example

**Given:**
- Agent $A$ is at position $A = (x_A, y_A) = (2,3)$ km.
- Pursuer $P$ is at the origin $P = (x_P, y_P) = (0,0)$ km.
- Pursuer's maximum range: $R = 10$ km.
- Capture radius: $r = 0.1$ km.
- Agent speed: $v_A = 400$ m/s.
- Pursuer speed: $v_P = 500$ m/s.
- Aspect angle cases: $\xi = 0$ (head-on) and $\xi = \pi$ (tail-chase).

**To Estimate:**
- The speed ratio $\mu = v_A / v_P$.
- The EZ boundary $\rho(\xi)$ as a function of the aspect angle.
- Special cases $\rho_{\max}$ and $\rho_{\min}$.
- Interpretation: Is the Agent inside or outside the EZ?

---

**Step 1. Speed ratio.**  
\[
\mu = \frac{v_A}{v_P} = \frac{400}{500} = 0.8.
\]

---

**Step 2. General EZ boundary.**  
The EZ boundary in polar form is approximated as:  
\[
\rho(\xi) \approx (1 + \mu \cos \xi)R + r.
\]

---

**Step 3. Head-on case ($\xi=0$).**  
Substitute $\cos 0 = 1$:  
\[
\rho_{\max} \approx (1 + \mu)R + r.
\]  
\[
\rho_{\max} = (1 + 0.8)(10) + 0.1 = 18.1 \;\text{km}.
\]

---

**Step 4. Tail-chase case ($\xi=\pi$).**  
Substitute $\cos \pi = -1$:  
\[
\rho_{\min} \approx (1 - \mu)R + r.
\]  
\[
\rho_{\min} = (1 - 0.8)(10) + 0.1 = 2.1 \;\text{km}.
\]

---

**Interpretation.**  
- The EZ forms a **deformed bubble** around the Pursuer.  
- **Head-on ($\xi=0$):** the EZ stretches forward to $18.1$ km, much larger than $R=10$ km.  
- **Tail-chase ($\xi=\pi$):** the EZ compresses to only $2.1$ km behind the Agent.  
- If the Agent is **outside** this boundary, it is safe to continue without maneuvering.  
- If the Agent is **inside** the EZ, the Pursuer can guarantee capture unless the Agent changes course.**

# Meeting

## 09/10/2025

### Experiment

- Definining EZ as $\rho(\xi; \mu, R, r)$ (eq. 5)

  - $\xi = {0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330, 360}$
  - $\mu$ = 0.7
  - $R$ = 1
  - $r$ = 0.25


- After define EZ:
  - Now we include $A$
  - We have to decide in which conditions is safe or not base on $A_{(x,y)}$

##


## 09/09/2025

## Comments
**Goal:**  implementing the Engagement Zone based on a paper and adding wind uncertainty using the TMML paper

- ==To be consider==
  -  For our case, instead `pursuer` will be `intruder`

- EZ is static


### Task
- [ ] **Write a 1–2 page summary of the Engagement Zone (EZ)**
      - Define variables and assumptions: intruder (formerly pursuer), agent, simple-motion model, finite range $R$, capture radius $r$, speed ratio $\mu = v_A/v_P$, aspect angle $\xi$, static EZ assumption for fixed agent heading.
      - Include intuition and bounds: $\rho_{\max} \approx (1+\mu)R + r$ (frente) and $\rho_{\min} \approx (1-\mu)R + r$ (espaldas), with a small sketch/diagram.
- [ ] **Prototype code to compute and plot the EZ** (single self-contained script)
      - Implement a function `rho(xi, R, r, mu)` and evaluate $\rho(\xi)$ for $\xi \in [-\pi, \pi]$ to trace the boundary.
      - Produce two outputs: (i) a PNG plot of the EZ, and (ii) a CSV with columns `xi, rho`.
      - Keep it simple (no external solvers/optimization).
- [ ] **Compute the EZ radius $\rho$ for selected aspect angles**
      - Evaluate $\rho(\xi)$ at $\xi \in \{0, \pi/6, \pi/4, \pi/3, \pi/2, \pi\}$ and report results in a small table (include units).
- [ ] **Create visuals based on the discussed parameters**
      - Plot the EZ boundary plus reference circles: $R+r$, $(1+\mu)R + r$, $(1-\mu)R + r$.
      - (Optional) Add a simple path-planning sketch: ruta directa que cruza EZ vs. ruta “skimming” que bordea la EZ (sin entrar).
- [ ] **Review the paper to resolve open questions**
      - Verify the mathematical expression used for $\rho(\xi)$ and the conditions for the EZ boundary (use of $R$ and $r$).
      - Confirm that the “EZ is static” assumption is tied to fijar el rumbo del agente.
      - Note open items (e.g., how to incorporate wind uncertainty per TMML) as follow-up, not required here.


# Main Concepts

## Differential Game

## Zero-sum Differential Game

A **zero-sum differential game** is a type of mathematical game theory model that combines elements of both **zero-sum games** and **differential games**:

### Key Features

- **Zero-sum**: In this context, the sum of the payoffs for all players is always zero—meaning one player's gain is exactly the other's loss.
- **Differential Game**: This involves players making decisions over time, where the system's state evolves according to differential equations. It is a continuous-time generalization of game theory.


### Typical Structure

- There are two (sometimes more) players. One aims to maximize a certain payoff (the maximizer), and the other aims to minimize it (the minimizer).
- The game's dynamics are governed by differential equations:

$$
\dot{x}(t) = f(x(t), u(t), v(t), t)
$$

Where:
    - $x(t)$: The state of the system at time $t$
    - $u(t)$: Control action of player 1
    - $v(t)$: Control action of player 2
- The **payoff** or **cost functional** is typically expressed as:

$$
J(u, v) = \int_{t_0}^{T} L(x(t), u(t), v(t), t)\, dt + \psi(x(T))
$$
    - $L$: Running cost (or reward) rate
    - $\psi$: Terminal cost (or reward) at the end time
- Each player selects their controls $u(t)$ and $v(t)$ to optimize $J(u, v)$, with one trying to maximize and the other minimize.


### Applications

- **Pursuit-evasion problems:** e.g., missile and aircraft, predator-prey models
- **Economics:** Pricing strategies, resource allocation
- **Military strategy:** Interceptor/evader scenarios


### Solution Concepts

- Solutions often involve **Hamilton-Jacobi-Isaacs (HJI) equations**, which generalize the Hamilton-Jacobi-Bellman equation for differential games.
- The main solution sought is the **saddle point** or **value** of the game, which is analogous to the Nash equilibrium in zero-sum settings.

**In short:**
A zero-sum differential game models two competing agents dynamically interacting over time, where one's gain is exactly the other's loss, and the evolution of the game is governed by differential equations.

## The Nash Equilibrium

The **Nash equilibrium** is a fundamental concept in game theory, named after mathematician John Nash. It describes a situation in a strategic game where no player can benefit by changing their strategy while the other players keep theirs unchanged.

**Key Points:**

- At Nash equilibrium, every player's strategy is optimal given the strategies of all other players.
- If every player is playing their part of the equilibrium, no single player has an incentive to deviate.

**Example:**
Imagine two companies, A and B, deciding whether to advertise or not:

- If both advertise, they split the market (and profits decrease due to increased costs).
- If neither advertises, they also split the market, but profits are higher because of lower costs.
- If one advertises and the other doesn't, the advertiser gains more customers.

The Nash equilibrium in this case would be the set of strategies where neither company would benefit by solely changing their choice, given the choice of the other company.

**Formal Definition:**
If the strategies of all players are $(s_1^*, s_2^*, ..., s_n^*)$, then $(s_1^*, s_2^*, ..., s_n^*)$ is a Nash equilibrium if, for every player $i$:

$$
u_i(s_1^*, ..., s_i^*, ..., s_n^*) \geq u_i(s_1^*, ..., s_i, ..., s_n^*)
$$

for all alternative strategies $s_i$, where $u_i$ is the payoff function for player $i$.

**In summary:**

- **No player** can improve their payoff by unilaterally changing their strategy at a Nash equilibrium.
- There may be multiple Nash equilibria, or none, depending on the game.

## Colision Course

</file>
