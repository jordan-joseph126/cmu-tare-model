# Complete Methods Walkthrough: Heat Pump Adoption Analysis
## Plain Language Guide to the Three-Model Pipeline (CORRECTED VERSION)

---

## Important Discrepancy Note

**Feeder Count Clarification**: The original Gautam et al. paper states "57 representative feeders" throughout its methodology. However, the foundational PNNL Distribution System Taxonomy (PNNL-18035, Schneider et al. 2008) establishes **24 prototypical feeder models**. This walkthrough has been corrected to reflect the accurate PNNL taxonomy characteristics while noting where the paper's claimed numbers differ. The paper may be using an expanded or modified version of the taxonomy, or there may be an error in the reported feeder count.

---

## Table of Contents

1. [Big Picture: What This Study Does](#big-picture)

2. [The Three Models and How They Connect](#three-models)

3. [Stage 1: Building Energy Simulation (ResStock + EnergyPlus)](#stage-1)

4. [Stage 2: Economic Adoption Analysis (TARE Model)](#stage-2)

5. [Stage 3: Grid Operational Analysis (Power Flow)](#stage-3)

6. [Stage 4: Putting It All Together](#stage-4)

7. [Key Data Sources](#data-sources)

8. [Complete Example: One Home's Journey](#complete-example)

---

## Big Picture: What This Study Does {#big-picture}

**The Central Question**: 

> How many U.S. homes could realistically adopt heat pumps when we consider BOTH whether it makes economic sense AND whether the electrical grid can handle it?

**Why This Matters**:

- Previous studies usually looked at economics OR grid constraints separately

- This creates an incomplete picture

- A home might be economically perfect for a heat pump, but if the local grid can't support it, adoption won't happen

**The Innovation of the study is that it integrates**:

1. **Physics-based building simulation** (what heat pumps do to energy use)

2. **Household-level economics** (who will adopt based on costs)

3. **Three-phase power flow analysis** (what the grid can actually handle)

**The Mathematical Pipeline**:

$\bar{A}_R = \mathcal{F}(h(g(f(\mathcal{I}_R)))) \quad \text{(1)}$

**Translation**: Starting with a set of representative homes ($\mathcal{I}_R$), we:

- $f(\cdot)$: Simulate their energy use with heat pumps

- $g(\cdot)$: Determine who would adopt based on economics

- $h(\cdot)$: Check if the grid can support that adoption

- $\mathcal{F}(\cdot)$: Aggregate results by region

- $\bar{A}_R$: Get realistic adoption estimates for region $R$

---

## The Three Models and How They Connect {#three-models}

### Model 1: ResStock + EnergyPlus
**What it does**: Simulates individual homes hour-by-hour for a full year

**Key inputs**:

- Housing characteristics (size, insulation, location, existing equipment)

- Weather data (outdoor temperature, humidity, solar radiation)

- Heat pump specifications (efficiency, capacity, performance curves)

**Key outputs**:

- $E_{ih}^{\text{baseline}}$: Electricity use for home $i$ at hour $h$ WITHOUT heat pump

- $E_{ih}^{\text{retrofit}}$: Electricity use for home $i$ at hour $h$ WITH heat pump

- $F_{ih}^{\text{baseline}}$: Natural gas use without heat pump

- $F_{ih}^{\text{retrofit}}$: Natural gas use with heat pump (usually zero)

**Sample size**: 8,000 homes representing 120 million U.S. dwelling units

---

### Model 2: TARE (Tradeoff Analysis for Residential Energy)
**What it does**: Calculates whether adopting a heat pump makes economic sense for each home

**Key inputs** (from ResStock):

- $E_{ih}^{\text{baseline}}$ and $E_{ih}^{\text{retrofit}}$ (electricity profiles)

- $F_{ih}^{\text{baseline}}$ and $F_{ih}^{\text{retrofit}}$ (gas profiles)

**Additional inputs**:

- Electricity prices by region ($/kWh)

- Natural gas prices by region ($/therm)

- Heat pump purchase/installation cost ($)

- Replacement equipment costs (furnace, AC)

- IRA subsidies (income-dependent)

- Discount rate (7%)

- Equipment lifetime (15 years)

**Key outputs**:

- $A_i$: Binary adoption decision (1 = adopt, 0 = don't adopt)

- $\hat{E}_{ih}$: Net change in electricity use if home $i$ adopts

- $\text{NPV}_i$: Net present value of adoption

- **Tier classification**: Which economic tier enables adoption

**Three economic tiers**:

1. **Tier 1**: Heat pump saves money compared to keeping existing equipment

2. **Tier 2**: Heat pump is worth the extra cost when replacing broken equipment

3. **Tier 3**: Heat pump becomes viable with IRA subsidies

---

### Model 3: Grid Operational Analysis (Three-Phase Power Flow)
**What it does**: Tests whether distribution grids can physically deliver power when homes adopt heat pumps

**Key inputs** (from TARE):

- $A_i$ (which homes want to adopt)

- $\hat{E}_{ih}$ (load changes from adoption)

**Additional inputs**:

- **PNNL taxonomy feeder networks** (24 prototypical feeders)
  
  - **Note**: Paper claims 57 feeders; original PNNL taxonomy has 24

  - Covers non-urban core radial distribution feeders only (excludes large urban networked systems)

- Transformer capacities

- Line impedances

- Voltage standards (±5% of nominal for Range A service)


**Key outputs**:

- $G_N$: Binary grid feasibility (1 = feasible, 0 = infeasible)

- $\hat{S}_n$: Infeasibility current sources at each node (measures constraint severity)

- $\hat{\mathcal{N}}$: Set of nodes with power imbalance

- **Voltage violations**: Nodes outside ±5% range

- **Transformer stress**: Equipment exceeding rated capacity

**Feasibility criteria**:

1. Solver converges (mathematical solution exists)

2. No power imbalance (only substation provides power)

3. All voltages within ±5% of nominal (ANSI C84.1 Range A)

4. No thermal limit violations (transformers, lines within rated capacity)

---

### Stage 4: Joint Feasibility
**What it does**: Combines economic and grid constraints

**The logic**:

$\tilde{A}_{Ni} = A_i \times G_N$ (2)

A home can adopt if and only if:
- $A_i = 1$ (economics work) AND

- $G_N = 1$ (grid can support it)

**Regional aggregation**:

$\bar{A}_R = \frac{1}{|R|} \sum_{N \in R} \frac{1}{|I_N|} \sum_{i \in I_N} \tilde{A}_{Ni}$ (3)

Average adoption rate across all homes in region $R$

**Final output**: 

- National/regional adoption estimates

- Market size (number of homes)

- Grid upgrade needs

---

## Stage 1: Building Energy Simulation (ResStock + EnergyPlus) {#stage-1}

### What ResStock Does

**ResStock** is a database of statistically representative homes. Think of it like this:

- The U.S. has ~120 million homes

- They vary by size, age, location, insulation, equipment

- ResStock creates a sample that mirrors this diversity

- This study uses 8,000 representative homes

**Home characteristics sampled**:

- **Geography**: Climate zone (1-8), urban/suburban/rural

- **Building**: Floor area, vintage (year built), foundation type

- **Envelope**: Insulation levels, window types, air tightness

- **Equipment**: Existing HVAC system (furnace, AC, heat pump)

- **Occupancy**: Number of people, thermostat settings

**Example home from sample**:

```
Home ID: 4,527
Location: Chicago suburbs (Climate Zone 5A, suburban)
Size: 1,800 sq ft, built 1985
Envelope: Moderate insulation, double-pane windows
Existing HVAC: Natural gas furnace + central AC
Occupancy: 3 people
Heating setpoint: 70°F
Cooling setpoint: 72°F
```


---

### What EnergyPlus Does

**EnergyPlus** is a physics-based building simulator. It solves thermodynamic equations hour-by-hour for an entire year.

**Key physics modeled**:

**1. Appliance Energy Use**

$E_{ih}^{\text{appliance}} = \sum_{a \in A} K_a \cdot \delta_{ah}$ (4)

**Plain language**: 

- Each appliance (fridge, oven, dryer) has a rated power $K_a$

- It runs according to a schedule $\delta_{ah}$ (0 = off, 1 = on)

- Energy use = power × time

**Example**:

```
Hour 8 AM:
- Coffee maker (1,200 W) ON for 15 min = 300 Wh

- Refrigerator (150 W) always ON = 150 Wh

- Oven (0 W) OFF = 0 Wh
Total: 450 Wh
```

---

**2. HVAC Control Logic**

$T_{ih}^{\text{target}} = \begin{cases}
T^{\text{heat}} & \text{if } T_{ih}^{\text{in}} < T^{\text{heat}} \\
T^{\text{cool}} & \text{if } T_{ih}^{\text{in}} > T^{\text{cool}} \\
T_{ih}^{\text{in}} & \text{otherwise}
\end{cases}$ (5)

**Plain language**:

- If indoor temp drops below heating setpoint → turn on heat

- If indoor temp rises above cooling setpoint → turn on AC

- Otherwise → equipment coasts (fan only or off)

**Example**:

```
Hour 6 AM, February:
- Indoor temp: 68°F

- Heating setpoint: 70°F

- Status: 68 < 70, so HEAT MODE

- Target: 70°F
```


---

**3. HVAC Energy Use**

$E_{ih}^{\text{HVAC}} = \frac{|\hat{K}_{ih}|}{\text{COP}(T_{ih}^{\text{in}}, T_{ih}^{\text{out}}, T_{ih}^{\text{target}})}$ (6)

**Plain language**:

- $\hat{K}_{ih}$ = heating or cooling load needed (W)

- $\text{COP}$ = Coefficient of Performance (efficiency)

- Energy use = load needed ÷ efficiency

**Understanding COP**:

- COP is like "miles per gallon" for heat pumps

- COP = 3.0 means: deliver 3 units of heating using 1 unit of electricity

- COP varies with temperature:
  - Higher COP when outdoor temp is mild (less work)
  - Lower COP when outdoor temp is extreme (more work)

**Example - Heating on cold day**:

```
Hour 6 AM, February:
- Outdoor temp: 20°F

- Indoor temp: 68°F

- Target temp: 70°F

- Heating load needed: 15,000 W

- COP at 20°F outdoor: 1.8

- Electricity use: 15,000 / 1.8 = 8,333 W

For comparison, same heating load on milder day:
- Outdoor temp: 40°F

- COP at 40°F: 3.2

- Electricity use: 15,000 / 3.2 = 4,688 W
(Much more efficient when it's warmer!)
```


**Why COP matters**:

- At very cold temperatures, COP drops

- Heat pump may need backup resistive heating

- Resistive heating has COP = 1.0 (very inefficient)

- This is why cold climates stress grids more

---

**4. Building Thermal Dynamics**

$T_{i(h+1)}^{\text{in}} = T_{ih}^{\text{in}} + \frac{1}{M_i}\left[U_i W_i (T_{ih}^{\text{in}} - T_{ih}^{\text{out}}) + \hat{K}_{ih}\right]$ (7)

**Plain language**:

- Buildings don't heat/cool instantly

- Temperature change depends on:
  - **Heat loss through walls**: $U_i W_i (T^{\text{in}} - T^{\text{out}})$
    - $U_i$ = How well insulated (thermal transmittance)
    - $W_i$ = Wall/window area
    - $(T^{\text{in}} - T^{\text{out}})$ = Temperature difference
  - **HVAC heating/cooling**: $\hat{K}_{ih}$
  - **Thermal mass**: $M_i$ (building's ability to store heat)

**Example - Heat loss calculation**:

```
House characteristics:
- Uᵢ = 0.5 W/(m²·°C) (moderate insulation)

- Wᵢ = 200 m² (wall/window area)

- Mᵢ = 5,000 Wh/°C (thermal capacitance)

Hour 6 AM, February:
- T^in = 68°F (20°C)

- T^out = 20°F (-7°C)

- Temperature difference: 27°C

Heat loss through envelope:
Q_loss = 0.5 × 200 × 27 = 2,700 W

Heat provided by HVAC:
K̂ᵢₕ = 15,000 W (from heat pump)

Net heat gain:
15,000 - 2,700 = 12,300 W

Temperature rise in next hour:
ΔT = 12,300 / 5,000 = 2.46°C (4.4°F)

New temp at 7 AM: 68 + 4.4 = 72.4°F
(Overshot the 70°F target, so heat pump will cycle off)
```


---

### The Heat Pump Modeled in This Study

**Specific equipment specifications**:

- **Cooling capacity**: 3.12 tons (37,440 BTU/hr)

- **SEER**: 14.12 (cooling efficiency)

- **Heating capacity**: 2.91 tons (35,040 BTU/hr)

- **HSPF**: 8.34 (heating efficiency)

- **Power factor**: 0.95 lagging (for grid analysis)
  - Note: Typical range is 0.85-0.98; 0.95 is standard assumption

- **Cold-climate capable**: No lockout temperature

**What "no lockout temperature" means**:

- Older heat pumps stop working below ~35°F

- They switch to resistive heating (very inefficient)

- Modern cold-climate heat pumps operate at any temperature

- This dramatically improves economics in cold climates

**Comparison**:

```
Old heat pump at 20°F outdoor:
- Compressor locked out (can't run)

- Switches to 100% resistive heating

- COP = 1.0 (terrible)

- Electricity use: 15,000 W / 1.0 = 15,000 W

New heat pump at 20°F outdoor:
- Compressor still running

- Some resistive backup may engage

- COP = 1.8 (much better)

- Electricity use: 15,000 W / 1.8 = 8,333 W
(45% less electricity!)
```

---

### ResStock Simulation Process

**Step 1**: Select 8,000 representative homes

- Stratified sampling by climate zone and urbanization

- Weighted to represent 120 million U.S. homes

**Step 2**: Run TWO scenarios for each home

1. **Baseline scenario**: Home as-is with existing equipment

2. **Retrofit scenario**: Home with new heat pump installed

**Step 3**: Simulate hour-by-hour for full year (8,760 hours)

For each hour:

- Get outdoor weather (temp, humidity, solar)
- Calculate appliance loads
- Calculate heating/cooling needs
- Calculate HVAC energy use
- Update indoor temperature
- Record electricity and gas consumption

**Step 4**: Generate annual profiles

- 8,760 hourly values for each home

- Electricity use (kWh)

- Natural gas use (therms)

- Breakdown by end-use (heating, cooling, appliances)

---

### ResStock Outputs ($f(\cdot)$ in the pipeline)

**For each of 8,000 homes**:

**Baseline profile** (existing equipment):

$E_{ih}^{\text{baseline}}, \quad F_{ih}^{\text{baseline}}$ (8)

**Retrofit profile** (with heat pump):

$E_{ih}^{\text{retrofit}}, \quad F_{ih}^{\text{retrofit}}$ (9)

**Delta profile** (the change):

$\Delta E_{ih} = E_{ih}^{\text{retrofit}} - E_{ih}^{\text{baseline}}$ (10)

**Example home - annual summary**:

```
Home 4,527 (Chicago suburbs):

BASELINE (gas furnace + AC):
- Annual electricity: 12,500 kWh

- Annual gas: 850 therms

- Peak winter electricity: 8 kW (January morning)

- Peak summer electricity: 6 kW (July afternoon)

RETROFIT (heat pump):
- Annual electricity: 18,200 kWh

- Annual gas: 0 therms

- Peak winter electricity: 15 kW (January morning)

- Peak summer electricity: 6 kW (July afternoon)

DELTA:
- Electricity increase: +5,700 kWh/year (+46%)

- Gas decrease: -850 therms/year (-100%)

- Peak load increase: +7 kW in winter (almost doubled!)

- Summer peak: No change
```


**Why this matters for the grid**:

- Winter peak load almost doubled (8 → 15 kW)

- If 20 homes on same transformer all adopt: 
  - Old peak: 20 × 8 = 160 kW
  - New peak: 20 × 15 = 300 kW
  - Transformer may be rated for only 200 kW → PROBLEM

---

### Key Insights from ResStock Stage

**1. Geographic variation is huge**:

```
Miami (Climate Zone 1A):
- Mild winters, hot summers

- Heat pump mostly for cooling

- Peak load: Summer afternoon

- Delta electricity: +20% (cooling more efficient)

Minneapolis (Climate Zone 7B):
- Harsh winters, mild summers  

- Heat pump mostly for heating

- Peak load: Winter morning

- Delta electricity: +120% (heating electrified)
```


**2. Heat pump performance varies dramatically**:

- Warm climates: COP stays high year-round (3.0-4.0)

- Cold climates: COP drops in winter (1.5-2.5)

- Economics and grid impacts both depend on this

**3. The "lockout temperature" matters**:

- Old heat pumps: Switch to resistive heating at 35°F

- New heat pumps: Keep running at -10°F

- This study models new technology (more realistic for future)

**4. Peak timing matters**:

- Baseline: Summer afternoon peaks (AC)

- Retrofit: Winter morning peaks (heating)

- Grid designed for summer peaks → winter peaks create new problems

---

## Stage 2: Economic Adoption Analysis (TARE Model) {#stage-2}

### What TARE Does

**TARE** = Tradeoff Analysis for Residential Energy

**Core principle**: Homeowners adopt heat pumps if they save money over equipment lifetime

**Key assumption**: Rational economic decision-making

- Homeowners calculate Net Present Value (NPV)

- If NPV > 0, they adopt

- If NPV < 0, they don't adopt

**Time horizon**: 15 years (typical equipment lifetime)

**Discount rate**: 7% (homeowner time preference for money)

---

### The Three Economic Tiers

TARE defines three progressively inclusive adoption scenarios:

#### Tier 1: "It Just Makes Sense" (Most Restrictive)

**Decision rule**:

$\text{NPV}_i = \sum_{t=1}^{15} \frac{B_{yi} - R_{yi}}{(1.07)^t} - C_{\text{equipment}} > 0$ (11)

**Plain language**:

- Compare: Operating existing equipment vs. operating heat pump

- If heat pump saves enough money to pay for itself → adopt

- No subsidies considered

- Accounts for eventual replacement of existing equipment

**When this applies**:

- Existing equipment is expensive to run (old, inefficient, or uses expensive fuel)

- Electricity is cheap relative to gas

- Climate is moderate (heat pump very efficient)

**Example - Home 2,341 (Houston)**:

```
COSTS WITH EXISTING EQUIPMENT (gas furnace + AC):
Year 1: Electricity ($1,450) + Gas ($420) = $1,870
Year 2-15: Similar, with 2% annual fuel price escalation
Also: Will need to replace furnace in Year 8 ($4,500)
Present value of 15 years: $28,400

COSTS WITH HEAT PUMP:
Upfront: $12,000 (purchase + installation)
Year 1: Electricity ($1,950) + Gas ($0) = $1,950
Year 2-15: Similar, with 2% escalation
Present value of 15 years: $12,000 + $20,100 = $32,100

NPV = $28,400 - $32,100 = -$3,700
Conclusion: DON'T ADOPT (costs more than existing system)
This home is NOT in Tier 1
```


---

#### Tier 2: "When My Furnace Dies" (Moderately Restrictive)

**Decision rule**:

$\text{NPV}_i^{\text{alternative}} = \sum_{t=1}^{15} \frac{B_{yi} - R_{yi}}{(1.07)^t} - (C_{\text{equipment}} - C_{\text{replacement}}) > 0$ (12)

**Plain language**:

- Your existing furnace/AC just broke

- You need new equipment regardless

- Is the incremental cost of a heat pump worth it?

- Only compares difference in upfront cost

**Key difference from Tier 1**:

- Tier 1: Compare heat pump to keeping existing equipment

- Tier 2: Compare heat pump to buying new conventional equipment

**When this applies**:

- Equipment failure forces replacement decision

- Captures "replacement window" economics

- Very common in practice (most HVAC purchases are emergency replacements)

**Example - Same Home 2,341 (Houston), but furnace just died**:

```
COSTS WITH NEW CONVENTIONAL EQUIPMENT:
Upfront: New furnace ($4,500) + keep existing AC
Year 1-15: Electricity ($1,450) + Gas ($420) = $1,870/year
Present value: $4,500 + $19,500 = $24,000

COSTS WITH HEAT PUMP:
Upfront: $12,000
Year 1-15: Electricity ($1,950) = $1,950/year  
Present value: $12,000 + $20,100 = $32,100

NPV^alternative = $24,000 - $32,100 = -$8,100
Conclusion: Still don't adopt (heat pump costs $8,100 more even with replacement scenario)

BUT: If gas prices were higher or electricity cheaper, this could flip to positive
```

---

#### Tier 3: "With Government Help" (Least Restrictive)

**Decision rule**:

$\text{NPV}_i^{\text{subsidized}} = \sum_{t=1}^{15} \frac{B_{yi} - R_{yi}}{(1.07)^t} - (C_{\text{equipment}} - C_{\text{replacement}} - \text{rebate}_i) > 0$ (13)

**Plain language**:

- Same as Tier 2, but includes IRA subsidies

- Subsidies reduce upfront cost

- Makes adoption attractive to more households

**IRA (Inflation Reduction Act) subsidies**:

```
Income-based rebates:
- Low income (≤80% AMI): Up to $8,000 rebate + $2,000 tax credit = $10,000 total

- Moderate income (80-150% AMI): Up to $2,000 tax credit

- Higher income (>150% AMI): May qualify for partial tax credit

Where AMI = Area Median Income (varies by location)
```

---

### The NPV Calculation in Detail

**Annual fuel costs**:

$B_{yi} = \beta_y \left[\lambda_y^{\text{electricity}} \sum_h E_{ih}^{\text{baseline}} + \lambda_y^{\text{gas}} \sum_h F_{ih}^{\text{baseline}}\right]$ (14)

$R_{yi} = \beta_y \left[\lambda_y^{\text{electricity}} \sum_h E_{ih}^{\text{retrofit}} + \lambda_y^{\text{gas}} \sum_h F_{ih}^{\text{retrofit}}\right]$ (15)

**Plain language**:

- $B_{yi}$ = Baseline fuel costs in year $y$

- $R_{yi}$ = Retrofit fuel costs in year $y$

- $\lambda^{\text{electricity}}$ = Electricity price ($/kWh)

- $\lambda^{\text{gas}}$ = Gas price ($/therm)

- $\beta_y$ = Climate adjustment factor for year $y$ (accounts for climate change)

**Example calculation - Home 4,527 (Chicago)**:

```
From ResStock outputs:
- Baseline electricity: 12,500 kWh/year

- Baseline gas: 850 therms/year

- Retrofit electricity: 18,200 kWh/year

- Retrofit gas: 0 therms/year

Chicago energy prices (2025):
- Electricity: $0.14/kWh

- Gas: $1.20/therm

Year 1 costs:
By1 = 1.0 × [$0.14 × 12,500 + $1.20 × 850]
    = 1.0 × [$1,750 + $1,020]
    = $2,770

Ry1 = 1.0 × [$0.14 × 18,200 + $1.20 × 0]
    = 1.0 × [$2,548 + $0]
    = $2,548

Savings in Year 1: $2,770 - $2,548 = $222
```

**Present value calculation**:

$\text{NPV}_i = \sum_{t=1}^{15} \frac{B_{yi} - R_{yi}}{(1.07)^t} - C_{\text{equipment}}$ (16)

**Plain language**:

- Sum up savings over 15 years

- Discount future savings (money now worth more than money later)

- Subtract upfront equipment cost

- If positive → adopt

**Example continued - Home 4,527**:

```
Year 1: ($222) / 1.07^1 = $207 present value
Year 2: ($235) / 1.07^2 = $205 present value (prices rose 2%)
Year 3: ($249) / 1.07^3 = $203 present value
...
Year 15: ($302) / 1.07^15 = $109 present value

Sum of all years: $2,850 total present value of savings

Heat pump cost: $12,000

NPV = $2,850 - $12,000 = -$9,150
Conclusion: DON'T ADOPT (loses $9,150 in present value)
```

**Why negative?**

- Heat pump uses MORE electricity (electrifying heating)

- Saves on gas, but gas is cheap in Chicago

- Electricity savings from more efficient cooling don't offset heating costs

- This is why Midwest has lower economic adoption

---

### Regional Cost Variations

**Key factors driving regional differences**:

**1. Electricity prices**:

```
Hawaii: $0.33/kWh (expensive)
Louisiana: $0.09/kWh (cheap)  
National average: $0.14/kWh
California: $0.25/kWh (expensive)
```

**2. Natural gas prices**:

```
New England: $2.50/therm (expensive)
Gulf Coast: $1.00/therm (cheap)
National average: $1.50/therm
Northeast: $2.20/therm (expensive)
```

**3. Climate zone multiplier** ($\beta_y$):

```
Climate Zone 1 (Miami): 
- Cooling-dominated

- Heat pump saves money on cooling

- Little heating needed

- βy ≈ 0.98 (small adjustment for warming)

Climate Zone 7 (Minneapolis):
- Heating-dominated  

- Heat pump electrifies major heating load

- Economics depend heavily on gas vs. electricity price

- βy ≈ 0.95 (warming reduces heating needs slightly)
```

**4. Equipment installation costs**:

```
Varies by market:
- Urban: Higher labor costs

- Rural: Higher equipment transport costs

- Cold climate: May need larger capacity → higher cost

- Ductwork modifications: +$2,000-5,000 if needed
```

---

### TARE Model Decision Logic

**Binary adoption variable**:

$A_i = \begin{cases}
1 & \text{if } \text{NPV}_i > 0 \text{ OR } \text{NPV}_i^{\text{alternative}} > 0 \text{ OR } \text{NPV}_i^{\text{subsidized}} > 0 \\
0 & \text{otherwise}
\end{cases}$ (17)

**Plain language**:

- Check Tier 1: Does heat pump beat existing equipment?

- If no, check Tier 2: Does it beat replacement alternative?

- If no, check Tier 3: Does it work with subsidies?

- If any tier says yes → adopt ($A_i = 1$)

- If all tiers say no → don't adopt ($A_i = 0$)

**Special case - No ductwork**:

```
If home has no existing ductwork:
  - This ducted heat pump can't be installed
  - Set Ryi = ∞ (infinite cost)
  - Automatically makes all NPVs negative
  - Aᵢ = 0 (can't adopt)
  
About 30% of U.S. homes lack ductwork
This study excludes them (limitation)
Other heat pump types (mini-splits) could work, but not modeled here
```

---

### Load Profile Output

**After adoption decision, calculate load change**:

$\hat{E}_{ih} = A_i \times (E_{ih}^{\text{retrofit}} - E_{ih}^{\text{baseline}})$ (18)

**Plain language**:

- If home adopts ($A_i = 1$): 
  - $\hat{E}_{ih}$ = actual electricity change at hour $h$
- If home doesn't adopt ($A_i = 0$):
  - $\hat{E}_{ih} = 0$ (no change to grid load)

**This becomes the input to grid analysis**

---

### Key Insights from TARE Stage

**1. Regional clustering of adoption**:

```
HIGH ADOPTION REGIONS (Tier 1 + 2):
- South (Gulf Coast): Electricity cheap, gas expensive, mild climate

- California: Gas expensive, environmental policies

- Total: ~30 million homes without IRA

MODERATE ADOPTION (Tier 3 with IRA):
- Midwest, Northeast: Only with subsidies

- Total: +55 million homes with IRA (85 million total)

LOW ADOPTION:
- Areas with very cheap gas and expensive electricity

- Very cold climates where heat pump efficiency drops
```

**2. IRA policy impact**:

```
Without IRA: 30 million homes (economically attractive)
With IRA: 85 million homes (+55 million enabled by subsidies)

Subsidy impact: 183% increase in economically viable homes

Greatest impact: Moderate-income households (80-150% AMI)
  - These homes get $2,000 tax credit
  - Often tips NPV from negative to positive
  
Limited impact: Very cheap gas areas
  - Economics still don't work even with subsidies
  - Gas at $0.80/therm vs. electricity at $0.15/kWh
```

**3. Equipment replacement timing**:

```
Tier 2 adoptions depend on equipment failure cycles:
- Average furnace lifetime: 15-20 years

- Average AC lifetime: 12-15 years

- Peak replacement rate: ~6% of homes per year

Implication: Even if economically viable, gradual adoption over 10-15 years
Not all 85 million homes adopt immediately
```

---

### TARE Outputs ($g(\cdot)$ in the pipeline)

**For each of 8,000 homes**:

**Primary output**:

$A_i \in \{0, 1\}$ (Binary adoption decision) (19)

**Secondary outputs**:

- $\text{NPV}_i$ (Net present value - Tier 1)

- $\text{NPV}_i^{\text{alternative}}$ (NPV vs. replacement - Tier 2)

- $\text{NPV}_i^{\text{subsidized}}$ (NPV with subsidies - Tier 3)

- Tier classification (Which tier enabled adoption)

- $\hat{E}_{ih}$ (Load change profile - 8,760 hours)

---

## Stage 3: Grid Operational Analysis (Power Flow) {#stage-3}

### What the Grid Model Does

**Core question**: Can distribution grids physically deliver the power if economically-attractive homes adopt heat pumps?

**Why this matters**:

- Economics says: 85 million homes want to adopt (with IRA)

- But: Distribution grids were designed for historical loads

- Heat pumps create fundamentally different demand patterns

- Need to check: Will the grid actually work?

---

### Key Grid Concepts

**Distribution grid hierarchy**:

```
Transmission grid (high voltage, long distance)
  ↓
Substation (steps down voltage)
  ↓
Primary distribution lines (medium voltage, 12.47 kV most common)
  ↓
Distribution transformers (step down to household voltage)
  ↓
Secondary lines (low voltage, 240/120 V split-phase)
  ↓
Individual homes
```

**Critical constraints**:

1. **Transformer capacity**: Can't exceed rated power (kVA)

2. **Voltage quality**: Must stay within ±5% of nominal (ANSI C84.1 Range A)

3. **Power balance**: All power must come from substation

4. **Thermal limits**: Lines and transformers within rated capacity

---

### The PNNL Taxonomy Feeders

**Source**: Pacific Northwest National Laboratory Distribution System Taxonomy (PNNL-18035, Schneider et al. 2008)

**Important Note**: The original PNNL taxonomy establishes **24 prototypical feeder models**, not 57 as stated in the paper. The paper may be using an expanded version or there may be an error in the reported count.

**Feeder characteristics** (from PNNL-18035):

- **Climate regions**: 5 zones
  - West Coast (temperate)
  - North Central/Northeast (cold)
  - Southwest (hot-arid)
  - Southeast/Central (hot-cold)
  - Southeast (hot-humid)

- **Voltage classes**: Three levels
  - 12.47 kV (most common for residential)
  - 25 kV
  - 35 kV

- **Node counts**: 27 to 2,000 nodes per feeder (broader range than typical 100-1,000)

- **Transformer categories**: Four types
  - Residential
  - Commercial
  - Industrial
  - Agricultural

- **System types**: 
  - Three-phase primary distribution
  - Split-phase triplex secondary
  
**Important limitation**: Taxonomy **excludes large urban core networked systems**. It covers only **non-urban core, radial distribution feeders**. Urban, suburban, and rural loads are represented, but large networked downtown grids are not included.

---

### The Mapping Challenge (Section 2.6)

**Problem**: 

- ResStock gives individual home profiles (8,000 homes)

- Grid has aggregated nodes (27-2,000 per feeder)

- Need to assign homes to nodes realistically

**Solution**:

**Step 1 - Calculate average home load**:

$\bar{L}_{\text{home}} = \frac{1}{N_{\text{homes}}} \sum_{i} \max_h E_{ih}^{\text{baseline}}$ (20)

**Example**:

```
8,000 homes in sample
Total of all peak loads: 62,000 kW
Average: 62,000 / 8,000 = 7.75 kW per home
```

**Step 2 - Estimate homes per node**:

$H_n = \left\lfloor \frac{L_n^{\text{obs}}}{\bar{L}_{\text{home}}} \right\rfloor$ (21)

**Example - Node 17**:

```
Observed peak load: 31 kW (from taxonomy feeder data)
Homes assigned: ⌊31 / 7.75⌋ = 4 homes
```

**Step 3 - Assign specific homes to nodes**:

```
Match by:
- Climate zone (don't put Miami homes in Chicago feeder)

- Urbanization (match urban/suburban/rural)

- Housing type (single-family, multi-family)

Example - Node 17 in suburban Chicago feeder:
Randomly select 4 homes from:
  - Climate Zone 5A
  - Suburban classification
  - Single-family homes
```

**Step 4 - Aggregate loads with adoption**:

$\tilde{L}_{nh} = L_{nh}^{\text{obs}} + \sum_{i \in I_n} \hat{E}_{ih}$ (22)

**Where**:

- $L_{nh}^{\text{obs}}$ = Original observed load at node $n$, hour $h$

- $\hat{E}_{ih}$ = Load change from adopted heat pumps

- Sum over all homes at node $n$

**Example - Node 17 at peak hour (6 AM, January)**:

```
Original load: 31 kW
Home 1: Adopted, Êih = +7 kW
Home 2: Didn't adopt, Êih = 0 kW  
Home 3: Adopted, Êih = +8 kW
Home 4: Didn't adopt, Êih = 0 kW

New load: 31 + 7 + 0 + 8 + 0 = 46 kW (+48% increase!)
```

---

### Peak Hour Selection

**Critical simplification**: Analyze only the annual peak demand hour

$\hat{h} = \arg\max_h \left\{\sum_n \tilde{L}_{nh}\right\}$ (23)

**Plain language**: 
Find the hour when total system load is highest
Only test grid at this worst-case hour

**Example**:

```
Without heat pumps:
- Peak hour: July 15, 3 PM (summer AC peak)

- Total load: 2,500 kW

With 50% heat pump adoption:
- Peak hour: January 22, 6 AM (winter heating peak) 

- Total load: 3,200 kW (+28%)

Peak completely shifts from summer to winter!
```

**What this captures**:

- Worst-case capacity requirement

- Standard utility planning practice

- Infrastructure must be sized for peak

**What this misses**:

- Multi-hour sustained peaks (transformer thermal aging)

- Daily cycling effects

- Ramping constraints

---

### Three-Phase Power Flow Analysis

**The fundamental problem**: 

Given:

- Known power demands at each node

- Known network impedances

- Fixed substation voltage

Find:

- Voltage at every node

- Current through every line

- Power flow through every transformer

Check:

- Are all voltages within ±5% (ANSI C84.1 Range A)?

- Are all transformers below rated capacity?

- Are all lines within thermal limits?

- Can substation supply all the power?

---

### The Infeasibility Optimization (Section 2.7)

**Citation**: Foster, E., Pandey, A., & Pileggi, L. (2022). "Three-phase infeasibility analysis for distribution grid studies." *Electric Power Systems Research*, Volume 212, Article 108486.

**Standard power flow**: 

- Binary result: Converges or doesn't converge

- If doesn't converge → "infeasible" → no additional info

**TPIA-MP innovation**:

- If doesn't converge → **HOW infeasible?**

- **WHERE** is the bottleneck?

- **HOW MUCH** additional support needed?

**Objective function**:

$\min \sum_{n \in \mathcal{N}} \sum_{\phi \in \Phi} c_n \left[\hat{P}_{n\phi}^2 + \hat{Q}_{n\phi}^2\right]$ (24)

**Subject to**:

$\sum_{n \in \mathcal{N}} \sum_{\phi \in \Phi} \sqrt{\hat{P}_{n\phi}^2 + \hat{Q}_{n\phi}^2} = \frac{1}{\mu} \sum_{n \in \mathcal{N}} \tilde{L}_{n\hat{h}}$ (25)

**Where**:

- $\hat{P}_{n\phi}$ = Real component of infeasibility current source at node $n$, phase $\phi$

- $\hat{Q}_{n\phi}$ = Reactive component of infeasibility current source at node $n$, phase $\phi$

- $c_n$ = Penalty weight

- $\mu$ = Power factor (0.95 for heat pumps)

**Plain language**:

- Minimize total "infeasibility current sources" needed to make grid work

- These represent additional power support beyond the substation

- If sources needed at nodes → grid inadequate

**Terminology note**: The standard term is "infeasibility current sources" or "slack current sources", not "virtual power injections"

**Example analogy - Water system**:

```
Grid = Water distribution system
Substation = Water tower (main source)
Nodes = Individual houses
Infeasibility sources = Booster pumps needed at houses

If all water comes from tower: System adequate
If houses need booster pumps: System inadequate
Size of booster pumps needed: Severity of problem
```

---

### Grid Feasibility Criteria

**Four conditions must ALL be satisfied**:

**Criterion 1 - Solver convergence**:

```
Can the power flow equations find a mathematical solution?

If NO → Physics doesn't work (impossible to serve load)
If YES → Proceed to Criterion 2
```

**Criterion 2 - Power balance**:

$\hat{\mathcal{N}} = \{n \in \mathcal{N} \mid S_n > 0\}$ (26)

**Where**:

$S_n = \sum_{\phi \in \Phi} \sqrt{\hat{P}_{n\phi}^2 + \hat{Q}_{n\phi}^2}$ (27)

**Check**: 

Is $\hat{\mathcal{N}} = \{\text{substation only}\}$?

```
If YES → All power comes from substation (GOOD)
If NO → Need power at other nodes (BAD)

Example:
N̂ = {substation, node 17, node 23}
→ Grid infeasible (nodes 17 and 23 need support)
```

**Criterion 3 - Voltage quality (ANSI C84.1 Range A)**:

$0.95 \leq \frac{|V_{n\phi}|}{V_{\text{nominal}}} \leq 1.05 \quad \forall n \in \mathcal{N}, \phi \in \Phi$ (28)

**Example at Node 17**:

```
- Nominal voltage: 120 V

- Acceptable range: 114 V to 126 V (±5%)

- Simulated voltage: 108 V

- Status: VIOLATION (too low)

Why voltage violations happen:
- Long lines: High impedance → large voltage drop

- Heavy loads: More current → more voltage drop

- Heat pumps: Both long lines AND heavy loads
```

**Criterion 4 - Thermal limits**:

$S_{\text{transformer}} \leq S_{\text{rated}}$ (29)
$I_{\text{line}} \leq I_{\text{ampacity}}$ (30)

**Overall feasibility**:

$G_N = \begin{cases}
1 & \text{if all four criteria satisfied} \\
0 & \text{if any criterion violated}
\end{cases}$ (31)

---

### Transformer Stress Analysis

**Beyond binary feasibility, track equipment stress**:

**Transformer loading**:

$\text{Utilization} = \frac{S_{\text{actual}}}{S_{\text{rated}}} \times 100\%$ (32)

**Categories**:

```
- < 80%: Normal operation

- 80-100%: High stress, reduced lifetime

- > 100%: Overload, risk of failure

Example - Node 17 transformer:
- Rated capacity: 50 kVA

- Typical residential range: 10-167 kVA per IEEE C57.12.20
  (25-50 kVA most common)
- Baseline load: 31 kW ÷ 0.95 pf = 32.6 kVA (65% utilization - OK)

- With heat pump adoption: 46 kW ÷ 0.95 pf = 48.4 kVA (97% utilization - STRESSED)

- If 4th home also adopted: 54 kW ÷ 0.95 pf = 56.8 kVA (114% - OVERLOADED)
```

**Note on transformer ratings**: Residential distribution transformers typically range from 10-167 kVA per IEEE C57.12.20, with 25-50 kVA being most common for residential applications.

---

### Voltage vs. Transformer Constraints: Context Matters

**Important nuance**: Whether voltage violations or transformer overload occurs first depends on grid topology, not a universal rule:

**Suburban grids**: 

- Shorter line distances

- Higher load density

- **More vulnerable to transformer/line capacity constraints**

- Voltage typically remains acceptable

**Rural grids**:

- Long feeder lengths (high impedance)

- Lower load density

- **More affected by voltage deviations**

- Transformers often have capacity margin

**Urban grids** (note: not included in PNNL taxonomy):

- Dense loads

- Short secondary lines

- Can show transformer overloads with no voltage violations

- Example: TU Delft study found 331% transformer overloads with voltages within limits

**The paper's claim** that voltage violations are more common may be accurate for the specific feeders analyzed, but is not universally true across all grid types.

---

### Line Impedance Technical Details

**Distribution line impedances**:

- **Resistance**: 
  - 0.3-0.6 Ω/mile for main feeders (336.4-4/0 ACSR conductors)
  - 336.4 ACSR: 0.306 Ω/mile
  - 4/0 ACSR: 0.592 Ω/mile

- **Total impedance** (including reactance): 0.5-0.7 Ω/mile for main feeders

- **Lateral conductors** (#1/0 to #4 ACSR): 1.1-2.5 Ω/mile (much higher)

**Simplified voltage drop calculation**:

$\Delta V \approx \frac{P \times R + Q \times X}{V}$ (33)

**Where**:

- $\Delta V$ = Voltage drop (V)

- $P$ = Real power flow (W)

- $R$ = Line resistance (Ω)

- $Q$ = Reactive power flow (VAR)

- $X$ = Line reactance (Ω)

- $V$ = Receiving end voltage (V)

---

### Scenario Testing

**Penetration levels tested**:

```
1. 0% (Baseline): No adoption

2. Cost-based: Only economically-attractive homes (Aᵢ = 1)

3. 25%, 50%, 75%: Uniform adoption rates

4. Ductwork-limited: All homes with ductwork

5. 100%: All homes (theoretical maximum)
```

**For each scenario, test on taxonomy feeders**:

- 24 prototypical feeders (per PNNL-18035)

- Paper reports testing on 57 feeders (possible expanded version)

- 5 climate zones

- 3 urbanization levels (urban, suburban, rural)  

- 3 voltage classes (12.47 kV, 25 kV, 35 kV)

- Multiple feeder types per category

---

### Grid Analysis Outputs ($h(\cdot)$ in the pipeline)

**For each feeder $N$ and scenario**:

**Binary feasibility**:

$G_N \in \{0, 1\}$ (Can grid support this adoption level?) (34)

**Constraint details**:

- $\hat{\mathcal{N}}$ (Set of nodes with power imbalance)

- $S_n$ (Infeasibility current sources at each node - severity)

- $V_{n\phi}$ (Voltage at each node and phase)

- **Violations**: 

List of constraint violations:

  - Voltage too low/high
  - Transformer overloaded  
  - Line thermal limit exceeded
  - Power imbalance

**Upgrade needs** (estimated):

```
Transformers needing replacement: [list]
Voltage regulators needed: [locations]
Line upgrades needed: [miles]

Rough cost estimates:
- Transformer: $2,000-5,000 each (residential distribution)

- Voltage regulator: $20,000-50,000 each  

- Line reconductoring: $50,000-200,000 per mile
```

---

### Key Insights from Grid Stage

**1. Grid constraints are binding**:

```
Economics: 85 million homes want to adopt (with IRA)
Grid: 25 million homes can adopt without problems
Gap: 60 million homes BLOCKED by grid (72% reduction)

Without IRA:
Economics: 30 million homes
Grid: 15 million homes  
Gap: 15 million homes blocked (50% reduction)
```

**2. Regional patterns matter**:

```
SOUTH:

- Economic potential: 28 million homes

- Grid-constrained: 19 million homes

- Reduction: 32% (best resilience)

- Why: Existing AC infrastructure, mild winters

MIDWEST:

- Economic potential: 19 million homes

- Grid-constrained: 6 million homes

- Reduction: 68% (worst resilience)

- Why: High winter heating loads, aging infrastructure

NORTHEAST:

- Economic potential: 22 million homes

- Grid-constrained: 8 million homes

- Reduction: 64%

- Why: High heating loads, older infrastructure

WEST:

- Economic potential: 16 million homes

- Grid-constrained: 12 million homes

- Reduction: 25% (good resilience)

- Why: Mild climate, newer infrastructure
```

**3. Most common constraints (topology-dependent)**:

```
Varies by grid type:

RURAL feeders:
- Primary issue: Voltage violations (long lines)

- Secondary: Transformer overload less common

SUBURBAN feeders:
- Primary issue: Transformer overload (high density)

- Secondary: Voltage typically acceptable

The relative frequency depends on the specific mix
of feeders analyzed.
```

---

## Stage 4: Putting It All Together {#stage-4}

### Joint Economic-Grid Feasibility

**The logic**:

$\tilde{A}_{Ni} = A_i \times G_N$ (35)

**Where**:

- $A_i$ = Economic adoption decision (from TARE)

- $G_N$ = Grid feasibility (from power flow)

- $\tilde{A}_{Ni}$ = Joint feasibility (can ACTUALLY adopt)

**Truth table**:

```
Aᵢ | GN | Ã_Ni | Interpretation
---|-------|------|---------------
0  | 0     | 0    | Can't adopt (economics don't work, grid doesn't work)
0  | 1     | 0    | Can't adopt (economics don't work, even though grid OK)
1  | 0     | 0    | Can't adopt (BLOCKED BY GRID despite good economics)
1  | 1     | 1    | CAN ADOPT (both economics and grid work)
```

---

### Regional Aggregation

**Calculate penetration rate for region $R$**:

$\bar{A}_R = \frac{1}{|R|} \sum_{N \in R} \frac{1}{|I_N|} \sum_{i \in I_N} \tilde{A}_{Ni}$ (36)

**Plain language**:

1. For each feeder $N$ in region $R$

2. Count how many homes can adopt ($\tilde{A}_{Ni} = 1$)

3. Divide by total homes on that feeder

4. Average across all feeders in region

**Example - Midwest region**:

```
Feeder 1 (suburban Chicago):
- Total homes: 1,200

- Economic adopters: 480 (40%)

- Grid feasible: GN = 0 (infeasible)

- Joint adopters: 480 × 0 = 0 (0%)

Feeder 2 (urban Cleveland):
- Total homes: 800

- Economic adopters: 320 (40%)

- Grid feasible: GN = 1 (feasible)

- Joint adopters: 320 × 1 = 320 (40%)

Feeder 3 (rural Wisconsin):
- Total homes: 400  

- Economic adopters: 160 (40%)

- Grid feasible: GN = 0 (voltage violations)

- Joint adopters: 160 × 0 = 0 (0%)

Regional penetration:
Ā_Midwest = (0 + 320 + 0) / (1,200 + 800 + 400)
          = 320 / 2,400
          = 13.3% (only 1/3 of economic potential!)
```

---

### Scaling to National Market Size

**From penetration rate to absolute numbers**:

$\text{Market size}_R = \bar{A}_R \times \text{Housing stock}_R$ (37)

**Where**:

- $\text{Housing stock}_R$ = Total homes in region $R$

- From U.S. Census: ~120 million total homes

---

### The Final Results

**National market size by scenario**:

```
NO GRID CONSTRAINTS (economics only):
- Without IRA: 30 million homes

- With IRA: 85 million homes

WITH GRID CONSTRAINTS (realistic):
- Without IRA: 15 million homes (50% reduction)

- With IRA: 25 million homes (72% reduction)

THE GAP:
- With IRA: 85M - 25M = 60 million homes blocked by grid

- Without IRA: 30M - 15M = 15 million homes blocked by grid
```

**Regional breakdown**:

```
SOUTH:
- Economic (with IRA): 28M homes

- Grid-constrained: 19M homes

- Reduction: 32% (least constrained)

MIDWEST:
- Economic (with IRA): 19M homes

- Grid-constrained: 6M homes  

- Reduction: 68% (most constrained)

NORTHEAST:
- Economic (with IRA): 22M homes

- Grid-constrained: 8M homes

- Reduction: 64% (highly constrained)

WEST:
- Economic (with IRA): 16M homes

- Grid-constrained: 12M homes

- Reduction: 25% (least constrained)
```

---

## Key Data Sources {#data-sources}

### 1. ResStock Database

```
Source: National Renewable Energy Laboratory (NREL)
What: Statistically representative U.S. housing stock
Size: 550,000 building models (this study uses 8,000)
Updated: Based on 2020 Residential Energy Consumption Survey (RECS)
Geographic: All 50 states, by PUMA codes

Key characteristics sampled:
- Building type (single-family, multi-family, mobile home)

- Floor area, vintage, foundation type

- Insulation levels, window types

- Existing HVAC equipment

- Occupancy patterns, thermostat settings

- Climate zone (ASHRAE IECC)

- Urbanization (urban, suburban, rural)
```

### 2. EnergyPlus Weather Data

```
Source: U.S. Department of Energy
What: Typical Meteorological Year (TMY) data
Period: Based on 1991-2005 historical weather
Resolution: Hourly for 8,760 hours per year
Variables: Temperature, humidity, solar radiation, wind

Used for: Driving building energy simulations
```

### 3. Heat Pump Specifications

```
Source: Manufacturer data (proprietary)
Equipment: Modern cold-climate air-source heat pump (ASHP)
Specifications:
- Cooling: 3.12 tons, SEER 14.12

- Heating: 2.91 tons, HSPF 8.34

- Power factor: 0.95 lagging (typical range 0.85-0.98)

- No lockout temperature (operates at all outdoor conditions)

- Ducted system (requires existing ductwork)
```

### 4. Energy Prices

```
Source: U.S. Energy Information Administration (EIA)
What: State-level average retail prices
Variables:
- Residential electricity ($/kWh)

- Residential natural gas ($/therm)

- Projected price escalation rates

Used for: TARE economic calculations
```

### 5. Equipment Costs

```
Source: Multiple sources
Heat pump costs:
- National Residential Efficiency Measures Database (NREMD)

- Manufacturer MSRP data

- Installation labor rates by region

Replacement equipment costs:
- NREMD for furnaces, air conditioners

- By efficiency level, capacity, fuel type

Used for: TARE NPV calculations
```

### 6. IRA Subsidies

```
Source: Inflation Reduction Act legislation (2022)
Structure:
- 25C tax credit: Up to $2,000 (30% of cost)

- HOMES rebates: Up to $8,000 (income-based)

- Varies by household income relative to Area Median Income (AMI)

Income brackets:
- ≤80% AMI (low income): $8,000 rebate + $2,000 credit

- 80-150% AMI (moderate): $2,000 credit

- >150% AMI (higher income): Varies by state

Used for: TARE Tier 3 calculations
```

### 7. PNNL Taxonomy Feeders

```
Source: Pacific Northwest National Laboratory (PNNL-18035, Schneider et al. 2008)
What: 24 prototypical distribution grid networks
  Note: Paper claims 57 feeders; original PNNL taxonomy has 24
  
Categories:
- 5 climate zones (West Coast, North Central/Northeast, Southwest, 
  Southeast/Central, Southeast hot-humid)
- 3 urbanization levels (urban, suburban, rural)
  - Excludes large urban core networked systems
  - Covers only non-urban core radial feeders
- 3 voltage classes (12.47 kV, 25 kV, 35 kV)

Characteristics:
- 27-2,000 nodes per feeder (broader than typical 100-1,000)

- Four transformer categories (residential, commercial, industrial, agricultural)

- Three-phase primary and split-phase triplex secondary sections

- Line impedances, transformer ratings, load profiles

Used for: Grid power flow analysis
```

### 8. Baseline Load Profiles

```
Source: NREL EULP (End-Use Load Profiles) + synthetic data
What: Pre-existing load patterns on taxonomy feeders
Resolution: Hourly for 8,760 hours
Components:
- Residential loads

- Commercial loads  

- Industrial loads (where applicable)

Used for: Baseline grid conditions before heat pump adoption
```

---

## Complete Example: One Home's Journey {#complete-example}

Let me trace a single home through the entire pipeline to show how all the pieces connect.

### Meet Home #6,247

**Location**: Minneapolis, Minnesota
**Climate Zone**: 7B (Cold, dry)
**Urbanization**: Suburban
**Building**: Single-family home, 1,950 sq ft, built 1992
**Existing Equipment**: 

- Natural gas furnace (80% AFUE, 20 years old)

- Central air conditioner (SEER 10, 15 years old)

**Occupancy**: Family of 3
**Household Income**: $92,000 (close to area median income)

---

### Stage 1: ResStock + EnergyPlus Simulation

**Inputs to simulation**:

```
Building envelope:
- Floor area: 1,950 sq ft

- Insulation: R-13 walls, R-30 attic (moderate)

- Windows: Double-pane (older)

- U-factor: 0.52 W/(m²·°C)

- Thermal mass: 6,200 Wh/°C

Weather (Minneapolis TMY):
- January avg low: -7°F

- July avg high: 83°F

- Heating degree days: 7,800

- Cooling degree days: 750

Occupant behavior:
- Heating setpoint: 70°F

- Cooling setpoint: 73°F

- Occupancy: 3 people, typical weekday/weekend schedules
```

**Baseline scenario simulation** (existing equipment):

```
Energy use:
- Annual electricity: 14,200 kWh
  - Cooling: 1,800 kWh (summer)
  - Appliances/lighting: 12,400 kWh (year-round)
- Annual natural gas: 1,150 therms
  - Heating: 1,050 therms (winter)
  - Water heating: 100 therms (year-round)

Peak loads:
- Summer (July 15, 4 PM): 7.2 kW (AC running)

- Winter (January 22, 6 AM): 5.8 kW (no electric heating, just baseload)

Key insight: Gas furnace means low winter electricity peak
```

**Retrofit scenario simulation** (with heat pump):

```
Energy use:
- Annual electricity: 24,800 kWh
  - Heating: 12,000 kWh (winter - NOW ELECTRIFIED)
  - Cooling: 1,600 kWh (summer - slightly less, more efficient)
  - Appliances/lighting: 12,400 kWh (unchanged)
  - Water heating: 100 therms gas (unchanged - didn't retrofit this)
- Annual natural gas: 100 therms
  - Heating: 0 therms (now electric)
  - Water heating: 100 therms (still gas)

Peak loads:
- Summer (July 15, 4 PM): 6.8 kW (actually LOWER - more efficient cooling)

- Winter (January 22, 6 AM): 16.4 kW (ALMOST TRIPLED!)

Key insight: Massive winter peak increase from electrified heating
```

**Delta profile** (what changes):

$\Delta E_{ih} = E_{ih}^{\text{retrofit}} - E_{ih}^{\text{baseline}}$

```
Annual change: +10,600 kWh electricity, -1,050 therms gas

Peak changes:
- Summer: -0.4 kW (slightly better)

- Winter: +10.6 kW (huge increase!)

Hourly patterns:
- Most winter mornings: +8 to +12 kW

- Cold snaps: +15 to +18 kW (backup resistive heating kicks in)

- Summer: -0.2 to +0.5 kW (varies by hour)
```

**Why winter peaks are so high**:

```
January 22, 6 AM:
- Outdoor temp: -5°F

- Indoor temp: 66°F (dropped overnight)

- Target temp: 70°F

- Temperature difference: 75°F (huge!)

Heating load needed:
Q = U × A × ΔT + infiltration + recovering setback
Q = 0.52 × 180 m² × 41°C + losses
Q ≈ 18,000 W (61,400 BTU/hr)

Heat pump performance at -5°F:
- COP: 1.65 (much lower than at 40°F)

- Some backup resistive heating needed

- Total electricity: 18,000 / 1.65 + backup ≈ 16,400 W

This 16.4 kW peak is the problem for the grid!
```

---

### Stage 2: TARE Economic Analysis

**Inputs from ResStock**:

$E^{\text{baseline}} = 14,200 \text{ kWh/year}$
$F^{\text{baseline}} = 1,150 \text{ therms/year}$

$E^{\text{retrofit}} = 24,800 \text{ kWh/year}$
$F^{\text{retrofit}} = 100 \text{ therms/year}$

**Minneapolis energy prices** (2025):

```
Electricity: $0.135/kWh
Natural gas: $1.15/therm
Price escalation: 2% per year
```

**Equipment costs**:

```
Heat pump:
- Equipment: $10,500

- Installation: $3,500

- Total: $14,000

Replacement alternatives (if furnace/AC died):
- New gas furnace: $4,500

- New AC: $3,200

- Total: $7,700

IRA subsidy available:
- Household income: $92,000

- Area median income: $95,000

- Ratio: 92/95 = 97% (moderate income bracket)

- Eligible for: $2,000 tax credit
```

**Annual cost calculations**:

$B_{y1} = \beta_1 \left[\lambda_1^{\text{elec}} \sum_h E_{ih}^{\text{baseline}} + \lambda_1^{\text{gas}} \sum_h F_{ih}^{\text{baseline}}\right]$

```
BASELINE costs:
Year 1: ($0.135 × 14,200) + ($1.15 × 1,150) = $1,917 + $1,323 = $3,240
Year 2-15: Increase 2% per year for fuel price escalation
Also: Will need to replace furnace in ~Year 8 ($4,500)

Present value of baseline (15 years):
Fuel: $37,800
Replacement: $3,100 (discounted to present)
Total: $40,900

RETROFIT costs:
Year 1: ($0.135 × 24,800) + ($1.15 × 100) = $3,348 + $115 = $3,463
Year 2-15: Increase 2% per year
No equipment replacements needed

Present value of retrofit (15 years):
Fuel: $40,400
Equipment: $14,000 upfront
Total: $54,400
```

**Tier 1: Direct economic benefit**:

$\text{NPV} = \sum_{t=1}^{15} \frac{B_{yi} - R_{yi}}{(1.07)^t} - C_{\text{equipment}}$

```
NPV = $40,900 - $54,400 = -$13,500

NEGATIVE → Does not meet Tier 1
Heat pump costs more than keeping existing equipment
```

**Tier 2: Better than alternative** (furnace dies):

$\text{NPV}^{\text{alternative}} = \sum_{t=1}^{15} \frac{B_{yi} - R_{yi}}{(1.07)^t} - (C_{\text{equipment}} - C_{\text{replacement}})$

```
Present value of new conventional equipment:
Fuel: $37,800 (same as baseline)
Equipment: $7,700 upfront (furnace + AC)
Total: $45,500

NPV^alternative = $45,500 - $54,400 = -$8,900

STILL NEGATIVE → Does not meet Tier 2
Heat pump costs $8,900 more even when comparing to replacements
```

**Tier 3: With IRA subsidy**:

$\text{NPV}^{\text{subsidized}} = \sum_{t=1}^{15} \frac{B_{yi} - R_{yi}}{(1.07)^t} - (C_{\text{equipment}} - C_{\text{replacement}} - \text{rebate}_i)$

```
Retrofit cost with subsidy:
Equipment: $14,000 - $2,000 = $12,000 effective
Fuel: $40,400
Total: $52,400

NPV^subsidized = $45,500 - $52,400 = -$6,900

STILL NEGATIVE → Does not meet Tier 3!
Even with $2,000 subsidy, loses $6,900
```

**Economic decision**:

$A_{6247} = 0 \quad \text{(DO NOT ADOPT)}$

**Why?**

- Minneapolis has relatively cheap natural gas ($1.15/therm)

- Electricity more expensive ($0.135/kWh)

- Cold climate means heat pump efficiency drops

- High electricity use in winter doesn't pay off

- Even $2,000 subsidy isn't enough

**Load profile output**:

$\hat{E}_{ih} = A_{6247} \times (E_{ih}^{\text{retrofit}} - E_{ih}^{\text{baseline}}) = 0 \times \Delta E_{ih} = 0$

Since home doesn't adopt, no change to grid load.

---

### Hypothetical: What if Home #6,247 DID Adopt?

Let's imagine this home was in a "forced adoption" scenario (100% penetration test), so $A_i = 1$ regardless of economics.

### Stage 3: Grid Analysis

**Home is assigned to feeder**:

```
Climate zone: 7B (Cold) → Matches to cold-climate feeder
Urbanization: Suburban → Suburban feeder type
Selected feeder: From PNNL taxonomy (cold climate, suburban type)
  - Node count: 27-2,000 range (this feeder has 485 nodes)
  - Residential transformers: 342
  - Peak baseline load: 3,450 kW
```


**Home assigned to specific node**:

```
Node 187 characteristics:
- Observed baseline peak: 27 kW

- Estimated homes: H_n = ⌊27 / 7.75⌋ = 3.5 → 4 homes

- Transformer: 50 kVA center-tapped

- Line distance from substation: 0.8 miles
```

**Penetration scenario: 50%**:

```
4 homes at Node 187:
- Home A: Adopts (Aᵢ = 1)

- Home B: Doesn't adopt (Aᵢ = 0)

- Home C: Adopts (Aᵢ = 1) ← THIS IS HOME #6,247

- Home D: Doesn't adopt (Aᵢ = 0)

2 out of 4 homes adopt (50% penetration)
```


**Load at Node 187**:

$\tilde{L}_{n\hat{h}} = L_{n\hat{h}}^{\text{obs}} + \sum_{i \in I_n} \hat{E}_{i\hat{h}}$


```
Peak hour (January 22, 6 AM):

Baseline load: 27 kW

Additional from Home A: +10.2 kW (similar profile to #6,247)
Additional from Home C (#6,247): +10.6 kW
Additional from Homes B & D: 0 (didn't adopt)

New load: 27 + 10.2 + 10.6 = 47.8 kW

Increase: 77%!
```


**Transformer check**:

$\text{Utilization} = \frac{S_{\text{actual}}}{S_{\text{rated}}} \times 100\%$


```
Transformer at Node 187:
- Rated capacity: 50 kVA (typical range 10-167 kVA per IEEE C57.12.20)

- Baseline load: 27 kW ÷ 0.95 pf = 28.4 kVA (57% utilization - OK)

- New load: 47.8 kW ÷ 0.95 pf = 50.3 kVA (101% utilization - OVERLOAD)

Status: TRANSFORMER OVERLOAD
Severity: 1% over rated capacity (minor, but failure risk over time)
```


**Voltage check**:

$\Delta V \approx \frac{P \times R + Q \times X}{V}$


```
Voltage calculation (simplified):
V_187 ≈ V_substation - (I × Z × distance)

Where:
- V_substation = 7,200 V (nominal, phase-to-neutral)

- I = current flowing to Node 187

- Z = line impedance (0.3 Ω/mile for this line - within 0.3-0.6 Ω/mile range)

- distance = 0.8 miles

Baseline:
Current: 27,000 W / (7,200 V × √3) ≈ 2.2 A per phase
Voltage drop: 2.2 × 0.3 × 0.8 = 0.53 V
Node voltage: 7,200 - 0.53 = 7,199.47 V (99.99% of nominal - PERFECT)

With heat pumps:
Current: 47,800 W / (7,200 V × √3) ≈ 3.8 A per phase
Voltage drop: 3.8 × 0.3 × 0.8 = 0.91 V  
Node voltage: 7,200 - 0.91 = 7,199.09 V (99.99% of nominal - STILL OK)

At secondary (household voltage):
Transformer steps down 7,199 V → 119.8 V (OK, within 114-126 V range per ANSI C84.1 Range A)

Status: VOLTAGE OK (barely - at 0.8 miles)
```


**But what about longer lines?**:

```
Node 203 (rural area, 3.2 miles from substation):
Similar load increase
Current: 3.8 A
Voltage drop: 3.8 × 0.3 × 3.2 = 3.6 V
Node voltage: 7,200 - 3.6 = 7,196.4 V
Secondary voltage: 119.4 V → 117.7 V after transformer

Still OK, but cutting it close!

Node 247 (very rural, 5.8 miles):
Voltage drop: 3.8 × 0.3 × 5.8 = 6.6 V
Node voltage: 7,193.4 V
Secondary voltage: 115.2 V after transformer

Getting marginal - near lower limit of 114 V

With 75% or 100% penetration, this node would violate!
```


**Power flow optimization result**:

For this feeder at 50% penetration:

**Criterion checks**:

- Solver: CONVERGES OK

- Power balance: $\hat{\mathcal{N}} = \{\text{substation}\}$ OK

- Voltage: 3 nodes at >5 miles violate FAILS

- Thermal: Some transformers overloaded FAILS

**Grid feasibility**: 

$G_N = 0 \quad \text{(INFEASIBLE due to voltage violations and transformer overload)}$

**Infeasibility current sources** ($\hat{S}_n$) **for failed nodes**:

```
- Node 247: Needs 2.1 kW additional support

- Node 268: Needs 1.8 kW additional support  

- Node 291: Needs 3.2 kW additional support
```


**Upgrade needed**: 

```
- Install voltage regulator at mile 4 ($35,000)

- OR: Reconductor last 2 miles with larger wire ($180,000)

- Replace 5-8 overloaded transformers ($15,000-40,000 total)
```


---

### Stage 4: Final Outcome

**If Home #6,247 had adopted** (hypothetical):

$\tilde{A}_i = A_i \times G_N = 0 \times 0 = 0$

```
Economic feasibility: Aᵢ = 0 (didn't make economic sense)
Grid feasibility: GN = 0 (feeder infeasible at this penetration)
Joint feasibility: Ã_i = 0 × 0 = 0

Blocked by: BOTH economics AND grid!
```

**Actual outcome**:

```
Economics blocked adoption before grid even mattered
Home keeps existing gas furnace + AC
No impact on grid
```


---

### Summary of Home #6,247's Journey

**ResStock simulation showed**:

- Heat pump would increase winter electricity peak from 5.8 kW to 16.4 kW

- Annual electricity up 75% (+10,600 kWh)

- Annual gas down 91% (-1,050 therms)

**TARE economic analysis showed**:

- NPV = -$13,500 (Tier 1 fail)

- NPV$^{\text{alternative}}$ = -$8,900 (Tier 2 fail)

- NPV$^{\text{subsidized}}$ = -$6,900 (Tier 3 fail)

- Conclusion: Doesn't adopt ($A_i = 0$)

- Reason: Cheap gas, expensive electricity, cold climate

**Grid analysis showed** (hypothetical if forced to adopt):

- Would cause transformer overload (101% utilization)

- Would contribute to voltage violations on rural nodes

- Feeder infeasible at 50% penetration ($G_N = 0$)

- Needs grid upgrades before higher adoption

**Final outcome**:

- Joint feasibility: $\tilde{A}_i = 0$

- Home #6,247 CANNOT adopt due to economic constraints

- Even if economics worked, grid would be a problem

- This illustrates why Midwest has such low adoption

---

### What Would Enable Home #6,247 to Adopt?

**Option 1: Change economics**:

```
Make gas more expensive OR electricity cheaper:
- Carbon tax on gas: +$0.50/therm → NPV becomes positive

- Time-of-use electricity rates: Save 30% on heat pump → Positive

- Larger subsidy: $5,000 instead of $2,000 → Positive
```


**Option 2: Upgrade the grid**:

```
Install voltage regulator: $35,000 (serves ~200 homes)
- Cost per home: $175

- Enables adoption without voltage violations

OR:
Replace transformers: $3,000 each (serves ~4 homes each)
- Cost per home: $750

- Enables adoption without overload
```


**Option 3: Demand response**:

```
Smart controls:
- Pre-heat home before peak hours (6-9 AM)

- Run heat pump 4-6 AM when grid less stressed

- Reduce peak from 16.4 kW to 11.2 kW

- Keeps transformer below 80% utilization

- Doesn't solve voltage issues though
```


**Option 4: Better heat pump**:

```
Newer generation:
- Higher COP at cold temperatures (2.2 instead of 1.65)

- Less backup resistive heating needed

- Peak reduced from 16.4 kW to 13.1 kW

- Improves both economics AND grid impacts
```


---

## Summary of Corrections Made

### Critical Corrections

1. Changed "57 feeders" to "24 feeders" with note about paper discrepancy

2. Corrected Foster & Pandey citation from 2020 to 2022 (EPSR Vol. 212)

3. Added caveat that taxonomy excludes large urban core networked systems

### Minor Adjustments

4. Changed "virtual power injections" to "infeasibility current sources"

5. Expanded node count range from "100-1,000" to "27-2,000"

6. Added nuance to voltage vs. transformer claim—noted topology dependence

7. Added thermal limit constraints to grid feasibility criteria

8. Updated equation formatting to IEEE power systems conventions

9. Added power factor range (0.85-0.98) with 0.95 as standard

10. Added transformer rating range (10-167 kVA per IEEE C57.12.20)

11. Specified line impedance details (main feeders vs. laterals)

12. Added ANSI C84.1 Range A specification for voltage standards

### Verified as Accurate
- Climate zones, voltage classes, transformer types

- ANSI C84.1 ±5% voltage standard

- Technical parameters: 12.47 kV primary, 240/120V secondary, 0.95 power factor

- Home-to-node mapping methodology

- Infeasibility optimization concept

---

**This completes the corrected comprehensive walkthrough with all verification report corrections applied.**
