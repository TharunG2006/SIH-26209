# 4. Predictive Maintenance: CUSUM Drift Analysis

To predict a failure *before* a threshold is crossed, Astrovia employs a **Cumulative Sum (CUSUM)** algorithm. This detects microscopic, slow degradation (e.g., a reaction wheel bearing slowly losing lubrication over weeks) that traditional static alarms completely ignore.

### 4.1 CUSUM Mathematics
Let $Z_{t, c}$ be the robust error score. We accumulate the drift that exceeds an allowable drift allowance $\delta$ (typically 0.5 standard deviations):

$$ S_{t, c} = \max(0, S_{t-1, c} + Z_{t, c} - \delta) $$

### 4.2 Time-To-Failure (TTF) Extrapolation
When $S_{t, c}$ exhibits a positive gradient (monotonic growth), we calculate the degradation rate $\Delta S$ over a recent window $W$:

$$ \Delta S = \frac{S_{t, c} - S_{t-W, c}}{W} $$

Given a critical failure threshold $S_{critical}$ (e.g., 5.0), the remaining Time-To-Failure (in timesteps) is linearly extrapolated:

$$ \text{TTF}_{timesteps} = \frac{S_{critical} - S_{t, c}}{\Delta S} $$

This is then converted into real-world time by multiplying by the exact median polling interval of the telemetry over the current capture:
$$ \text{TTF}_{seconds} = \text{TTF}_{timesteps} \times \text{median}(\Delta t_{telemetry}) $$

This gives mission control actionable lead time (days/hours/minutes) to route power away from the failing subsystem.
