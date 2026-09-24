# 2. Robust Statistical Thresholding (Non-Parametric)

Static alarm limits (e.g., "Alert if temperature > 80°C") cause severe alarm fatigue in satellite operations because they trigger on harmless operational changes. Astrovia uses **Median Absolute Deviation (MAD)** to create robust, dynamic thresholds that adapt automatically to the noise floor of each specific sensor.

### 2.1 Robust Z-Scores
Standard deviation ($\sigma$) is heavily skewed by anomalies (e.g., one huge spike ruins the variance calculation). MAD is robust against extreme outliers.

The median error for a channel over a historical window is:
$$ \tilde{e}_c = \text{median}(e_{1, c}, \dots, e_{T, c}) $$

The MAD is computed as:
$$ \text{MAD}_c = \text{median}(|e_{t, c} - \tilde{e}_c|) $$

The robust anomaly score (z-score) at time $t$ is:
$$ Z_{t, c} = \frac{e_{t, c} - \tilde{e}_c}{\max(\text{MAD}_c \cdot 1.4826, \epsilon)} $$

If $Z_{t, c} > \tau$ (where $\tau$ is a dynamic threshold, typically 3.0 to 5.0), the channel is flagged as anomalous. This ensures that a naturally noisy sensor doesn't drown out a quiet sensor, as they are normalized onto a common statistical footing.
