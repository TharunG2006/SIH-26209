# 5. Space Weather Immunity: Random Matrix Theory (RMT)

Satellite telemetry is highly susceptible to external environmental shocks, specifically Solar Flares (CME) and Geomagnetic storms. A traditional AI will see all sensors fluctuate and trigger a catastrophic alarm. Astrovia uses **Random Matrix Theory** combined with live NOAA API data to filter these out.

### 5.1 The Covariance Matrix
We construct a covariance matrix $C$ of the residual errors $E$ across all sensors $N$ over time window $T$:
$$ C = \frac{1}{T} E E^T $$

### 5.2 The Marčenko-Pastur Distribution
In a healthy state, the eigenvalues $\lambda$ of $C$ should follow the Marčenko-Pastur (MP) distribution. The theoretical maximum eigenvalue for pure random noise is:
$$ \lambda_+ = \sigma^2 \left( 1 + \sqrt{\frac{N}{T}} \right)^2 $$

### 5.3 Fault Isolation Logic
We perform Eigenvalue Decomposition on $C$. 
1. **Hardware Fault (Internal):** A localized hardware failure causes a single subset of sensors to correlate heavily. This causes exactly **one** dominant eigenvalue $\lambda_{max}$ to "detach" and spike far above $\lambda_+$, while the rest of the distribution remains stable.
2. **Space Weather (External Shock):** A solar flare affects the entire satellite hull simultaneously. This causes the **entire distribution of eigenvalues** to shift rightward. The center of mass of the eigenvalues moves, but no single eigenvalue detaches disproportionately.

By cross-referencing this eigenvalue distribution shift with the **NOAA live Space Weather API**, Astrovia can mathematically veto the anomaly, classifying it as a benign environmental shock rather than a hardware failure, thereby dropping false alarms to zero.
