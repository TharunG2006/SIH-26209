# 3. First-Mover Causal Inference (Explainability)

When a catastrophic event occurs on a satellite, dozens of sensors will trigger simultaneously (e.g., a power drop causes thermals to crash, which causes comms to fail). If an AI just flags all of them, the operator is overwhelmed. Astrovia automatically pinpoints the **root cause** by finding the first mover.

### 3.1 Lead-Lag Calculation
For a sequence of anomalies spanning time $t_{start}$ to $t_{end}$, we isolate the subset of channels $C_{anom}$ that crossed their robust thresholds. 
For each channel $c \in C_{anom}$, we find the exact timestep $t_{deviation}^{(c)}$ where the gradient of the error exceeded the baseline variance:

$$ t_{deviation}^{(c)} = \arg\min_{t \in [t_{start}, t_{end}]} \{ t \mid Z_{t, c} > \tau \} $$

The channels are then sorted by $t_{deviation}$. The sensor that moved *first* is identified as the causal root. This mathematically proves the origin of the fault (e.g. "The battery failed 3 seconds before the thermal system triggered") without requiring computationally heavy black-box interpreters like SHAP.
