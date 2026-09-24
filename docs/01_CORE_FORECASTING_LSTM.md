# 1. Core Forecasting: Multivariate Long Short-Term Memory (LSTM) Networks

To detect anomalies, Astrovia first learns the normal operational boundaries of the satellite by forecasting future telemetry states based on historical windows.

### 1.1 Architecture
For each telemetry channel $c_i$, an independent LSTM model is trained. Unlike purely univariate models, the inputs to this LSTM are **multivariate**—meaning the forecast for the reaction wheel takes into account power draw, thermal states, and attitude simultaneously.

Given a time-series matrix $X = [x_1, x_2, \dots, x_t]$ where $x_t \in \mathbb{R}^n$, the LSTM processes a sliding window of size $W$:

$$ h_t, c_t = \text{LSTM}(x_t, h_{t-1}, c_{t-1}) $$
$$ \hat{y}_{t+1} = W_{out} h_t + b_{out} $$

### 1.2 The Residual Error Matrix
The core of the detection engine is not the forecast itself, but the **error**. The residual error at time $t$ for channel $c$ is:
$$ e_{t, c} = |y_{t, c} - \hat{y}_{t, c}| $$

By tracking $e_{t, c}$, we strip away the normal operating dynamics of the satellite and are left purely with the "surprise" factor.
