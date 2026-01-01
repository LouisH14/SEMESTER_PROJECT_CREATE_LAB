# Active Control of a Continuum Robot with Sequential Locking

**Louis Horak** - CREATE Lab, EPFL - December 2024

This repository contains the Python control system and technical report for a hybrid continuum robotic arm with closed-loop control.

---

## Hardware Setup

### Components:

- 2× Dynamixel XC330 motors (differential tendon drive)
- BNO055 IMU (distal orientation sensing)
- 4× optical encoders (3.75° resolution)
- U2D2 interface (motor communication)
- FTDI adapter (IMU communication)

### Port Configuration:

- Motors: COM8, 57600 baud
- IMU: COM3, 115200 baud

---

## Installation

```bash
pip install -r requirements.txt
```

See `INSTALLATION.md` for detailed setup instructions and troubleshooting.

---

## Technical Report

The complete technical report with methodology, results, and analysis is available as a PDF. The report source (LaTeX) is maintained locally and generated via Overleaf.

Key sections included in the report:

- System identification and transfer function analysis
- P-controller design with stability analysis
- Experimental validation (step response, disturbance rejection)
- Complete discussion and conclusions

---

## Python Scripts Usage

### 1. System Identification (`MotorWidebandID.py`)

Performs frequency-domain system identification using chirp excitation.

```bash
python python_scripts/MotorWidebandID.py
```

**Steps:**

1. Connect hardware and verify ports
2. Click "Connection" to initialize motors and IMU
3. Click "Generate & Preview" to create chirp signal
4. Click "Start Identification" to execute the test
5. Results saved to `ID_Result_YYYYMMDD-HHMMSS_bode.csv` and `_time.csv`

**Output:** Bode diagram data (magnitude, phase vs frequency)

---

### 2. Transfer Function Estimation (`TransferFunctionEstimator.py`)

Fits a transfer function model to measured Bode data.

```bash
python python_scripts/TransferFunctionEstimator.py
```

**Steps:**

1. Load Bode CSV file from system identification
2. Click "Auto Fit" or manually adjust poles/zeros
3. Validate fit quality in magnitude and phase plots
4. Transfer function displayed in the interface

**Output:** Transfer function H(s) with estimated parameters

---

### 3. Calibration (`SequentialJointCalibration.py`)

Calibrates K_i coefficients relating joint angles to IMU yaw.

```bash
python python_scripts/SequentialJointCalibration.py
```

**Steps:**

1. Connect hardware
2. Click "Start Calibration" for automated ±30° sweep
3. Linear regression computes K_i coefficients
4. Results saved to `Calibration_Ki.json`

**Output:** K_i coefficients for state estimation

---

### 4. Performance Testing (`robot_performance_tests.py`)

Runs automated tests with real-time visualization.

```bash
python python_scripts/robot_performance_tests.py
```

**Available Tests:**

- **Step Response:** 20° step command, measures rise time, settling time, overshoot
- **Disturbance Rejection:** Apply/remove 1kg load, measures recovery time
- **Reset to Zero:** Returns arm to neutral position

**Output:**

- Real-time plots (yaw angle, control signal, motor currents)
- CSV data files: `performance_test_*.csv`
- Metrics files: `performance_test_*_metrics.txt`

---

## Hardware Interface (`dynamixel_controller.py`)

Low-level hardware abstraction layer. Used by all other scripts.

**Features:**

- Dynamixel Protocol 2.0 communication
- Sync read/write for dual motors
- IMU serial communication with median filtering
- Multiple control modes: PWM, velocity, position, current

---

## Results Summary

The implemented P-controller (Kp = -0.5) achieves:

- Rise time: 1.07s
- Settling time: 1.67s
- Steady-state error: -0.11°
- Disturbance rejection: ~1.5s recovery time for 1kg load

---

## License

MIT License - See `LICENSE` file for details.
