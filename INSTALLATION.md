# Installation & Setup Guide

## System Requirements

### Software

- **Python:** 3.8 or higher
- **OS:** Windows, Linux, or macOS
- **LaTeX** (optional): For compiling the technical report

### Hardware (for running the scripts)

- **Motors:** 2× Dynamixel XC330-W350-T servos
- **Sensor:** Bosch BNO055 IMU (9-axis)
- **Interfaces:**
  - U2D2 (Dynamixel Protocol 2.0 communication)
  - FTDI USB-to-Serial adapter
- **Communications:** 2 USB ports available

---

## Installation Steps

### 1. Python Environment Setup

**Option A: Using pip (recommended)**

```bash
# Navigate to the project directory
cd RENDU_FINAL

# Install all required dependencies
pip install -r requirements.txt
```

**Option B: Using conda (alternative)**

```bash
conda create -n continuum-robot python=3.9
conda activate continuum-robot
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import numpy, pandas, matplotlib, scipy, serial; print('✓ All packages installed')"
```

---

## Hardware Configuration

### Port Mapping

By default, the scripts expect:

- **Motor Controller (U2D2):** COM8 (Windows) or /dev/ttyUSB0 (Linux)
- **IMU Sensor (FTDI):** COM3 (Windows) or /dev/ttyUSB1 (Linux)

### Modifying Ports

Edit the port configuration in each script:

**MotorWidebandID.py:**

```python
self.entry_motor.insert(0, "COM8")      # Change to your motor port
self.entry_sensor.insert(0, "COM3")     # Change to your sensor port
```

**SequentialJointCalibration.py:**

```python
self.MOTOR_PORT = "COM8"                # Change as needed
self.SENSOR_PORT = "COM3"               # Change as needed
```

**robot_performance_tests.py:**

```python
self.motor_port = "COM8"                # Change as needed
self.sensor_port = "COM3"               # Change as needed
```

---

## Running the Scripts

### Basic Workflow

**1. System Identification (Frequency Analysis)**

```bash
python python_scripts/MotorWidebandID.py
```

- Generates a logarithmic chirp signal (0.1-3 rad/s)
- Acquires input/output data
- Computes Bode diagram
- Exports CSV files with frequency response

**2. Transfer Function Estimation**

```bash
python python_scripts/TransferFunctionEstimator.py
```

- Load the Bode CSV from step 1
- Auto-fit or manually adjust poles/zeros
- Validates the identified transfer function H(s)

**3. Automated Calibration**

```bash
python python_scripts/SequentialJointCalibration.py
```

- Sweeps the arm through ±30° with ramped velocity
- Correlates encoder counts with IMU yaw
- Extracts K_i coefficients via linear regression
- Exports `Calibration_Ki.json`

**4. Performance Testing**

```bash
python python_scripts/robot_performance_tests.py
```

- Executes automated tests:
  - Step response (20° command)
  - Disturbance rejection (1 kg load)
  - Return-to-zero accuracy
- Exports results and metrics to CSV

---

## Troubleshooting

### Issue: "Port already in use" Error

**Cause:** Hardware already connected to another application
**Solution:**

```bash
# Close any other programs using the ports (serial monitor, etc.)
# Unplug and reconnect the USB cables
# Run the script again
```

### Issue: "No module named 'dynamixel_sdk'"

**Cause:** Package not installed
**Solution:**

```bash
pip install dynamixel-sdk --upgrade
```

### Issue: "Failed to open port COM8"

**Cause:** Incorrect port number or hardware not connected
**Solution:**

```bash
# Check available COM ports (Windows):
# Device Manager → Ports (COM & LPT)

# Or in Python:
import serial
ports = [p.device for p in serial.tools.list_ports.comports()]
print(ports)  # Shows all available ports
```

### Issue: IMU data not reading

**Cause:** Wrong baudrate or IMU not responding
**Solution:**

- Verify BNO055 is on the correct COM port
- Check baudrate: 115200 bps expected
- Verify USB-to-serial adapter working (FTDI drivers)

---

## Compiling the Technical Report

If you have LaTeX installed:

```bash
cd RENDU_FINAL

# Compile PDF (3 passes for references)
pdflatex Rapport_LaTeX.tex
bibtex Rapport_LaTeX
pdflatex Rapport_LaTeX.tex
pdflatex Rapport_LaTeX.tex

# Output: Rapport_LaTeX.pdf
```

### LaTeX Installation

- **Windows:** MiKTeX (https://miktex.org/download)
- **Linux:** `sudo apt install texlive-full`
- **macOS:** MacTeX (https://tug.org/mactex/)

---

## File Structure After Running Scripts

```
RENDU_FINAL/
├── README.md                    # Project overview
├── INSTALLATION.md              # This file
├── Rapport_LaTeX.tex            # Technical report source
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── CHANGELOG.md                 # Version history
├── .gitignore                   # Git ignore rules
│
├── figures/                     # Report figures
│   ├── Wang_design.png
│   ├── Systeme_angle.png
│   ├── BODE_DIAGRAM.png
│   ├── performance_test_step_response.png
│   └── performance_test_disturbance_rejection.png
│
├── python_scripts/              # Main Python scripts
│   ├── MotorWidebandID.py
│   ├── SequentialJointCalibration.py
│   ├── TransferFunctionEstimator.py
│   ├── robot_performance_tests.py
│   └── dynamixel_controller.py
│
└── results/                     # Generated outputs (after running)
    ├── ID_Result_*.csv          # System identification data
    ├── Calibration_Ki.json      # Calibration coefficients
    └── performance_test_*.csv   # Test results
```

---

## Support & Contact

For questions or issues:

- **Author:** Louis Horak
- **Email:** louis.horak@epfl.ch
- **Lab:** CREATE Lab, EPFL

---

**Last Updated:** December 2024
