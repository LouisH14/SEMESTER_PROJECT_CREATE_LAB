#!/usr/bin/env python3
"""
robot_performance_tests.py
Automated performance validation tests for the robotic arm.

Tests:
1. Step Response Test: Measures rise time, settling time, and steady-state error
2. Active Stiffness & Disturbance Rejection: Logs response to manual disturbances

Author: Antigravity AI Assistant
Date: 2025-12-19
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import sys
import time
import serial
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import Dynamixel controller
try:
    from dynamixel_controller import DynamixelController, BaseModel
except ImportError:
    print("Error: dynamixel_controller.py not found.")
    sys.exit(1)


class MedianFilter:
    """Median filter for IMU data to remove noise spikes."""

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def update(self, value):
        """Add new value and return median of current window."""
        self.buffer.append(value)
        if len(self.buffer) > 0:
            return float(np.median(self.buffer))
        return value

    def reset(self):
        """Clear the filter buffer."""
        self.buffer.clear()


class RobotController:
    """Low-level controller for robot actuation and sensing."""

    def __init__(
        self,
        motor_port="COM8",
        sensor_port="COM3",
        motor_baudrate=57600,
        sensor_baudrate=115200,
    ):
        self.MOTOR_PORT = motor_port
        self.SENSOR_PORT = sensor_port
        self.MOTOR_BAUDRATE = motor_baudrate
        self.SENSOR_BAUDRATE = sensor_baudrate

        self.motor_controller = None
        self.sensor_serial = None
        self.median_filter = MedianFilter(window_size=5)

        self.K_GAIN = -0.5  # P-Controller Gain (matches RealTimeYawControl)
        self.MAX_U = 5.0  # Maximum velocity command (rad/s)
        # Motor current scaling (mA per LSB). Default X-series ~2.69 mA
        self.current_scale_mA = 2.69

    def set_current_scale(self, scale_mA: float):
        try:
            self.current_scale_mA = float(scale_mA)
        except Exception:
            pass

    def connect(self):
        """Initialize hardware connections."""
        print(f"Connecting motors on {self.MOTOR_PORT}...")
        self.motor_controller = DynamixelController(
            port_name=self.MOTOR_PORT,
            motor_list=[BaseModel(1), BaseModel(0)],
            baudrate=self.MOTOR_BAUDRATE,
        )
        self.motor_controller.activate_controller()
        self.motor_controller.set_operating_mode_all("velocity_control")
        self.motor_controller.torque_on()

        print(f"Connecting sensor on {self.SENSOR_PORT}...")
        self.sensor_serial = serial.Serial(
            self.SENSOR_PORT, self.SENSOR_BAUDRATE, timeout=0.01
        )
        self.sensor_serial.reset_input_buffer()

        # Wait for IMU to stabilize
        print("Waiting for IMU to stabilize (2 seconds)...")
        time.sleep(2.0)
        self.median_filter.reset()
        print("Hardware connected successfully.")

    def disconnect(self):
        """Safely disconnect hardware."""
        if self.motor_controller:
            try:
                self.motor_controller.set_goal_velocity_rad([0.0, 0.0])
                self.motor_controller.close()
            except:
                pass
        if self.sensor_serial:
            try:
                self.sensor_serial.close()
            except:
                pass
        print("Hardware disconnected.")

    def read_yaw_imu(self):
        """Read and filter yaw from IMU. Returns (raw_yaw, filtered_yaw) or (None, None)."""
        raw_yaw = None
        if self.sensor_serial and self.sensor_serial.in_waiting:
            try:
                lines = (
                    self.sensor_serial.read_all().decode(errors="ignore").splitlines()
                )
                # Search last valid numeric yaw in the batch
                for i in range(len(lines) - 1, -1, -1):
                    parts = lines[i].split(",")
                    if len(parts) >= 9:
                        token = parts[8].strip()
                        val = self._safe_float(token)
                        if val is not None:
                            raw_yaw = val
                            break
            except Exception:
                # Swallow transient serial parse errors
                return None, None

        if raw_yaw is not None:
            # Apply unwrap and median filter
            yaw_unwrapped = self._unwrap_90_absolute(raw_yaw)
            yaw_filtered = self.median_filter.update(yaw_unwrapped)
            return raw_yaw, yaw_filtered

        return None, None

    def _unwrap_90_absolute(self, val):
        """Forces value into [-90, 90] range."""
        while val > 90:
            val -= 90
        while val < -90:
            val += 90
        return val

    def set_velocity(self, u):
        """Set motor velocity command (rad/s). Applies safety clipping."""
        u_clipped = np.clip(u, -self.MAX_U, self.MAX_U)
        try:
            # Opposed mode: motors rotate in opposite directions
            self.motor_controller.set_goal_velocity_rad([u_clipped, -u_clipped])
        except Exception as e:
            print(f"Motor command error: {e}")

    def read_motor_load(self):
        """Read current motor load/current. Returns list of loads for each motor."""
        try:
            # Read telemetry using read_info_with_unit to get load
            _, _, _, load = self.motor_controller.read_info_with_unit(
                angle_unit="rad", current_unit="mA", fast_read=False, retry=True
            )
            if load is not None and len(load) >= 2:
                return [float(load[0]), float(load[1])]
        except Exception:
            pass
        return [0.0, 0.0]

    def read_motor_effort(self):
        """Read motor electrical effort (current in mA). Returns list [m1, m2]."""
        try:
            # Use read_info_with_unit which handles the conversion automatically
            _, _, current, _ = self.motor_controller.read_info_with_unit(
                angle_unit="rad", current_unit="mA", fast_read=False, retry=True
            )
            # print(
            #     f"DEBUG read_motor_effort: current = {current}, type = {type(current)}"
            # )
            if current is not None and len(current) >= 2:
                result = [float(current[0]), float(current[1])]
                # print(f"DEBUG: Current converted: {result}")
                return result
            else:
                pass
                # print(f"DEBUG: current is None or too short")
        except Exception as e:
            pass
            # print(f"DEBUG: read_motor_effort exception: {e}")
            # import traceback
            # traceback.print_exc()
        return [0.0, 0.0]

    def _safe_float(self, token):
        """Parse token to float, returning None if invalid (handles '-', '', NaN)."""
        if token is None:
            return None
        t = token.strip()
        if t == "" or t == "-":
            return None
        try:
            val = float(t)
            if np.isnan(val):
                return None
            return val
        except Exception:
            return None

    def control_to_target(self, target_yaw, current_yaw):
        """Compute P-control velocity command."""
        error = target_yaw - current_yaw
        u = self.K_GAIN * error
        return u


class SteadyStateDetector:
    """Detects when system has reached steady state."""

    def __init__(
        self, tolerance_deg=0.5, required_duration_sec=1.0, sample_rate_hz=50.0
    ):
        self.tolerance = tolerance_deg
        self.required_duration = required_duration_sec
        self.sample_rate = sample_rate_hz
        self.required_samples = int(required_duration_sec * sample_rate_hz)

        self.stable_count = 0

    def update(self, error):
        """Update detector with current error. Returns True if steady state reached."""
        if abs(error) <= self.tolerance:
            self.stable_count += 1
        else:
            self.stable_count = 0

        return self.stable_count >= self.required_samples

    def reset(self):
        """Reset the detector."""
        self.stable_count = 0


class PerformanceTests:
    """Automated performance validation tests."""

    def __init__(self, robot_controller):
        self.robot = robot_controller
        self.results = []

    def reset_to_zero_gui(
        self,
        tolerance=0.5,
        timeout=12.0,
        data_callback=None,
        status_callback=None,
        stop_check=None,
    ):
        """Drive the robot back to yaw=0 with logging and callbacks."""

        def log(msg):
            if status_callback:
                status_callback(msg)
            else:
                print(msg)

        log("Moving to 0° setpoint...")

        start_time = time.perf_counter()
        dt = 0.02
        next_wake = start_time
        stable_count = 0
        required_stable = 25  # ~0.5s at 50Hz

        log_data = []

        while time.perf_counter() - start_time < timeout:
            if stop_check and stop_check():
                log("Reset stopped by user")
                break

            loop_start = time.perf_counter()
            elapsed = loop_start - start_time

            raw_yaw, filtered_yaw = self.robot.read_yaw_imu()

            if filtered_yaw is not None:
                target = 0.0
                error = target - filtered_yaw
                if abs(error) <= tolerance:
                    stable_count += 1
                else:
                    stable_count = 0

                u = self.robot.control_to_target(target, filtered_yaw)
                self.robot.set_velocity(u)

                loads = self.robot.read_motor_load()
                currents = self.robot.read_motor_effort()
                data_point = {
                    "timestamp": elapsed,
                    "target_yaw": target,
                    "measured_yaw_imu": filtered_yaw,
                    "measured_yaw_raw": raw_yaw,
                    "motor_velocity": u,
                    "motor_load_1": loads[0] if len(loads) > 0 else 0.0,
                    "motor_load_2": loads[1] if len(loads) > 1 else 0.0,
                    "motor_current_1": currents[0] if len(currents) > 0 else 0.0,
                    "motor_current_2": currents[1] if len(currents) > 1 else 0.0,
                }
                log_data.append(data_point)
                if data_callback:
                    data_callback(data_point)

                if stable_count >= required_stable:
                    log("Reached 0° setpoint.")
                    break

            next_wake += dt
            sleep_time = next_wake - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Stop motors at the end
        self.robot.set_velocity(0.0)

        # Final error metric
        final_error = 0.0
        if len(log_data) > 0:
            final_error = 0.0 - log_data[-1]["measured_yaw_imu"]

        return {
            "test_type": "reset_to_zero",
            "data": log_data,
            "metrics": {"final_error": final_error},
            "parameters": {"tolerance": tolerance, "timeout": timeout},
        }

    def test_step_response_gui(
        self,
        initial_yaw=0.0,
        target_yaw=20.0,
        max_duration=15.0,
        data_callback=None,
        status_callback=None,
        stop_check=None,
    ):
        """
        Step Response Test with GUI callbacks.

        Args:
            initial_yaw: Starting position (degrees)
            target_yaw: Target position (degrees)
            max_duration: Maximum test duration (seconds)
            data_callback: Function to call with each data point
            status_callback: Function to call with status messages
            stop_check: Function that returns True if test should stop

        Returns:
            dict: Test results including metrics and logged data
        """

        def log(msg):
            if status_callback:
                status_callback(msg)
            else:
                print(msg)

        log(f"Moving to initial position {initial_yaw}°...")
        self._move_to_position(
            initial_yaw, timeout=10.0, status_callback=log, stop_check=stop_check
        )

        if stop_check and stop_check():
            return None

        # Reset filter and wait briefly
        self.robot.median_filter.reset()
        time.sleep(0.5)

        log(f"Executing step to {target_yaw}°...")

        steady_detector = SteadyStateDetector(
            tolerance_deg=0.5, required_duration_sec=1.0, sample_rate_hz=50.0
        )

        log_data = []
        start_time = time.perf_counter()
        step_executed = False
        step_time = None

        # Sample rate ~50Hz
        dt = 0.02
        next_wake = start_time

        while True:
            if stop_check and stop_check():
                log("Test stopped by user")
                break

            loop_start = time.perf_counter()
            elapsed = loop_start - start_time

            if elapsed > max_duration:
                log("Maximum test duration reached.")
                break

            # Read sensor
            raw_yaw, filtered_yaw = self.robot.read_yaw_imu()

            if filtered_yaw is not None:
                # Execute step command at t=0
                if not step_executed:
                    current_target = target_yaw
                    step_executed = True
                    step_time = elapsed
                    log(f"Step command issued at t={step_time:.3f}s")
                else:
                    current_target = target_yaw

                # Compute control
                u = self.robot.control_to_target(current_target, filtered_yaw)
                self.robot.set_velocity(u)

                # Read motor effort (current mA) and load if available
                motor_currents = self.robot.read_motor_effort()
                motor_loads = self.robot.read_motor_load()

                # Log data
                data_point = {
                    "timestamp": elapsed,
                    "target_yaw": current_target,
                    "measured_yaw_imu": filtered_yaw,
                    "measured_yaw_raw": raw_yaw,
                    "motor_velocity": u,
                    "motor_current_1": (
                        motor_currents[0] if len(motor_currents) > 0 else 0.0
                    ),
                    "motor_current_2": (
                        motor_currents[1] if len(motor_currents) > 1 else 0.0
                    ),
                    "motor_load_1": motor_loads[0] if len(motor_loads) > 0 else 0.0,
                    "motor_load_2": motor_loads[1] if len(motor_loads) > 1 else 0.0,
                }
                log_data.append(data_point)

                if data_callback:
                    data_callback(data_point)

                # Check steady state
                error = current_target - filtered_yaw
                if step_executed and steady_detector.update(error):
                    log(f"Steady state reached at t={elapsed:.3f}s")
                    # Continue logging for a bit more to confirm
                    if elapsed - step_time > 3.0:
                        break

            # Timing
            next_wake += dt
            sleep_time = next_wake - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Stop motors
        self.robot.set_velocity(0.0)

        # Analyze results
        log("Analyzing step response...")
        metrics = self._analyze_step_response(log_data, initial_yaw, target_yaw)

        return {
            "test_type": "step_response",
            "data": log_data,
            "metrics": metrics,
            "parameters": {"initial_yaw": initial_yaw, "target_yaw": target_yaw},
        }

    def test_disturbance_rejection_gui(
        self,
        setpoint=-45.0,
        duration=20.0,
        data_callback=None,
        status_callback=None,
        stop_check=None,
    ):
        """
        Active Stiffness & Disturbance Rejection Test with GUI callbacks.

        Args:
            setpoint: Target angle to maintain (degrees)
            duration: Test duration (seconds)
            data_callback: Function to call with each data point
            status_callback: Function to call with status messages
            stop_check: Function that returns True if test should stop

        Returns:
            dict: Test results with logged data
        """

        def log(msg):
            if status_callback:
                status_callback(msg)
            else:
                print(msg)

        # Move to setpoint first
        log(f"Moving to setpoint {setpoint}°...")
        self._move_to_position(
            setpoint, timeout=10.0, status_callback=log, stop_check=stop_check
        )

        if stop_check and stop_check():
            return None

        log("Test starting in 3 seconds...")
        log("Please prepare to apply manual disturbances (add mass, push, etc.)")
        time.sleep(3.0)

        log("Test started! Apply disturbances now...")

        log_data = []
        start_time = time.perf_counter()

        # High frequency logging
        dt = 0.01  # ~100Hz for high-freq capture
        next_wake = start_time

        last_log_time = 0

        while True:
            if stop_check and stop_check():
                log("Test stopped by user")
                break

            loop_start = time.perf_counter()
            elapsed = loop_start - start_time

            if elapsed > duration:
                break

            # Read sensor
            raw_yaw, filtered_yaw = self.robot.read_yaw_imu()

            if filtered_yaw is not None:
                # Control to maintain setpoint
                u = self.robot.control_to_target(setpoint, filtered_yaw)
                self.robot.set_velocity(u)

                # Read motor effort (current mA) and load if available
                motor_currents = self.robot.read_motor_effort()
                motor_loads = self.robot.read_motor_load()

                # Log data
                data_point = {
                    "timestamp": elapsed,
                    "target_yaw": setpoint,
                    "measured_yaw_imu": filtered_yaw,
                    "measured_yaw_raw": raw_yaw,
                    "motor_velocity": u,
                    "motor_current_1": (
                        motor_currents[0] if len(motor_currents) > 0 else 0.0
                    ),
                    "motor_current_2": (
                        motor_currents[1] if len(motor_currents) > 1 else 0.0
                    ),
                    "motor_load_1": motor_loads[0] if len(motor_loads) > 0 else 0.0,
                    "motor_load_2": motor_loads[1] if len(motor_loads) > 1 else 0.0,
                    "error": setpoint - filtered_yaw,
                }
                log_data.append(data_point)

                if data_callback:
                    data_callback(data_point)

                # Print progress every 2 seconds
                if elapsed - last_log_time >= 2.0:
                    log(
                        f"t={elapsed:.1f}s: Yaw={filtered_yaw:.2f}°, Error={setpoint-filtered_yaw:.2f}°"
                    )
                    last_log_time = elapsed

            # Timing
            next_wake += dt
            sleep_time = next_wake - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Stop motors
        self.robot.set_velocity(0.0)

        log("Test completed!")

        # Compute basic statistics
        if len(log_data) > 0:
            df = pd.DataFrame(log_data)
            metrics = {
                "mean_error": df["error"].abs().mean(),
                "max_error": df["error"].abs().max(),
                "std_error": df["error"].std(),
                "max_control_effort": df["motor_velocity"].abs().max(),
                "mean_control_effort": df["motor_velocity"].abs().mean(),
            }
        else:
            metrics = {
                "mean_error": 0.0,
                "max_error": 0.0,
                "std_error": 0.0,
                "max_control_effort": 0.0,
                "mean_control_effort": 0.0,
            }

        return {
            "test_type": "disturbance_rejection",
            "data": log_data,
            "metrics": metrics,
            "parameters": {"setpoint": setpoint, "duration": duration},
        }

    def test_step_response(self, initial_yaw=0.0, target_yaw=20.0, max_duration=15.0):
        """
        Step Response Test: Measure system response to step input.

        Args:
            initial_yaw: Starting position (degrees)
            target_yaw: Target position (degrees)
            max_duration: Maximum test duration (seconds)

        Returns:
            dict: Test results including metrics and logged data
        """
        print("\n" + "=" * 60)
        print("STEP RESPONSE TEST")
        print("=" * 60)
        print(f"Initial: {initial_yaw}°, Target: {target_yaw}°")

        # Phase 1: Move to initial position
        print(f"\nPhase 1: Moving to initial position {initial_yaw}°...")
        self._move_to_position(initial_yaw, timeout=10.0)

        # Reset filter and wait briefly
        self.robot.median_filter.reset()
        time.sleep(0.5)

        # Phase 2: Execute step and log data
        print(f"\nPhase 2: Executing step to {target_yaw}°...")

        steady_detector = SteadyStateDetector(
            tolerance_deg=0.5, required_duration_sec=1.0, sample_rate_hz=50.0
        )

        log_data = []
        start_time = time.perf_counter()
        step_executed = False
        step_time = None

        # Sample rate ~50Hz
        dt = 0.02
        next_wake = start_time

        while True:
            loop_start = time.perf_counter()
            elapsed = loop_start - start_time

            if elapsed > max_duration:
                print("Maximum test duration reached.")
                break

            # Read sensor
            raw_yaw, filtered_yaw = self.robot.read_yaw_imu()

            if filtered_yaw is not None:
                # Execute step command at t=0
                if not step_executed:
                    current_target = target_yaw
                    step_executed = True
                    step_time = elapsed
                    print(f"Step command issued at t={step_time:.3f}s")
                else:
                    current_target = target_yaw

                # Compute control
                u = self.robot.control_to_target(current_target, filtered_yaw)
                self.robot.set_velocity(u)

                # Read motor load
                motor_loads = self.robot.read_motor_load()

                # Log data
                log_data.append(
                    {
                        "timestamp": elapsed,
                        "target_yaw": current_target,
                        "measured_yaw_imu": filtered_yaw,
                        "measured_yaw_raw": raw_yaw,
                        "motor_velocity": u,
                        "motor_load_1": motor_loads[0] if len(motor_loads) > 0 else 0.0,
                        "motor_load_2": motor_loads[1] if len(motor_loads) > 1 else 0.0,
                    }
                )

                # Check steady state
                error = current_target - filtered_yaw
                if step_executed and steady_detector.update(error):
                    print(f"Steady state reached at t={elapsed:.3f}s")
                    # Continue logging for a bit more to confirm
                    if elapsed - step_time > 3.0:
                        break

            # Timing
            next_wake += dt
            sleep_time = next_wake - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Stop motors
        self.robot.set_velocity(0.0)

        # Analyze results
        print("\nAnalyzing step response...")
        metrics = self._analyze_step_response(log_data, initial_yaw, target_yaw)

        # Print metrics
        print("\n" + "-" * 40)
        print("STEP RESPONSE METRICS:")
        print(f"  Rise Time (t_r):       {metrics['rise_time']:.3f} s")
        print(f"  Settling Time (t_s):   {metrics['settling_time']:.3f} s")
        print(f"  Steady-State Error:    {metrics['steady_state_error']:.3f}°")
        print(f"  Overshoot:             {metrics['overshoot']:.2f}%")
        print(f"  Peak Value:            {metrics['peak_value']:.3f}°")
        print("-" * 40)

        return {
            "test_type": "step_response",
            "data": log_data,
            "metrics": metrics,
            "parameters": {"initial_yaw": initial_yaw, "target_yaw": target_yaw},
        }

    def test_disturbance_rejection(self, setpoint=-45.0, duration=20.0):
        """
        Active Stiffness & Disturbance Rejection Test.

        Hold at setpoint while user manually applies disturbances.

        Args:
            setpoint: Target angle to maintain (degrees)
            duration: Test duration (seconds)

        Returns:
            dict: Test results with logged data
        """
        print("\n" + "=" * 60)
        print("DISTURBANCE REJECTION TEST")
        print("=" * 60)
        print(f"Setpoint: {setpoint}°, Duration: {duration}s")

        # Move to setpoint first
        print(f"\nMoving to setpoint {setpoint}°...")
        self._move_to_position(setpoint, timeout=10.0)

        print("\n" + "!" * 60)
        print("TEST STARTING IN 3 SECONDS...")
        print("Please prepare to apply manual disturbances (add mass, push, etc.)")
        print("!" * 60)
        time.sleep(3.0)

        print("\nTest started! Apply disturbances now...")

        log_data = []
        start_time = time.perf_counter()

        # High frequency logging
        dt = 0.01  # ~100Hz for high-freq capture
        next_wake = start_time

        while True:
            loop_start = time.perf_counter()
            elapsed = loop_start - start_time

            if elapsed > duration:
                break

            # Read sensor
            raw_yaw, filtered_yaw = self.robot.read_yaw_imu()

            if filtered_yaw is not None:
                # Control to maintain setpoint
                u = self.robot.control_to_target(setpoint, filtered_yaw)
                self.robot.set_velocity(u)

                # Read motor load and current (control effort proxy)
                motor_loads = self.robot.read_motor_load()
                motor_currents = self.robot.read_motor_effort()

                # Log data
                log_data.append(
                    {
                        "timestamp": elapsed,
                        "target_yaw": setpoint,
                        "measured_yaw_imu": filtered_yaw,
                        "measured_yaw_raw": raw_yaw,
                        "motor_velocity": u,
                        "motor_load_1": motor_loads[0] if len(motor_loads) > 0 else 0.0,
                        "motor_load_2": motor_loads[1] if len(motor_loads) > 1 else 0.0,
                        "motor_current_1": (
                            motor_currents[0] if len(motor_currents) > 0 else 0.0
                        ),
                        "motor_current_2": (
                            motor_currents[1] if len(motor_currents) > 1 else 0.0
                        ),
                        "error": setpoint - filtered_yaw,
                    }
                )

                # Print progress every 2 seconds
                if int(elapsed) % 2 == 0 and elapsed - int(elapsed) < dt:
                    print(
                        f"  t={elapsed:.1f}s: Yaw={filtered_yaw:.2f}°, Error={setpoint-filtered_yaw:.2f}°"
                    )

            # Timing
            next_wake += dt
            sleep_time = next_wake - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Stop motors
        self.robot.set_velocity(0.0)

        print("\nTest completed!")

        # Compute basic statistics
        df = pd.DataFrame(log_data)
        control_effort = df[["motor_current_1", "motor_current_2"]].abs().max(axis=1)
        metrics = {
            "mean_error": df["error"].abs().mean(),
            "max_error": df["error"].abs().max(),
            "std_error": df["error"].std(),
            "max_control_effort": control_effort.max(),  # mA
            "mean_control_effort": control_effort.mean(),  # mA
        }

        print("\n" + "-" * 40)
        print("DISTURBANCE REJECTION METRICS:")
        print(f"  Mean Absolute Error:   {metrics['mean_error']:.3f}°")
        print(f"  Max Absolute Error:    {metrics['max_error']:.3f}°")
        print(f"  Std Dev Error:         {metrics['std_error']:.3f}°")
        print(f"  Max Control Effort:    {metrics['max_control_effort']:.3f} mA")
        print(f"  Mean Control Effort:   {metrics['mean_control_effort']:.3f} mA")
        print("-" * 40)

        return {
            "test_type": "disturbance_rejection",
            "data": log_data,
            "metrics": metrics,
            "parameters": {"setpoint": setpoint, "duration": duration},
        }

    def _move_to_position(
        self, target, timeout=10.0, tolerance=0.5, status_callback=None, stop_check=None
    ):
        """Helper to move robot to a target position."""

        def log(msg):
            if status_callback:
                status_callback(msg)
            else:
                print(msg)

        start_time = time.perf_counter()
        stable_count = 0
        required_stable = 25  # ~0.5 seconds at 50Hz

        dt = 0.02
        next_wake = start_time

        while time.perf_counter() - start_time < timeout:
            if stop_check and stop_check():
                return False

            loop_start = time.perf_counter()

            raw_yaw, filtered_yaw = self.robot.read_yaw_imu()

            if filtered_yaw is not None:
                error = target - filtered_yaw

                if abs(error) < tolerance:
                    stable_count += 1
                    if stable_count >= required_stable:
                        log(f"Reached {target}° (error: {error:.2f}°)")
                        return True
                else:
                    stable_count = 0

                u = self.robot.control_to_target(target, filtered_yaw)
                self.robot.set_velocity(u)

            next_wake += dt
            sleep_time = next_wake - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

        log(f"Warning: Timeout reaching {target}°")
        return False

    def _analyze_step_response(self, log_data, initial, target):
        """Analyze step response data and compute metrics."""
        df = pd.DataFrame(log_data)

        if len(df) == 0:
            return {
                "rise_time": 0.0,
                "settling_time": 0.0,
                "steady_state_error": 0.0,
                "overshoot": 0.0,
                "peak_value": 0.0,
            }

        # Find when step was executed (when target changes)
        step_idx = 0
        for i in range(1, len(df)):
            if df.loc[i, "target_yaw"] != df.loc[i - 1, "target_yaw"]:
                step_idx = i
                break

        if step_idx == 0:
            step_idx = 0  # No explicit step found, assume from start

        # Extract response data after step
        response_df = df.iloc[step_idx:].copy()
        response_df["time_from_step"] = (
            response_df["timestamp"] - response_df.iloc[0]["timestamp"]
        )

        step_size = target - initial

        # Rise Time: Time to go from 10% to 90% of step
        ten_percent = initial + 0.1 * step_size
        ninety_percent = initial + 0.9 * step_size

        rise_start_idx = None
        rise_end_idx = None

        for i in range(len(response_df)):
            val = response_df.iloc[i]["measured_yaw_imu"]
            if rise_start_idx is None and (
                (step_size > 0 and val >= ten_percent)
                or (step_size < 0 and val <= ten_percent)
            ):
                rise_start_idx = i
            if rise_end_idx is None and (
                (step_size > 0 and val >= ninety_percent)
                or (step_size < 0 and val <= ninety_percent)
            ):
                rise_end_idx = i
                break

        if rise_start_idx is not None and rise_end_idx is not None:
            rise_time = (
                response_df.iloc[rise_end_idx]["time_from_step"]
                - response_df.iloc[rise_start_idx]["time_from_step"]
            )
        else:
            rise_time = 0.0

        # Settling Time: Time to stay within ±2% of final value
        settling_band = 0.02 * abs(step_size)
        if settling_band < 0.5:  # Minimum 0.5 degree band
            settling_band = 0.5

        # Use last 20% of data to estimate final value
        final_portion = response_df.iloc[int(0.8 * len(response_df)) :]
        final_value = final_portion["measured_yaw_imu"].mean()

        settling_idx = None
        for i in range(len(response_df) - 1, -1, -1):
            val = response_df.iloc[i]["measured_yaw_imu"]
            if abs(val - final_value) > settling_band:
                settling_idx = i + 1
                break

        if settling_idx is not None and settling_idx < len(response_df):
            settling_time = response_df.iloc[settling_idx]["time_from_step"]
        else:
            settling_time = response_df.iloc[-1]["time_from_step"]

        # Steady-State Error
        steady_state_error = target - final_value

        # Overshoot
        if step_size > 0:
            peak_value = response_df["measured_yaw_imu"].max()
            overshoot = max(0, (peak_value - target) / step_size * 100)
        else:
            peak_value = response_df["measured_yaw_imu"].min()
            overshoot = max(0, (target - peak_value) / abs(step_size) * 100)

        return {
            "rise_time": rise_time,
            "settling_time": settling_time,
            "steady_state_error": steady_state_error,
            "overshoot": overshoot,
            "peak_value": peak_value,
            "final_value": final_value,
        }

    def save_results(self, test_results, filename_prefix="performance_test"):
        """Save test results to CSV files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, result in enumerate(test_results):
            test_type = result["test_type"]
            filename = f"RESULTAT/{filename_prefix}_{test_type}_{timestamp}.csv"

            df = pd.DataFrame(result["data"])
            df.to_csv(filename, index=False)
            print(f"\nSaved {test_type} data to: {filename}")

            # Also save metrics to a separate file
            metrics_filename = (
                f"RESULTAT/{filename_prefix}_{test_type}_metrics_{timestamp}.txt"
            )
            with open(metrics_filename, "w") as f:
                f.write(f"Test Type: {test_type}\n")
                f.write(f"Date: {timestamp}\n")
                f.write("\nParameters:\n")
                for key, val in result["parameters"].items():
                    f.write(f"  {key}: {val}\n")
                f.write("\nMetrics:\n")
                for key, val in result["metrics"].items():
                    f.write(f"  {key}: {val}\n")
            print(f"Saved {test_type} metrics to: {metrics_filename}")


def main():
    """Main test execution with GUI."""
    root = tk.Tk()
    app = PerformanceTestGUI(root)
    root.mainloop()


class PerformanceTestGUI:
    """GUI for robot performance tests."""

    def __init__(self, root):
        self.root = root
        self.root.title("Robot Performance Tests")
        self.root.geometry("1400x900")

        # Hardware config
        self.MOTOR_PORT = "COM8"
        self.SENSOR_PORT = "COM3"
        self.BAUDRATE_MOTOR = 57600
        self.BAUDRATE_SENSOR = 115200

        # State
        self.is_connected = False
        self.test_running = False
        self.stop_requested = False

        # Hardware objects
        self.robot = None
        self.tests = None

        # Test results storage
        self.current_test_data = []
        self.all_results = []

        # Create UI
        self.create_widgets()

        # Protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # 1. Connection Panel
        conn_frame = ttk.LabelFrame(self.root, text="Hardware Connection", padding="10")
        conn_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(conn_frame, text=f"Motor Port: {self.MOTOR_PORT}").pack(
            side="left", padx=10
        )
        ttk.Label(conn_frame, text=f"Sensor Port: {self.SENSOR_PORT}").pack(
            side="left", padx=10
        )

        # Motor Model / Current Scale selector
        ttk.Label(conn_frame, text="Model:").pack(side="left", padx=10)
        self.combo_model = ttk.Combobox(
            conn_frame,
            state="readonly",
            values=["X-series (2.69 mA/LSB)", "XC330 (3.36 mA/LSB)"],
        )
        self.combo_model.set("XC330 (3.36 mA/LSB)")
        self.combo_model.pack(side="left")

        self.btn_connect = ttk.Button(
            conn_frame, text="Connect Hardware", command=self.toggle_connection
        )
        self.btn_connect.pack(side="left", padx=20)

        self.led_status = ttk.Label(
            conn_frame,
            text="DISCONNECTED",
            foreground="red",
            font=("Arial", 10, "bold"),
        )
        self.led_status.pack(side="left", padx=10)

        # 2. Test Selection Panel
        test_frame = ttk.LabelFrame(
            self.root, text="Test Selection & Parameters", padding="10"
        )
        test_frame.pack(fill="x", padx=10, pady=5)

        # Step Response Test
        step_frame = ttk.LabelFrame(
            test_frame, text="1. Step Response Test", padding="10"
        )
        step_frame.pack(fill="x", padx=5, pady=5)

        step_controls = ttk.Frame(step_frame)
        step_controls.pack(fill="x")

        ttk.Label(step_controls, text="Initial Yaw (°):").grid(
            row=0, column=0, padx=5, pady=2, sticky="w"
        )
        self.entry_step_initial = ttk.Entry(step_controls, width=10)
        self.entry_step_initial.insert(0, "0.0")
        self.entry_step_initial.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(step_controls, text="Target Yaw (°):").grid(
            row=0, column=2, padx=5, pady=2, sticky="w"
        )
        self.entry_step_target = ttk.Entry(step_controls, width=10)
        self.entry_step_target.insert(0, "20.0")
        self.entry_step_target.grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(step_controls, text="Max Duration (s):").grid(
            row=0, column=4, padx=5, pady=2, sticky="w"
        )
        self.entry_step_duration = ttk.Entry(step_controls, width=10)
        self.entry_step_duration.insert(0, "15.0")
        self.entry_step_duration.grid(row=0, column=5, padx=5, pady=2)

        self.btn_run_step = ttk.Button(
            step_controls,
            text="Run Step Test",
            command=lambda: self.run_test("step"),
            state="disabled",
        )
        self.btn_run_step.grid(row=0, column=6, padx=20, pady=2)

        # Disturbance Rejection Test
        dist_frame = ttk.LabelFrame(
            test_frame, text="2. Disturbance Rejection Test", padding="10"
        )
        dist_frame.pack(fill="x", padx=5, pady=5)

        dist_controls = ttk.Frame(dist_frame)
        dist_controls.pack(fill="x")

        ttk.Label(dist_controls, text="Setpoint (°):").grid(
            row=0, column=0, padx=5, pady=2, sticky="w"
        )
        self.entry_dist_setpoint = ttk.Entry(dist_controls, width=10)
        self.entry_dist_setpoint.insert(0, "-45.0")
        self.entry_dist_setpoint.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(dist_controls, text="Duration (s):").grid(
            row=0, column=2, padx=5, pady=2, sticky="w"
        )
        self.entry_dist_duration = ttk.Entry(dist_controls, width=10)
        self.entry_dist_duration.insert(0, "20.0")
        self.entry_dist_duration.grid(row=0, column=3, padx=5, pady=2)

        self.btn_run_dist = ttk.Button(
            dist_controls,
            text="Run Disturbance Test",
            command=lambda: self.run_test("disturbance"),
            state="disabled",
        )
        self.btn_run_dist.grid(row=0, column=4, padx=20, pady=2)

        # Reset button to return to yaw=0
        self.btn_reset = ttk.Button(
            test_frame, text="RESET TO 0°", command=self.reset_to_zero, state="disabled"
        )
        self.btn_reset.pack(pady=5)

        # Stop button
        self.btn_stop_test = ttk.Button(
            test_frame, text="STOP TEST", command=self.stop_test, state="disabled"
        )
        self.btn_stop_test.pack(pady=10)

        # 3. Status & Results Panel
        results_frame = ttk.LabelFrame(
            self.root, text="Test Status & Results", padding="10"
        )
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Status text area
        status_label = ttk.Label(results_frame, text="Status Log:")
        status_label.pack(anchor="w")

        self.status_text = scrolledtext.ScrolledText(
            results_frame, height=10, width=80, state="disabled", wrap="word"
        )
        self.status_text.pack(fill="both", expand=True, pady=5)

        # Progress bar
        self.progress = ttk.Progressbar(results_frame, mode="indeterminate")
        self.progress.pack(fill="x", pady=5)

        # Save results button
        self.btn_save = ttk.Button(
            results_frame,
            text="Save All Results to CSV",
            command=self.save_all_results,
            state="disabled",
        )
        self.btn_save.pack(pady=5)

        # 4. Real-time Plot
        plot_frame = ttk.LabelFrame(self.root, text="Real-Time Data", padding="10")
        plot_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.fig, (self.ax_yaw, self.ax_u, self.ax_effort) = plt.subplots(
            3, 1, figsize=(10, 8), sharex=True
        )
        self.fig.tight_layout(pad=3.0)

        (self.line_yaw,) = self.ax_yaw.plot(
            [], [], "b-", linewidth=2, label="Measured Yaw"
        )
        (self.line_target,) = self.ax_yaw.plot(
            [], [], "r--", linewidth=1.5, label="Target Yaw"
        )
        self.ax_yaw.set_ylabel("Yaw [deg]")
        self.ax_yaw.legend()
        self.ax_yaw.grid(True)
        self.ax_yaw.set_ylim(-60, 60)

        (self.line_u,) = self.ax_u.plot(
            [], [], "g-", linewidth=1.5, label="Control Signal"
        )
        self.ax_u.set_ylabel("Velocity [rad/s]")
        self.ax_u.set_xlabel("Time [s]")
        self.ax_u.legend()
        self.ax_u.grid(True)
        self.ax_u.set_ylim(-6, 6)

        # Motor effort subplot (current mA)
        (self.line_cur1,) = self.ax_effort.plot(
            [], [], "m-", linewidth=1.2, label="Motor Current 1 [mA]"
        )
        (self.line_cur2,) = self.ax_effort.plot(
            [], [], "c-", linewidth=1.2, label="Motor Current 2 [mA]"
        )
        self.ax_effort.set_ylabel("Current [mA]")
        self.ax_effort.set_xlabel("Time [s]")
        self.ax_effort.legend()
        self.ax_effort.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Start animation loop
        self.root.after(100, self.update_plot)

    def log_status(self, message):
        """Add message to status log."""
        self.status_text.config(state="normal")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert("end", f"[{timestamp}] {message}\n")
        self.status_text.see("end")
        self.status_text.config(state="disabled")

    def toggle_connection(self):
        if not self.is_connected:
            self.connect_hardware()
        else:
            self.disconnect_hardware()

    def connect_hardware(self):
        try:
            self.log_status("Connecting to hardware...")
            self.robot = RobotController(
                motor_port=self.MOTOR_PORT,
                sensor_port=self.SENSOR_PORT,
                motor_baudrate=self.BAUDRATE_MOTOR,
                sensor_baudrate=self.BAUDRATE_SENSOR,
            )

            # Redirect print to log
            import io
            import contextlib

            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                self.robot.connect()

            self.log_status("Hardware connected successfully.")

            self.tests = PerformanceTests(self.robot)

            # Apply current scale based on selected model
            model_sel = self.combo_model.get()
            if "XC330" in model_sel:
                self.robot.set_current_scale(3.36)
            else:
                self.robot.set_current_scale(2.69)

            self.is_connected = True
            self.led_status.config(text="CONNECTED", foreground="green")
            self.btn_connect.config(text="Disconnect")
            self.btn_run_step.config(state="normal")
            self.btn_run_dist.config(state="normal")
            self.btn_reset.config(state="normal")

        except Exception as e:
            self.log_status(f"Connection error: {e}")
            messagebox.showerror("Connection Error", str(e))
            self.disconnect_hardware()

    def disconnect_hardware(self):
        if self.robot:
            self.robot.disconnect()

        self.is_connected = False
        self.led_status.config(text="DISCONNECTED", foreground="red")
        self.btn_connect.config(text="Connect Hardware")
        self.btn_run_step.config(state="disabled")
        self.btn_run_dist.config(state="disabled")
        self.btn_reset.config(state="disabled")
        self.btn_stop_test.config(state="disabled")

        self.log_status("Hardware disconnected.")

    def run_test(self, test_type):
        if not self.is_connected or self.test_running:
            return

        self.test_running = True
        self.stop_requested = False
        self.current_test_data = []

        # Disable buttons
        self.btn_run_step.config(state="disabled")
        self.btn_run_dist.config(state="disabled")
        self.btn_reset.config(state="disabled")
        self.btn_stop_test.config(state="normal")
        self.btn_connect.config(state="disabled")

        # Start progress animation
        self.progress.start(10)

        # Run test in separate thread
        thread = threading.Thread(
            target=self._run_test_thread, args=(test_type,), daemon=True
        )
        thread.start()

    def _run_test_thread(self, test_type):
        try:
            if test_type == "step":
                initial = float(self.entry_step_initial.get())
                target = float(self.entry_step_target.get())
                duration = float(self.entry_step_duration.get())

                self.log_status(f"Starting Step Response Test: {initial}° → {target}°")

                result = self.tests.test_step_response_gui(
                    initial_yaw=initial,
                    target_yaw=target,
                    max_duration=duration,
                    data_callback=self.update_test_data,
                    status_callback=self.log_status,
                    stop_check=lambda: self.stop_requested,
                )

                if result:
                    self.all_results.append(result)
                    self.log_status("Step Response Test completed!")
                    self.log_status(
                        f"  Rise Time: {result['metrics']['rise_time']:.3f} s"
                    )
                    self.log_status(
                        f"  Settling Time: {result['metrics']['settling_time']:.3f} s"
                    )
                    self.log_status(
                        f"  Steady-State Error: {result['metrics']['steady_state_error']:.3f}°"
                    )
                    self.log_status(
                        f"  Overshoot: {result['metrics']['overshoot']:.2f}%"
                    )

            elif test_type == "disturbance":
                setpoint = float(self.entry_dist_setpoint.get())
                duration = float(self.entry_dist_duration.get())

                self.log_status(f"Starting Disturbance Rejection Test at {setpoint}°")
                self.log_status("Please prepare to apply manual disturbances...")

                result = self.tests.test_disturbance_rejection_gui(
                    setpoint=setpoint,
                    duration=duration,
                    data_callback=self.update_test_data,
                    status_callback=self.log_status,
                    stop_check=lambda: self.stop_requested,
                )

                if result:
                    self.all_results.append(result)
                    self.log_status("Disturbance Rejection Test completed!")
                    self.log_status(
                        f"  Mean Abs Error: {result['metrics']['mean_error']:.3f}°"
                    )
                    self.log_status(
                        f"  Max Abs Error: {result['metrics']['max_error']:.3f}°"
                    )
                    self.log_status(
                        f"  Max Control Effort: {result['metrics']['max_control_effort']:.3f} mA"
                    )

            if len(self.all_results) > 0:
                self.root.after(0, lambda: self.btn_save.config(state="normal"))

        except Exception as e:
            self.log_status(f"Test error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.root.after(0, self._test_finished)

    def update_test_data(self, data_point):
        """Callback to update real-time data during test."""
        self.current_test_data.append(data_point)

    def _test_finished(self):
        """Called when test finishes."""
        self.test_running = False
        self.stop_requested = False

        self.progress.stop()

        self.btn_run_step.config(state="normal")
        self.btn_run_dist.config(state="normal")
        self.btn_reset.config(state="normal")
        self.btn_stop_test.config(state="disabled")
        self.btn_connect.config(state="normal")

    def stop_test(self):
        """Request test to stop."""
        self.stop_requested = True
        self.log_status("Stop requested, finishing test...")

    def reset_to_zero(self):
        """Run a reset-to-zero action in a background thread."""
        if not self.is_connected or self.test_running:
            return

        self.test_running = True
        self.stop_requested = False
        self.current_test_data = []

        # Disable buttons during reset
        self.btn_run_step.config(state="disabled")
        self.btn_run_dist.config(state="disabled")
        self.btn_reset.config(state="disabled")
        self.btn_stop_test.config(state="normal")
        self.btn_connect.config(state="disabled")

        self.progress.start(10)

        thread = threading.Thread(target=self._run_reset_thread, daemon=True)
        thread.start()

    def _run_reset_thread(self):
        try:
            self.log_status("Resetting to 0°...")
            result = self.tests.reset_to_zero_gui(
                tolerance=0.5,
                timeout=12.0,
                data_callback=self.update_test_data,
                status_callback=self.log_status,
                stop_check=lambda: self.stop_requested,
            )
            if result:
                self.all_results.append(result)
                self.log_status("Reset completed.")
        except Exception as e:
            self.log_status(f"Reset error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.root.after(0, self._test_finished)

    def update_plot(self):
        """Update real-time plot."""
        if self.test_running and len(self.current_test_data) > 0:
            df = pd.DataFrame(self.current_test_data)

            t = df["timestamp"].values
            yaw = df["measured_yaw_imu"].values
            target = df["target_yaw"].values
            u = df["motor_velocity"].values
            cur1 = (
                df["motor_current_1"].values
                if "motor_current_1" in df.columns
                else None
            )
            cur2 = (
                df["motor_current_2"].values
                if "motor_current_2" in df.columns
                else None
            )

            self.line_yaw.set_data(t, yaw)
            self.line_target.set_data(t, target)
            self.line_u.set_data(t, u)
            if cur1 is not None and cur2 is not None:
                self.line_cur1.set_data(t, cur1)
                self.line_cur2.set_data(t, cur2)

            if len(t) > 0:
                self.ax_yaw.set_xlim(0, max(1.0, t[-1]))
                self.ax_u.set_xlim(0, max(1.0, t[-1]))
                self.ax_effort.set_xlim(0, max(1.0, t[-1]))

                # Auto-scale y if needed
                if len(yaw) > 0:
                    ymin, ymax = yaw.min(), yaw.max()
                    margin = (ymax - ymin) * 0.1 + 5
                    self.ax_yaw.set_ylim(ymin - margin, ymax + margin)
                if cur1 is not None and cur2 is not None:
                    cmin = min(np.min(cur1), np.min(cur2))
                    cmax = max(np.max(cur1), np.max(cur2))
                    cmargin = max(10.0, (cmax - cmin) * 0.1)
                    self.ax_effort.set_ylim(cmin - cmargin, cmax + cmargin)

            self.canvas.draw_idle()

        self.root.after(100, self.update_plot)

    def save_all_results(self):
        """Save all collected results to CSV files."""
        if len(self.all_results) == 0:
            messagebox.showwarning("No Data", "No test results to save.")
            return

        try:
            self.tests.save_results(self.all_results)
            self.log_status(f"Saved {len(self.all_results)} test result(s) to CSV.")
            messagebox.showinfo("Success", "Results saved successfully!")
        except Exception as e:
            self.log_status(f"Error saving results: {e}")
            messagebox.showerror("Save Error", str(e))

    def on_closing(self):
        """Clean up on window close."""
        if self.test_running:
            if not messagebox.askokcancel(
                "Test Running", "A test is running. Force quit?"
            ):
                return

        self.stop_requested = True
        if self.is_connected:
            self.disconnect_hardware()

        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    main()
