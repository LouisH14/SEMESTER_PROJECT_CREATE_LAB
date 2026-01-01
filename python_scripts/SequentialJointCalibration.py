#!/usr/bin/env python3
"""
SequentialJointCalibration.py
Calibration et contrôle séquentiel pour bras robotique 2D à 4 articulations.

Fonctionnalités:
1. Calibration: Établit les coefficients K_i reliant θ_i au Yaw_IMU global
2. Contrôle séquentiel: Atteint une configuration cible en verrouillant séquentiellement les articulations
3. Filtre médian pour lectures IMU
4. Commande lente pour respecter contraintes d'inertie

Auteur: Antigravity AI Assistant
Date: 2025-12-19
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import sys
import serial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
from scipy import stats
import json

# Import Dynamixel controller
try:
    from dynamixel_controller import DynamixelController, BaseModel
except ImportError:
    print("Error: dynamixel_controller.py not found.")
    sys.exit(1)


class MedianFilter:
    """Filtre médian pour supprimer les pics de bruit de l'IMU."""

    def __init__(self, window_size=9):
        """
        Args:
            window_size: Taille de la fenêtre du filtre (nombre impair recommandé)
        """
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def update(self, value):
        """
        Ajoute une nouvelle valeur et retourne la médiane filtrée.

        Args:
            value: Nouvelle valeur à filtrer

        Returns:
            Valeur filtrée (médiane de la fenêtre)
        """
        self.buffer.append(value)
        if len(self.buffer) > 0:
            return float(np.median(self.buffer))
        return value


#!/usr/bin/env python3
"""
SequentialJointCalibration.py
Calibration et contrôle séquentiel pour bras robotique 2D à 4 articulations.

Fonctionnalités:
1. Calibration: Établit les coefficients K_i reliant θ_i au Yaw_IMU global
2. Contrôle séquentiel: Atteint une configuration cible en verrouillant séquentiellement les articulations
3. Filtre médian pour lectures IMU
4. Commande lente pour respecter contraintes d'inertie

Auteur: Antigravity AI Assistant
Date: 2025-12-19
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import sys
import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import json

# Import Dynamixel controller
try:
    from dynamixel_controller import DynamixelController, BaseModel
except ImportError:
    print("Error: dynamixel_controller.py not found.")
    sys.exit(1)


class MedianFilter:
    """Filtre médian pour supprimer les pics de bruit de l'IMU."""

    def __init__(self, window_size=9):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def update(self, value):
        self.buffer.append(value)
        if len(self.buffer) > 0:
            return float(np.median(self.buffer))
        return value

    def reset(self):
        self.buffer.clear()


class JointCalibrator:
    """Gère la calibration des coefficients de distribution K_i."""

    def __init__(self, num_joints=4):
        self.num_joints = num_joints
        self.K_coefficients = np.zeros(num_joints)
        self.is_calibrated = False

        self.calibration_data = {
            "yaw_imu": [],
            "theta_encoders": [[] for _ in range(num_joints)],
            "timestamps": [],
        }

    def add_calibration_point(self, yaw_imu, theta_encoders, timestamp):
        self.calibration_data["yaw_imu"].append(yaw_imu)
        self.calibration_data["timestamps"].append(timestamp)
        for i in range(self.num_joints):
            if i < len(theta_encoders):
                self.calibration_data["theta_encoders"][i].append(theta_encoders[i])

    def compute_coefficients(self):
        if len(self.calibration_data["yaw_imu"]) < 10:
            print("Pas assez de données pour calibration (minimum 10 points)")
            return False
        yaw_array = np.array(self.calibration_data["yaw_imu"])
        for i in range(self.num_joints):
            theta_array = np.array(self.calibration_data["theta_encoders"][i])
            if len(theta_array) != len(yaw_array):
                print(f"Erreur: dimensions incompatibles pour articulation {i}")
                return False
            if np.sum(yaw_array**2) > 0:
                self.K_coefficients[i] = np.sum(theta_array * yaw_array) / np.sum(
                    yaw_array**2
                )
            else:
                print(f"Attention: variance nulle pour articulation {i}")
                self.K_coefficients[i] = 0.0
            print(f"K_{i+1} = {self.K_coefficients[i]:.4f}")
        self.is_calibrated = True
        return True

    def save_calibration(self, filename):
        calib_dict = {
            "K_coefficients": self.K_coefficients.tolist(),
            "num_joints": self.num_joints,
            "calibration_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(filename, "w") as f:
            json.dump(calib_dict, f, indent=4)
        print(f"Calibration sauvegardée: {filename}")

    def load_calibration(self, filename):
        try:
            with open(filename, "r") as f:
                calib_dict = json.load(f)
            self.K_coefficients = np.array(calib_dict["K_coefficients"])
            self.num_joints = calib_dict["num_joints"]
            self.is_calibrated = True
            print(f"Calibration chargée: {filename}")
            print(f"Coefficients K: {self.K_coefficients}")
            return True
        except Exception as e:
            print(f"Erreur chargement calibration: {e}")
            return False

    def predict_yaw_for_angle(self, joint_index, target_angle):
        if not self.is_calibrated:
            raise ValueError("Calibration non effectuée")
        if self.K_coefficients[joint_index] == 0:
            raise ValueError(f"K_{joint_index+1} est nul, calibration invalide")
        return target_angle / self.K_coefficients[joint_index]


class SequentialController:
    """Contrôleur séquentiel pour atteindre une configuration cible."""

    def __init__(
        self,
        calibrator,
        motor_controller,
        median_filter,
        velocity_cmd=5.0,
        tolerance_deg=0.5,
        stabilization_time=0.2,
        actuation_sign=1.0,
        max_du=10.0,
        imu_fresh_fn=None,
    ):
        self.calibrator = calibrator
        self.motor_controller = motor_controller
        self.median_filter = median_filter
        self.velocity_cmd = velocity_cmd
        self.tolerance_deg = tolerance_deg
        self.stabilization_time = stabilization_time
        self.max_du = max_du
        self.actuation_sign = 1.0 if actuation_sign is None else float(actuation_sign)
        self._imu_fresh_fn = (
            imu_fresh_fn if imu_fresh_fn is not None else (lambda: True)
        )

        self.current_yaw = 0.0
        self.is_running = False
        self.stop_requested = False

    def reach_target_configuration(
        self, theta_targets, yaw_callback=None, status_callback=None
    ):
        if not self.calibrator.is_calibrated:
            if status_callback:
                status_callback(-1, "ERREUR: Calibration requise")
            return False

        self.is_running = True
        self.stop_requested = False
        yaw_reference = self.current_yaw

        for joint_idx in range(len(theta_targets)):
            if self.stop_requested:
                if status_callback:
                    status_callback(joint_idx, "ARRÊT demandé")
                return False

            target_theta = theta_targets[joint_idx]
            if status_callback:
                status_callback(
                    joint_idx, f"Cible θ_{joint_idx+1} = {target_theta:.2f}°"
                )
            try:
                delta_yaw = target_theta / self.calibrator.K_coefficients[joint_idx]
                yaw_target = yaw_reference + delta_yaw
            except ZeroDivisionError:
                if status_callback:
                    status_callback(joint_idx, f"ERREUR: K_{joint_idx+1} = 0")
                return False

            # Vérification limite de sécurité sur yaw_target
            if abs(yaw_target) > 40.0:
                if status_callback:
                    status_callback(
                        joint_idx,
                        f"ERREUR: Yaw cible {yaw_target:.2f}° dépasse limite sécurité (±40°). Contrôle non faisable.",
                    )
                self.motor_controller.set_goal_velocity_rad([0.0, 0.0])
                self.is_running = False
                return False

            if status_callback:
                status_callback(joint_idx, f"Yaw cible = {yaw_target:.2f}°")
            if yaw_callback:
                yaw_callback(yaw_target)

            success = self._control_to_yaw_target(
                yaw_target, yaw_callback, status_callback, joint_idx
            )
            if not success:
                if status_callback:
                    status_callback(joint_idx, "ÉCHEC atteinte cible")
                return False

            yaw_reference = self.current_yaw
            if status_callback:
                status_callback(
                    joint_idx, f"✓ Articulation θ_{joint_idx+1} verrouillée"
                )
            time.sleep(self.stabilization_time)

        self.motor_controller.set_goal_velocity_rad([0.0, 0.0])
        if status_callback:
            status_callback(-1, "✓ Configuration cible atteinte")
        self.is_running = False
        return True

    def _control_to_yaw_target(
        self, yaw_target, yaw_callback, status_callback, joint_idx
    ):
        K_P = 0.5
        MAX_TIME = 30.0
        start_time = time.time()
        stable_count = 0
        STABLE_THRESHOLD = 10
        last_u = 0.0
        YAW_SOFT_LIMIT = 60.0

        while time.time() - start_time < MAX_TIME:
            if self.stop_requested:
                return False
            if not self._imu_fresh_fn():
                if status_callback:
                    status_callback(joint_idx, "ERREUR: IMU non rafraîchie (arrêt)")
                try:
                    self.motor_controller.set_goal_velocity_rad([0.0, 0.0])
                except:
                    pass
                return False

            error = yaw_target - self.current_yaw
            u = K_P * error
            u = np.clip(u, -self.velocity_cmd, self.velocity_cmd)
            du = np.clip(
                u - last_u, -self.max_du, self.max_du
            )  # Use configurable max_du
            u = last_u + du
            last_u = u

            try:
                s = self.actuation_sign if self.actuation_sign in (-1.0, 1.0) else 1.0
                self.motor_controller.set_goal_velocity_rad([s * u, -s * u])
            except:
                pass

            if abs(self.current_yaw) > YAW_SOFT_LIMIT:
                if status_callback:
                    status_callback(joint_idx, "LIMITE YAW atteinte (arrêt)")
                try:
                    self.motor_controller.set_goal_velocity_rad([0.0, 0.0])
                except:
                    pass
                return False

            # Debug toutes les 0.5 s
            if status_callback and ((time.time() - start_time) % 0.5) < 0.02:
                status_callback(
                    joint_idx,
                    f"DEBUG: yaw={self.current_yaw:.2f}°, cible={yaw_target:.2f}°, err={error:.2f}°, u={u:.2f} rad/s",
                )

            if abs(error) < self.tolerance_deg:
                stable_count += 1
                if stable_count >= STABLE_THRESHOLD:
                    self.motor_controller.set_goal_velocity_rad([0.0, 0.0])
                    return True
            else:
                stable_count = 0
            time.sleep(0.02)

        if status_callback:
            status_callback(joint_idx, "TIMEOUT")
        return False

    def stop(self):
        self.stop_requested = True
        if self.motor_controller:
            try:
                self.motor_controller.set_goal_velocity_rad([0.0, 0.0])
            except:
                pass


class SequentialCalibrationApp:
    """Application GUI pour calibration et contrôle séquentiel."""

    def __init__(self, root):
        self.root = root
        self.root.title("Calibration et Contrôle Séquentiel - Bras Robotique 4-DOF")
        self.root.geometry("1400x900")

        self.MOTOR_PORT = "COM8"
        self.SENSOR_PORT = "COM3"
        self.BAUDRATE_MOTOR = 57600
        self.BAUDRATE_SENSOR = 115200

        self.calibrator = JointCalibrator(num_joints=4)
        self.median_filter = MedianFilter(window_size=9)
        self.motor_controller = None
        self.sensor_serial = None
        self.sequential_controller = None

        self.is_connected = False
        self.is_calibrating = False
        self.is_controlling = False
        self.current_yaw_raw = 0.0
        self.current_yaw_filtered = 0.0
        self.current_yaw_desired = 0.0  # For plotting during calibration
        self.encoder_angles = [0.0] * 4  # filtered angles
        self.encoder_filters = [MedianFilter(window_size=9) for _ in range(4)]
        self.last_imu_ts = 0.0
        # Par sécurité, signe d'actionnement forcé à -1 (configuration câblage actuelle)
        self.actuation_sign = -1.0

        self.sensor_thread = None
        self.stop_sensor_thread = False

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        conn_frame = ttk.LabelFrame(
            self.root, text="Connexion Matérielle", padding="10"
        )
        conn_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(conn_frame, text=f"Moteurs: {self.MOTOR_PORT}").pack(
            side="left", padx=10
        )
        ttk.Label(conn_frame, text=f"Capteur: {self.SENSOR_PORT}").pack(
            side="left", padx=10
        )
        self.btn_connect = ttk.Button(
            conn_frame, text="Connecter", command=self.toggle_connection
        )
        self.btn_connect.pack(side="left", padx=20)
        self.led_status = ttk.Label(
            conn_frame, text="DÉCONNECTÉ", foreground="red", font=("Arial", 10, "bold")
        )
        self.led_status.pack(side="left", padx=10)

        calib_frame = ttk.LabelFrame(
            self.root, text="Calibration des Coefficients K_i", padding="10"
        )
        calib_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(
            calib_frame, text="1. Démarrer le balayage automatique (tous angles libres)"
        ).pack(anchor="w")
        ttk.Label(
            calib_frame, text="2. Le système collecte automatiquement les données"
        ).pack(anchor="w")
        ttk.Label(
            calib_frame, text="3. Les coefficients sont calculés automatiquement"
        ).pack(anchor="w")
        btn_row = ttk.Frame(calib_frame)
        btn_row.pack(fill="x", pady=10)
        self.btn_start_calib = ttk.Button(
            btn_row,
            text="Démarrer Balayage Auto",
            command=self.start_calibration,
            state="disabled",
        )
        self.btn_start_calib.pack(side="left", padx=5)
        self.btn_stop_calib = ttk.Button(
            btn_row,
            text="Arrêter et Calculer K_i",
            command=self.stop_calibration,
            state="disabled",
        )
        self.btn_stop_calib.pack(side="left", padx=5)
        self.btn_save_calib = ttk.Button(
            btn_row,
            text="Sauvegarder Calibration",
            command=self.save_calibration,
            state="disabled",
        )
        self.btn_save_calib.pack(side="left", padx=5)
        self.btn_load_calib = ttk.Button(
            btn_row,
            text="Charger Calibration",
            command=self.load_calibration,
            state="disabled",
        )
        self.btn_load_calib.pack(side="left", padx=5)
        self.lbl_coeffs = ttk.Label(
            calib_frame, text="Coefficients: Non calibrés", font=("Courier", 10)
        )
        self.lbl_coeffs.pack(anchor="w", pady=5)

        ctrl_frame = ttk.LabelFrame(self.root, text="Contrôle Séquentiel", padding="10")
        ctrl_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(ctrl_frame, text="Angles cibles (θ₁, θ₂, θ₃, θ₄) en degrés:").pack(
            anchor="w"
        )
        angles_row = ttk.Frame(ctrl_frame)
        angles_row.pack(fill="x", pady=5)
        self.entries_theta = []
        for i in range(4):
            ttk.Label(angles_row, text=f"θ{i+1}:").pack(side="left", padx=5)
            entry = ttk.Entry(angles_row, width=8)
            entry.insert(0, "0.0")
            entry.pack(side="left", padx=5)
            self.entries_theta.append(entry)
        ctrl_btn_row = ttk.Frame(ctrl_frame)
        ctrl_btn_row.pack(fill="x", pady=10)
        self.btn_reach = ttk.Button(
            ctrl_btn_row,
            text="Atteindre Configuration",
            command=self.start_sequential_control,
            state="disabled",
        )
        self.btn_reach.pack(side="left", padx=5)
        self.btn_stop_ctrl = ttk.Button(
            ctrl_btn_row,
            text="ARRÊT D'URGENCE",
            command=self.emergency_stop,
            state="disabled",
        )
        self.btn_stop_ctrl.pack(side="left", padx=5)
        self.txt_status = tk.Text(ctrl_frame, height=6, width=80, state="disabled")
        self.txt_status.pack(fill="x", pady=5)

        viz_frame = ttk.LabelFrame(
            self.root, text="Visualisation Temps Réel", padding="10"
        )
        viz_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.fig, (self.ax_yaw, self.ax_enc) = plt.subplots(
            2, 1, figsize=(12, 7), sharex=True
        )
        (self.line_yaw_raw,) = self.ax_yaw.plot(
            [], [], "k-", linewidth=0.5, alpha=0.3, label="Yaw Raw"
        )
        (self.line_yaw_filt,) = self.ax_yaw.plot(
            [], [], "b-", linewidth=2, label="Yaw Filtré"
        )
        (self.line_yaw_desired,) = self.ax_yaw.plot(
            [], [], "r--", linewidth=1.5, label="Yaw Désiré"
        )
        self.ax_yaw.set_title("Yaw IMU (Temps Réel)")
        self.ax_yaw.set_ylabel("Yaw (deg)")
        self.ax_yaw.legend()
        self.ax_yaw.grid(True)
        self.ax_yaw.set_ylim(-50, 50)

        # Traces temps-réel des 4 encodeurs
        colors = ["C0", "C1", "C2", "C3"]
        self.line_encoders = []
        for i in range(4):
            (line,) = self.ax_enc.plot([], [], color=colors[i], label=f"θ{i+1} (deg)")
            self.line_encoders.append(line)
        self.ax_enc.set_title("Encodeurs (Temps Réel)")
        self.ax_enc.set_xlabel("Temps (s)")
        self.ax_enc.set_ylabel("Angle (deg)")
        self.ax_enc.legend(loc="upper right")
        self.ax_enc.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.plot_t = deque(maxlen=800)
        self.plot_yaw_raw = deque(maxlen=800)
        self.plot_yaw_filt = deque(maxlen=800)
        self.plot_yaw_desired = deque(maxlen=800)
        self.plot_encoders = [deque(maxlen=800) for _ in range(4)]
        self.plot_start_time = time.time()
        self.after_id = None
        self.after_id = self.root.after(100, self.update_plot)

    def toggle_connection(self):
        if not self.is_connected:
            self.connect()
        else:
            self.disconnect()

    def connect(self):
        try:
            print(f"Connexion moteurs sur {self.MOTOR_PORT}...")
            self.motor_controller = DynamixelController(
                port_name=self.MOTOR_PORT,
                motor_list=[BaseModel(1), BaseModel(0)],
                baudrate=self.BAUDRATE_MOTOR,
            )
            self.motor_controller.activate_controller()
            self.motor_controller.set_operating_mode_all("velocity_control")
            self.motor_controller.torque_on()

            print(f"Connexion capteur sur {self.SENSOR_PORT}...")
            self.sensor_serial = serial.Serial(
                self.SENSOR_PORT, self.BAUDRATE_SENSOR, timeout=0.01
            )
            self.sensor_serial.reset_input_buffer()

            self.sequential_controller = SequentialController(
                self.calibrator,
                self.motor_controller,
                self.median_filter,
                velocity_cmd=5.0,
                tolerance_deg=0.5,
                stabilization_time=0.2,
                actuation_sign=-1.0,
                max_du=10.0,
                imu_fresh_fn=self._imu_fresh,
            )

            self.stop_sensor_thread = False
            self.last_imu_ts = (
                time.time()
            )  # Initialize as "fresh" - thread will update on each valid line
            self.sensor_thread = threading.Thread(
                target=self.sensor_read_loop, daemon=True
            )
            self.sensor_thread.start()

            self.is_connected = True
            self.led_status.config(text="CONNECTÉ", foreground="green")
            self.btn_connect.config(text="Déconnecter")
            self.btn_start_calib.config(state="normal")
            self.btn_load_calib.config(state="normal")
            self.root.after(500, self._auto_calibrate_polarity)
        except Exception as e:
            messagebox.showerror("Erreur Connexion", str(e))
            self.disconnect()

    def disconnect(self):
        self.is_connected = False
        self.stop_sensor_thread = True
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
        self.motor_controller = None
        self.sensor_serial = None
        self.led_status.config(text="DÉCONNECTÉ", foreground="red")
        self.btn_connect.config(text="Connecter")
        self.btn_start_calib.config(state="disabled")
        self.btn_stop_calib.config(state="disabled")
        self.btn_reach.config(state="disabled")

    def sensor_read_loop(self):
        print("[DEBUG] sensor_read_loop: Démarrage du thread de lecture IMU")
        line_count = 0
        while not self.stop_sensor_thread and self.is_connected:
            try:
                if self.sensor_serial and self.sensor_serial.in_waiting:
                    # Read all available data, process most recent complete line
                    try:
                        all_data = self.sensor_serial.read_all().decode(
                            "utf-8", errors="ignore"
                        )
                    except:
                        all_data = self.sensor_serial.read_all().decode(
                            "latin-1", errors="ignore"
                        )

                    lines = all_data.splitlines()
                    if len(lines) > 0:
                        last_line = lines[-1]  # Take most recent line
                        parts = last_line.split(",")

                        # Process if line has valid format
                        if len(parts) >= 9:
                            try:
                                raw_yaw = float(
                                    parts[8]
                                )  # Yaw at index 8 (matches RealTimeYawControl)
                            except:
                                raw_yaw = None

                            if raw_yaw is not None:
                                filt_yaw = self.median_filter.update(raw_yaw)
                                self.current_yaw_raw = raw_yaw
                                self.current_yaw_filtered = filt_yaw
                                self.last_imu_ts = time.time()
                                line_count += 1
                                if line_count <= 3:  # Log first 3 lines
                                    print(
                                        f"[DEBUG] IMU ligne #{line_count}: yaw={raw_yaw:.2f}°, filt={filt_yaw:.2f}°"
                                    )

                                if self.sequential_controller:
                                    self.sequential_controller.current_yaw = filt_yaw

                                try:
                                    counts = [int(float(parts[i])) for i in range(4)]
                                    angles_raw = self._counts_to_angles(counts)
                                    angles_filt = [
                                        self.encoder_filters[i].update(angles_raw[i])
                                        for i in range(4)
                                    ]
                                    self.encoder_angles = angles_filt
                                except:
                                    pass

                                if self.is_calibrating:
                                    t = time.time()
                                    self.calibrator.add_calibration_point(
                                        filt_yaw, self.encoder_angles, t
                                    )
            except Exception as e:
                print(f"[DEBUG] sensor_read_loop error: {e}")
                pass
            time.sleep(0.01)  # 100 Hz polling with in_waiting check prevents blocking

    def _counts_to_angles(self, counts):
        DEG_PER_COUNT = 3.75
        angles = []
        sgn = -1
        for c in counts:
            angles.append(sgn * c * DEG_PER_COUNT)
            sgn *= -1
        return angles

    def start_calibration(self):
        self.calibrator.calibration_data = {
            "yaw_imu": [],
            "theta_encoders": [[] for _ in range(4)],
            "timestamps": [],
        }
        self.median_filter.reset()
        for f in self.encoder_filters:
            f.reset()
        self.is_calibrating = True
        self.btn_start_calib.config(state="disabled")
        self.btn_stop_calib.config(state="normal")
        if not self._wait_for_fresh_imu(timeout=5.0):
            self.log_status(
                "IMU non disponible (pas de données récentes). Vérifiez le port SENSOR_PORT et la connexion."
            )
            self.is_calibrating = False
            self.btn_start_calib.config(state="normal")
            self.btn_stop_calib.config(state="disabled")
            return
        self.log_status("Balayage automatique démarré...")
        sweep_thread = threading.Thread(target=self._automatic_sweep, daemon=True)
        sweep_thread.start()

    def _automatic_sweep(self):
        step_deg = 3.0
        up = np.arange(0.0, 30.0 + step_deg / 2.0, step_deg)  # 0 -> 30
        down = np.arange(
            30.0 - step_deg, -30.0 - step_deg / 2.0, -step_deg
        )  # 27 -> -30
        back = np.arange(-30.0 + step_deg, 0.0 + step_deg / 2.0, step_deg)  # -27 -> 0
        yaw_targets = np.concatenate([up, down, back])
        num_steps = len(yaw_targets)
        wait_time = 2.0
        self.log_status(
            f"Balayage sécurisé: 0→30→-30→0 par pas de {step_deg}° ({num_steps} étapes)"
        )
        K_P = 0.4
        last_u = 0.0
        MAX_DU = 0.2
        s = self.actuation_sign if self.actuation_sign in (-1.0, 1.0) else 1.0

        for i, yaw_target in enumerate(yaw_targets):
            if not self.is_calibrating:
                self.log_status("Balayage interrompu")
                break
            self.current_yaw_desired = yaw_target  # Update desired yaw for plotting
            self.log_status(
                f"Position {i+1}/{num_steps}: Yaw cible = {yaw_target:.1f}°"
            )
            if not self._wait_for_fresh_imu(timeout=2.0):
                self.log_status("IMU non rafraîchie, saut de cette position")
                continue
            start_time = time.time()
            timeout = 10.0
            stable_count = 0
            while time.time() - start_time < timeout:
                if not self.is_calibrating:
                    break
                if not self._imu_fresh():
                    time.sleep(0.05)
                    continue
                error = yaw_target - self.current_yaw_filtered
                u = K_P * error
                u = np.clip(u, -1.0, 1.0)
                du = np.clip(u - last_u, -MAX_DU, MAX_DU)
                u = last_u + du
                last_u = u
                if self.motor_controller:
                    try:
                        self.motor_controller.set_goal_velocity_rad([s * u, -s * u])
                    except:
                        pass
                if abs(error) < 1.0:
                    stable_count += 1
                    if stable_count >= 20:
                        break
                else:
                    stable_count = 0
                time.sleep(0.02)
            if self.motor_controller:
                try:
                    self.motor_controller.set_goal_velocity_rad([0.0, 0.0])
                except:
                    pass
            self.log_status(f"  Position atteinte, collecte de données...")
            time.sleep(wait_time)
        self.log_status("Retour à la position zéro...")
        self._goto_yaw(0.0)
        if self.motor_controller:
            try:
                self.motor_controller.set_goal_velocity_rad([0.0, 0.0])
            except:
                pass
        self.log_status(
            f"Balayage terminé. {len(self.calibrator.calibration_data['yaw_imu'])} points collectés"
        )

    def _goto_yaw(self, yaw_target, timeout=10.0):
        K_P = 0.4
        s = self.actuation_sign if self.actuation_sign in (-1.0, 1.0) else 1.0
        last_u = 0.0
        MAX_DU = 0.2
        start_time = time.time()
        stable_count = 0
        while time.time() - start_time < timeout:
            if not self.is_calibrating:
                break
            if not self._imu_fresh():
                time.sleep(0.05)
                continue
            error = yaw_target - self.current_yaw_filtered
            u = K_P * error
            u = np.clip(u, -1.0, 1.0)
            du = np.clip(u - last_u, -MAX_DU, MAX_DU)
            u = last_u + du
            last_u = u
            if self.motor_controller:
                try:
                    self.motor_controller.set_goal_velocity_rad([s * u, -s * u])
                except:
                    pass
            if abs(error) < 1.0:
                stable_count += 1
                if stable_count >= 20:
                    break
            else:
                stable_count = 0
            time.sleep(0.02)

    def _imu_fresh(self):
        age = time.time() - self.last_imu_ts
        is_fresh = age < 0.25
        return is_fresh

    def _wait_for_fresh_imu(self, timeout=2.0):
        print(
            f"[DEBUG] _wait_for_fresh_imu: Attente de données IMU pendant {timeout}s..."
        )
        t0 = time.time()
        attempt = 0
        while time.time() - t0 < timeout:
            age = time.time() - self.last_imu_ts
            is_fresh = age < 0.25
            if is_fresh:
                print(
                    f"[DEBUG] IMU données fraiches reçues après {time.time() - t0:.2f}s (age={age:.3f}s)"
                )
                return True
            if attempt % 10 == 0:  # Print every 500ms
                print(
                    f"[DEBUG] En attente... age={age:.2f}s, elapsed={time.time() - t0:.1f}s"
                )
            attempt += 1
            time.sleep(0.05)
        print(
            f"[DEBUG] TIMEOUT: IMU pas disponible après {timeout}s (last_age={time.time() - self.last_imu_ts:.2f}s)"
        )
        return False

    def _auto_calibrate_polarity(self):
        # Sécurité: forcer le signe à -1 (désactive l'auto-détection qui peut être dangereuse)
        self.actuation_sign = -1.0
        if self.sequential_controller:
            self.sequential_controller.actuation_sign = -1.0
        self.log_status("Signe d'actionnement forcé à -1 (auto-détection désactivée)")

    def stop_calibration(self):
        self.is_calibrating = False
        self.current_yaw_desired = 0.0  # Reset desired yaw
        if self.motor_controller:
            try:
                self.motor_controller.set_goal_velocity_rad([0.0, 0.0])
            except:
                pass
        time.sleep(0.5)
        self.log_status("Calcul des coefficients K_i...")
        success = self.calibrator.compute_coefficients()
        if success:
            coeffs_str = ", ".join(
                [
                    f"K_{i+1}={k:.4f}"
                    for i, k in enumerate(self.calibrator.K_coefficients)
                ]
            )
            self.lbl_coeffs.config(text=f"Coefficients: {coeffs_str}")
            self.log_status("✓ Calibration terminée avec succès")
            self.btn_save_calib.config(state="normal")
            self.btn_reach.config(state="normal")
            # Affiche les courbes comparatives (encodeurs vs Yaw × K_i)
            self.plot_calibration_trajectories()
        else:
            self.log_status("✗ Échec calibration (données insuffisantes)")
        self.btn_start_calib.config(state="normal")
        self.btn_stop_calib.config(state="disabled")

    def save_calibration(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if filename:
            self.calibrator.save_calibration(filename)

    def plot_calibration_trajectories(self):
        data = self.calibrator.calibration_data
        yaw = np.array(data.get("yaw_imu", []))
        ts = np.array(data.get("timestamps", []))
        if len(yaw) < 2 or len(ts) != len(yaw):
            return
        t = ts - ts[0]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        axes = axes.ravel()
        for i in range(4):
            if i >= len(data.get("theta_encoders", [])):
                continue
            theta = np.array(data["theta_encoders"][i])
            if len(theta) != len(yaw):
                continue
            pred = yaw * self.calibrator.K_coefficients[i]
            ax = axes[i]
            ax.plot(t, theta, color="C0", label="θ mesurée")
            ax.plot(t, pred, "--", color="C1", label="Yaw × K_i")
            ax.set_title(
                f"Articulation {i+1} (K={self.calibrator.K_coefficients[i]:.3f})"
            )
            ax.set_ylabel("Angle (deg)")
            ax.grid(True)
            if i == 2 or i == 3:
                ax.set_xlabel("Temps (s)")
            if i == 0:
                ax.legend()
        fig.suptitle("Trajectoires de calibration : encodeurs vs Yaw × K_i")
        fig.tight_layout()
        plt.show(block=False)

    def load_calibration(self):
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            success = self.calibrator.load_calibration(filename)
            if success:
                coeffs_str = ", ".join(
                    [
                        f"K_{i+1}={k:.4f}"
                        for i, k in enumerate(self.calibrator.K_coefficients)
                    ]
                )
                self.lbl_coeffs.config(text=f"Coefficients: {coeffs_str}")
                self.log_status(f"✓ Calibration chargée: {filename}")
                self.btn_save_calib.config(state="normal")
                self.btn_reach.config(state="normal")
            else:
                self.log_status("✗ Échec chargement calibration")

    def start_sequential_control(self):
        try:
            theta_targets = [float(e.get()) for e in self.entries_theta]
        except ValueError:
            messagebox.showerror("Erreur", "Angles cibles invalides")
            return
        self.log_status(f"Démarrage contrôle vers {theta_targets}")
        self.is_controlling = True
        self.btn_reach.config(state="disabled")
        self.btn_stop_ctrl.config(state="normal")
        ctrl_thread = threading.Thread(
            target=self._run_sequential_control, args=(theta_targets,), daemon=True
        )
        ctrl_thread.start()

    def _run_sequential_control(self, theta_targets):
        def status_cb(idx, msg):
            # ignore idx, forward message to logger
            self.log_status(msg)

        def yaw_cb(yaw_target):
            self.current_yaw_desired = yaw_target

        success = self.sequential_controller.reach_target_configuration(
            theta_targets, yaw_callback=yaw_cb, status_callback=status_cb
        )
        self.is_controlling = False
        self.btn_reach.config(state="normal")
        self.btn_stop_ctrl.config(state="disabled")
        if success:
            self.log_status("✓✓✓ Configuration cible atteinte avec succès ✓✓✓")
        else:
            self.log_status("✗ Contrôle interrompu ou échec")

    def emergency_stop(self):
        self.log_status("!!! ARRÊT D'URGENCE !!!")
        if self.sequential_controller:
            self.sequential_controller.stop()
        if self.motor_controller:
            try:
                self.motor_controller.set_goal_velocity_rad([0.0, 0.0])
            except:
                pass
        self.is_controlling = False
        self.btn_reach.config(state="normal")
        self.btn_stop_ctrl.config(state="disabled")

    def log_status(self, message):
        self.txt_status.config(state="normal")
        self.txt_status.insert("end", f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.txt_status.see("end")
        self.txt_status.config(state="disabled")
        print(message)

    def update_plot(self):
        if not self.is_connected:
            self.after_id = self.root.after(100, self.update_plot)
            return
        try:
            t_now = time.time() - self.plot_start_time
            self.plot_t.append(t_now)
            self.plot_yaw_raw.append(self.current_yaw_raw)
            self.plot_yaw_filt.append(self.current_yaw_filtered)
            self.plot_yaw_desired.append(self.current_yaw_desired)
            for i in range(4):
                self.plot_encoders[i].append(self.encoder_angles[i])
            if len(self.plot_t) > 1:
                self.line_yaw_raw.set_data(self.plot_t, self.plot_yaw_raw)
                self.line_yaw_filt.set_data(self.plot_t, self.plot_yaw_filt)
                self.line_yaw_desired.set_data(self.plot_t, self.plot_yaw_desired)
                for i in range(4):
                    self.line_encoders[i].set_data(self.plot_t, self.plot_encoders[i])
                self.ax_yaw.set_xlim(max(0, t_now - 12), t_now + 0.5)
                self.ax_enc.set_xlim(max(0, t_now - 12), t_now + 0.5)
                # Auto-scale encoders lightly around data range
                try:
                    all_enc = [v for enc in self.plot_encoders for v in enc]
                    if all_enc:
                        mn, mx = min(all_enc), max(all_enc)
                        pad = max(5, 0.1 * max(1, abs(mx), abs(mn)))
                        self.ax_enc.set_ylim(mn - pad, mx + pad)
                except:
                    pass
                self.canvas.draw()
        except:
            pass
        self.after_id = self.root.after(100, self.update_plot)

    def on_closing(self):
        self.emergency_stop()
        self.disconnect()
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except:
                pass
        try:
            self.root.destroy()
        except:
            pass
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = SequentialCalibrationApp(root)
    root.mainloop()
