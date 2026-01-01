import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import sys
import serial
import numpy as np
import math
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal
from scipy.interpolate import interp1d

# Import Dynamixel controller
from dynamixel_controller import DynamixelController, BaseModel


class WidebandIDApp:
    """GUI application for wideband (chirp) motor system identification.
    Performs frequency sweep identification and computes Bode diagram."""

    def __init__(self, root):
        self.root = root
        self.root.title("Wideband Identification (Chirp)")
        self.root.geometry("1400x900")

        # --- State ---
        self.is_running_id = False
        self.is_connected = False
        self.stop_requested = False
        self.start_time = 0.0

        # Hardware
        self.motor_controller = None
        self.sensor_serial = None
        self.is_reading_sensors = False

        # Data Buffers
        self.measured_time = []
        self.measured_input = []  # Velocity Sent
        self.measured_output = []  # Yaw Angle Measured

        # Signal Parameters
        self.fs_control = 50.0  # Hz
        self.duration = 30.0  # seconds
        self.f_min = 0.1  # Hz (approx 0.6 rad/s)
        self.f_max = 1.0  # Hz (approx 12 rad/s) - User said 3 rad/s, we can adjust
        self.pos_bound = 5.0  # Max integral value

        self.freq_data = None
        self.time_data = None

        # Filter State
        self.filter_window = 5
        self.filter_history = deque(maxlen=self.filter_window)

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        """Create all GUI widgets for the wideband identification interface."""
        # Configuration Frame
        config_frame = ttk.LabelFrame(self.root, text="Configuration", padding="10")
        config_frame.pack(fill="x", padx=10, pady=5)

        # Hardware Ports
        ttk.Label(config_frame, text="Motors (COM8):").grid(row=0, column=0, padx=5)
        self.entry_motor = ttk.Entry(config_frame, width=10)
        self.entry_motor.insert(0, "COM8")
        self.entry_motor.grid(row=0, column=1, padx=5)

        ttk.Label(config_frame, text="Sensors (COM3):").grid(row=0, column=2, padx=5)
        self.entry_sensor = ttk.Entry(config_frame, width=10)
        self.entry_sensor.insert(0, "COM3")
        self.entry_sensor.grid(row=0, column=3, padx=5)

        self.btn_connect = ttk.Button(
            config_frame, text="Connect", command=self.toggle_connection
        )
        self.btn_connect.grid(row=0, column=4, padx=20)

        # Signal Parameters
        ttk.Label(config_frame, text="Duration (s):").grid(row=0, column=5, padx=5)
        self.entry_dur = ttk.Entry(config_frame, width=5)
        self.entry_dur.insert(0, "90")
        self.entry_dur.grid(row=0, column=6, padx=5)

        ttk.Label(config_frame, text="Max Int (rad):").grid(row=0, column=7, padx=5)
        self.entry_bound = ttk.Entry(config_frame, width=5)
        self.entry_bound.insert(0, "4.5")  # Safety margin < 5

        self.entry_bound.grid(row=0, column=8, padx=5)

        ttk.Label(config_frame, text="Filter K:").grid(row=0, column=9, padx=5)
        self.entry_k = ttk.Entry(config_frame, width=5)
        self.entry_k.insert(0, "5")
        self.entry_k.grid(row=0, column=10, padx=5)

        # Action Buttons
        self.btn_gen = ttk.Button(
            config_frame, text="Generate & Preview", command=self.generate_signal
        )
        self.btn_gen.grid(row=0, column=11, padx=20)

        self.btn_start = ttk.Button(
            config_frame,
            text="Start Identification",
            command=self.start_identification,
            state="disabled",
        )
        self.btn_start.grid(row=0, column=12, padx=20)

        self.btn_stop = ttk.Button(
            config_frame, text="STOP", command=self.emergency_stop, state="disabled"
        )
        self.btn_stop.grid(row=0, column=13, padx=20)

        self.btn_save = ttk.Button(
            config_frame, text="Save", command=self.save_results, state="disabled"
        )
        self.btn_save.grid(row=0, column=14, padx=20)

        # Status and Results display
        self.lbl_status = ttk.Label(
            config_frame,
            text="Waiting for signal generation...",
            font=("Arial", 9, "italic"),
        )
        self.lbl_status.grid(row=1, column=0, columnspan=13, pady=5)

        # Plots
        self.fig, (self.ax_sig, self.ax_mag, self.ax_phase) = plt.subplots(
            3, 1, figsize=(10, 10)
        )
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)

        # Initialize time-domain plot axes
        self.ax_sig.set_title("Input Signal (Velocity) and Output (Yaw)")
        self.ax_sig.set_xlabel("Time (s)")

        # Line objects for real-time data updates
        (self.line_rt_input,) = self.ax_sig.plot(
            [], [], "b-", label="Measured Input", alpha=0.7
        )
        self.ax_sig_twin = self.ax_sig.twinx()
        (self.line_rt_output,) = self.ax_sig_twin.plot(
            [], [], "r--", label="Measured Output", alpha=0.7
        )

        # Initialize frequency-domain plot axes
        self.ax_mag.set_title("Bode Diagram: Magnitude")
        self.ax_mag.set_ylabel("Gain (dB)")
        self.ax_mag.grid(True, which="both")

        self.ax_phase.set_title("Bode Diagram: Phase")
        self.ax_phase.set_ylabel("Phase (deg)")
        self.ax_phase.set_xlabel("Frequency (rad/s)")
        self.ax_phase.grid(True, which="both")

        self.input_u = None
        self.input_int = None
        self.time_vec = None

    def unwrap_90_absolute(self, val):
        """Forces value into [-90, 90] assuming modulo behavior."""
        while val > 90:
            val -= 90
        while val < -90:
            val += 90
        return val

    def save_results(self):
        """Save identification results to CSV files with timestamp."""
        if self.freq_data is None or self.time_data is None:
            return

        from tkinter import filedialog
        import csv

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base_filename = f"ID_Result_{timestamp}"

        # Ask user for save directory
        save_dir = filedialog.askdirectory(title="Choose save directory")
        if not save_dir:
            return

        try:
            # 1. Save Plot
            plot_path = f"{save_dir}/{base_filename}.png"
            self.fig.savefig(plot_path)

            # 2. Save Time Data
            time_path = f"{save_dir}/{base_filename}_time.csv"
            t, u, y = self.time_data
            with open(time_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Time", "Input_Vel_rad_s", "Output_Yaw_deg"])
                for i in range(len(t)):
                    writer.writerow([t[i], u[i], y[i]])

            # 3. Save Freq Data
            freq_path = f"{save_dir}/{base_filename}_bode.csv"
            w, m, p = self.freq_data
            with open(freq_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Omega_rad_s", "Magnitude_dB", "Phase_deg"])
                for i in range(len(w)):
                    writer.writerow([w[i], m[i], p[i]])

            messagebox.showinfo("Success", f"Files saved in:\n{save_dir}")

        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def generate_signal(self):
        """Generate logarithmic chirp signal for wideband identification.
        Uses position-based approach: y_target(t) = bound * sin(phi(t))
        to ensure integral stays within bounds during frequency sweep."""
        try:
            T = float(self.entry_dur.get())
            bound = float(self.entry_bound.get())

            w_min = 0.1  # rad/s
            w_max = 3.0  # rad/s (User req)

            dt = 1.0 / self.fs_control
            num_samples = int(T * self.fs_control)
            t = np.linspace(0, T, num_samples)

            # --- CHIRP DESIGN ON POSITION ---
            # Standard Linear Chirp for Velocity u(t)
            # Using position-based approach prevents integral drift:
            # y_target(t) = bound * sin(phi(t))
            # This guarantees position stays bounded by 'bound'
            # u(t) = y_target'(t)

            # Logarithmic Chirp Phase:
            # phi(t) = (w_min * T / ln(w_max/w_min)) * (exp(ln(w_max/w_min)*t/T) - 1)

            k = np.log(w_max / w_min)
            phi = (w_min * T / k) * (np.exp(k * t / T) - 1)

            # Apply hanning window to start and stop smoothly to 0
            window = np.hanning(num_samples)
            # Or better: simply start at 0 phase and ensure velocity starts at 0.
            # Ideally we want full amplitude periodic-like signal.

            # Position Target using Sine to start at 0
            # If pos = bound * sin(phi), then pos(0) = 0. Integral is centered [-bound, bound].
            pos_target = bound * np.sin(phi)

            # Fade in/out position to ensure we start/end at 0?
            # Actually, standard chirp is fine.
            # Let's compute numerical derivative for u(t)

            vel_target = np.gradient(pos_target, dt)

            # Verify constraints
            max_vel = np.max(np.abs(vel_target))
            max_pos = np.max(np.abs(pos_target))

            self.time_vec = t
            self.input_u = vel_target
            self.input_int = pos_target

            # Preview Plot
            self.ax_sig.clear()
            # Re-init twinx is tricky if ax_sig cleared, better to just update lines or re-create
            # Simple approach: clear fig and re-create for preview? Or just use same axes.
            # Let's reset the axes
            self.ax_sig_twin.remove()  # Remove old twin
            self.ax_sig_twin = self.ax_sig.twinx()

            self.ax_sig.plot(t, vel_target, "b", label="Velocity Command (u)")
            # Position is integral of velocity, use twin axis for different scale
            self.ax_sig_twin.plot(
                t, pos_target, "g--", label="Integral (Theoretical Position)"
            )
            self.ax_sig_twin.axhline(bound, color="r", linestyle=":", label="Bound +")
            self.ax_sig_twin.axhline(-bound, color="r", linestyle=":", label="Bound -")

            self.ax_sig.legend(loc="upper left")
            self.ax_sig_twin.legend(loc="upper right")
            self.ax_sig.set_title(
                f"Preview: Max Vel={max_vel:.2f}, Max Pos={max_pos:.2f} (Bound={bound})"
            )
            self.canvas.draw()

            self.lbl_status.config(text="Signal generated. Ready to start.")
            if self.is_connected:
                self.btn_start.config(state="normal")

        except Exception as e:
            messagebox.showerror("Generation Error", str(e))

    def toggle_connection(self):
        """Toggle hardware connection state."""
        if not self.is_connected:
            self.connect()
        else:
            self.disconnect()

    def connect(self):
        """Establish connections to motor controller and IMU sensor."""
        try:
            self.motor_controller = DynamixelController(
                port_name=self.entry_motor.get(),
                motor_list=[BaseModel(1), BaseModel(0)],
                baudrate=57600,
            )
            self.motor_controller.activate_controller()
            self.motor_controller.set_operating_mode_all("velocity_control")
            self.motor_controller.torque_on()

            self.sensor_serial = serial.Serial(
                self.entry_sensor.get(), 115200, timeout=0.01
            )
            self.sensor_serial.reset_input_buffer()

            self.is_connected = True
            self.btn_connect.config(text="Disconnect")
            if self.input_u is not None:
                self.btn_start.config(state="normal")
            self.lbl_status.config(text="Hardware connected.")

            # Start background sensor reading thread
            self.is_reading_sensors = True
            threading.Thread(target=self.sensor_loop, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Connection Error", str(e))
            self.disconnect()

    def disconnect(self):
        """Close all hardware connections gracefully."""
        self.is_connected = False
        self.is_reading_sensors = False
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
        self.btn_connect.config(text="Connect")
        self.btn_start.config(state="disabled")

    def sensor_loop(self):
        """Background thread: continuously read IMU data from serial port.
        Applies unwrapping and median filtering to yaw measurements."""
        while self.is_reading_sensors and self.sensor_serial:
            try:
                line = self.sensor_serial.readline()
                if line:
                    try:
                        s = line.decode().strip()
                        parts = s.split(",")
                        if len(parts) >= 9:
                            # Extract yaw angle (field 8) from IMU output
                            yaw_raw = float(parts[8])

                            # --- FILTERING PIPELINE ---
                            # 1. Unwrap angle to [-90, 90] range
                            yaw_corr = self.unwrap_90_absolute(yaw_raw)

                            # 2. Apply Median Filter for noise reduction
                            # Note: filter_history is shared, producer appends here
                            self.filter_history.append(yaw_corr)

                            if len(self.filter_history) > 0:
                                yaw_filt = float(np.median(self.filter_history))
                            else:
                                yaw_filt = yaw_corr

                            # Store filtered measurement with timestamp during identification
                            if self.is_running_id:
                                self.measured_output.append(
                                    (time.perf_counter(), yaw_filt)
                                )
                    except:
                        pass
            except:
                pass

    def start_identification(self):
        """Start the wideband identification test with current chirp signal."""
        if not self.is_connected or self.input_u is None:
            return

        self.is_running_id = True
        self.stop_requested = False
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_gen.config(state="disabled")
        self.btn_save.config(state="disabled")

        # Reset Data buffers for new test
        self.measured_input = []
        self.measured_output = []
        self.measured_time = []

        # Configure filter from UI parameters
        try:
            k = int(self.entry_k.get())
            if k < 1:
                k = 1
            if k % 2 == 0:
                k += 1  # Ensure odd window size
            self.filter_window = k
            self.filter_history = deque(maxlen=self.filter_window)
        except:
            # Fallback to default if invalid input
            self.filter_window = 5
            self.filter_history = deque(maxlen=5)

        # Prepare Real-time Plot display
        self.ax_sig.clear()
        self.ax_sig_twin.remove()
        self.ax_sig_twin = self.ax_sig.twinx()

        self.ax_sig.set_title("Identification in progress...")
        self.ax_sig.set_xlabel("Time (s)")
        self.ax_sig.set_ylabel("Velocity Input (rad/s)", color="b")
        self.ax_sig_twin.set_ylabel("Yaw Output (deg)", color="r")

        (self.line_rt_input,) = self.ax_sig.plot([], [], "b-", label="Measured Input")
        (self.line_rt_output,) = self.ax_sig_twin.plot(
            [], [], "r--", label="Measured Output"
        )

        # Start control loop in separate thread
        threading.Thread(target=self.control_worker, daemon=True).start()

    def emergency_stop(self):
        """Request immediate stop of identification test."""
        self.stop_requested = True

    def control_worker(self):
        """Worker thread: execute control loop and send velocity commands.
        Maintains 50 Hz synchronization for consistent sampling."""
        try:
            dt = 1.0 / self.fs_control
            start_t = time.perf_counter()
            self.start_time = start_t

            for i, u_val in enumerate(self.input_u):
                if self.stop_requested:
                    break

                # Synchronize loop timing
                target_wake = start_t + (i * dt)
                now = time.perf_counter()
                sleep_time = target_wake - now
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Send velocity command to both motors
                try:
                    self.motor_controller.set_goal_velocity_rad([u_val, -u_val])
                    # Store exact timing and input sent for later analysis
                    t_sent = time.perf_counter()
                    self.measured_input.append((t_sent, u_val))
                except:
                    pass

                # Update UI periodically (every 10 samples)
                if i % 10 == 0:
                    prog = (i / len(self.input_u)) * 100
                    self.root.after(
                        0,
                        lambda p=prog: self.lbl_status.config(
                            text=f"Acquisition... {p:.1f}%"
                        ),
                    )
                    self.root.after(0, self.update_realtime_plot)

            # End of test: stop motors and process results
            self.motor_controller.set_goal_velocity_rad([0.0, 0.0])
            self.lbl_status.config(text="Acquisition complete. Processing...")
            self.is_running_id = False
            self.root.after(0, self.process_results)

        except Exception as e:
            print(e)
            self.is_running_id = False

    def update_realtime_plot(self):
        """Update real-time plot during test execution with measured data."""
        if not self.measured_input or not self.measured_output:
            return

        try:
            # Prepare time arrays relative to test start
            t_in = [x[0] - self.start_time for x in self.measured_input]
            u_in = [x[1] for x in self.measured_input]

            t_out = [x[0] - self.start_time for x in self.measured_output]
            y_out = [x[1] for x in self.measured_output]

            # Update line objects
            self.line_rt_input.set_data(t_in, u_in)
            self.line_rt_output.set_data(t_out, y_out)

            # Auto-scale axes to show all data
            self.ax_sig.relim()
            self.ax_sig.autoscale_view()
            self.ax_sig_twin.relim()
            self.ax_sig_twin.autoscale_view()

            self.canvas.draw()
        except:
            pass

    def process_results(self):
        """Process recorded data and compute frequency response (Bode diagram)."""
        self.lbl_status.config(text="Processing data...")
        self.root.update()

        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_gen.config(state="normal")
        self.btn_save.config(state="normal")

        if len(self.measured_input) == 0 or len(self.measured_output) == 0:
            messagebox.showwarning("Warning", "No measurement data recorded.")
            return

        t_in = np.array([x[0] for x in self.measured_input])
        u_in = np.array([x[1] for x in self.measured_input])

        t_out = np.array([x[0] for x in self.measured_output])
        y_out = np.array([x[1] for x in self.measured_output])

        t0 = t_in[0]
        t_in -= t0
        t_out -= t0

        try:
            if t_out[-1] < t_in[-1] - 1.0:
                print("Warning: Output measurement stopped early.")

            # Resample output to match input time grid using linear interpolation
            interp_func = interp1d(
                t_out, y_out, kind="linear", fill_value="extrapolate"
            )
            y_resampled = interp_func(t_in)

            # Save time domain data
            self.time_data = (t_in, u_in, y_resampled)

            # --- FINAL TIME DOMAIN PLOT (RAW DATA) ---
            self.ax_sig.clear()
            self.ax_sig_twin.remove()
            self.ax_sig_twin = self.ax_sig.twinx()

            self.ax_sig.set_title("Time Domain Validation: Real Input vs Real Output")

            color = "tab:blue"
            self.ax_sig.set_xlabel("Time (s)")
            self.ax_sig.set_ylabel("Velocity Input (rad/s)", color=color)
            self.ax_sig.plot(t_in, u_in, color=color, label="Measured Input")
            self.ax_sig.tick_params(axis="y", labelcolor=color)

            color = "tab:red"
            self.ax_sig_twin.set_ylabel("Yaw Output (deg)", color=color)
            # Display raw data only - no detrending for visualization
            self.ax_sig_twin.plot(
                t_in, y_resampled, color=color, linestyle="--", label="Measured Output"
            )
            self.ax_sig_twin.tick_params(axis="y", labelcolor=color)

            # --- SPECTRAL ANALYSIS ---
            fs = self.fs_control

            # CRITICAL: For low frequencies (0.1 rad/s ~ 0.016 Hz), long window needed.
            # Using full signal length (Periodogram) maximizes frequency resolution
            # at expense of variance. Averaging would destroy low frequency content.
            nperseg = len(u_in)

            # Calculate Cross Spectral Density matrices
            # detrend='constant' removes DC bias from FFT
            f_u, P_uu = signal.csd(
                u_in, u_in, fs=fs, nperseg=nperseg, window="boxcar", detrend="constant"
            )
            f_y, P_yu = signal.csd(
                u_in,
                y_resampled,
                fs=fs,
                nperseg=nperseg,
                window="boxcar",
                detrend="constant",
            )
            _, P_yy = signal.csd(
                y_resampled,
                y_resampled,
                fs=fs,
                nperseg=nperseg,
                window="boxcar",
                detrend="constant",
            )

            # H1 transfer function estimator: H1 = P_yu / P_uu
            H1 = P_yu / P_uu
            # Coherence: measures quality of estimate
            Cxy = (np.abs(P_yu) ** 2) / (P_uu * P_yy)

            # Convert to dB and degrees
            mag_db = 20 * np.log10(np.abs(H1))
            phase_rad = np.unwrap(np.angle(H1))
            phase_deg_unwrapped = np.degrees(phase_rad)

            # Save frequency response data
            w = f_u * 2 * np.pi
            self.freq_data = (w, mag_db, phase_deg_unwrapped)

            # --- PLOT BODE DIAGRAM ---

            # Frequency in rad/s for Bode
            w = f_u * 2 * np.pi

            # LIMIT: 0.1 to 3.0 rad/s (User requirement)
            valid_idx = (w > 0.09) & (w < 3.1)

            # Magnitude plot
            self.ax_mag.clear()
            self.ax_mag.semilogx(
                w[valid_idx], mag_db[valid_idx], "bo-", label="Magnitude", markersize=4
            )
            self.ax_mag.set_ylabel("Magnitude (dB)", color="b")
            self.ax_mag.grid(True, which="both", linestyle="-", alpha=0.6)
            self.ax_mag.set_title("Bode Diagram: Magnitude")
            self.ax_mag.set_xlim(0.1, 3.0)

            # Phase plot
            self.ax_phase.clear()
            self.ax_phase.semilogx(
                w[valid_idx],
                phase_deg_unwrapped[valid_idx],
                "ro-",
                label="Phase",
                markersize=4,
            )
            self.ax_phase.set_ylabel("Phase (deg)", color="r")
            self.ax_phase.set_xlabel("Frequency (rad/s)")
            self.ax_phase.grid(True, which="both", linestyle="-", alpha=0.6)
            self.ax_phase.set_title("Bode Diagram: Phase")
            self.ax_phase.set_xlim(0.1, 3.0)

            self.fig.tight_layout()
            self.canvas.draw()

            self.lbl_status.config(
                text=f"Results displayed. (Freq Res: {2*np.pi*fs/nperseg:.3f} rad/s)"
            )

        except Exception as e:
            print(f"Analysis Error: {e}")
            messagebox.showerror("Analysis Error", str(e))

    def on_closing(self):
        """Handle application window close event."""
        self.disconnect()
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = WidebandIDApp(root)
    root.mainloop()
