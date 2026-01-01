import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import minimize, least_squares
import cmath

class TFEstimatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Estimateur de Fonction de Transfert")
        self.root.geometry("1400x900")
        
        # Data
        self.df = None
        self.freqs = None
        self.mag_db = None
        self.phase_deg = None
        self.complex_resp = None
        
        # UI
        self.create_widgets()
        
    def create_widgets(self):
        # Control Panel
        frame_ctrl = ttk.Frame(self.root, padding="10")
        frame_ctrl.pack(fill="x")
        
        ttk.Button(frame_ctrl, text="Charger CSV Bode", command=self.load_csv).pack(side="left", padx=5)
        
        ttk.Label(frame_ctrl, text="Bande (rad/s) :").pack(side="left", padx=5)
        self.ent_wmin = ttk.Entry(frame_ctrl, width=5)
        self.ent_wmin.insert(0, "0.1")
        self.ent_wmin.pack(side="left")
        ttk.Label(frame_ctrl, text="-").pack(side="left")
        self.ent_wmax = ttk.Entry(frame_ctrl, width=5)
        self.ent_wmax.insert(0, "3.0")
        self.ent_wmax.pack(side="left")
        
        ttk.Separator(frame_ctrl, orient="vertical").pack(side="left", padx=10, fill="y")
        
        ttk.Label(frame_ctrl, text="Ordre (Pôles/Zéros) :").pack(side="left", padx=5)
        self.ent_nb_poles = ttk.Entry(frame_ctrl, width=3)
        self.ent_nb_poles.insert(0, "2")
        self.ent_nb_poles.pack(side="left")
        ttk.Label(frame_ctrl, text="/").pack(side="left")
        self.ent_nb_zeros = ttk.Entry(frame_ctrl, width=3)
        self.ent_nb_zeros.insert(0, "0")
        self.ent_nb_zeros.pack(side="left")
        
        ttk.Separator(frame_ctrl, orient="vertical").pack(side="left", padx=10, fill="y")
        
        ttk.Label(frame_ctrl, text="Filtre K (Retard) :").pack(side="left", padx=5)
        self.ent_k_filter = ttk.Entry(frame_ctrl, width=4)
        self.ent_k_filter.insert(0, "5")
        self.ent_k_filter.pack(side="left")
        
        self.var_integrator = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame_ctrl, text="Forcer Intégrateur (1/s)", variable=self.var_integrator).pack(side="left", padx=10)
        
        ttk.Button(frame_ctrl, text="Estimer (Manuel)", command=self.estimate_tf).pack(side="left", padx=10)
        
        ttk.Button(frame_ctrl, text="Auto-Fit (Chercher le meilleur ordre)", command=self.auto_fit).pack(side="left", padx=10)
        
        # Plots
        self.fig, (self.ax_mag, self.ax_phase, self.ax_pz) = plt.subplots(3, 1, figsize=(10, 10))
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Result Text
        self.txt_res = tk.Text(self.root, height=8)
        self.txt_res.pack(fill="x", padx=10, pady=5)
        
    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path: return
        
        try:
            self.df = pd.read_csv(path)
            # Check cols
            req_cols = ["Omega_rad_s", "Magnitude_dB", "Phase_deg"]
            if not all(col in self.df.columns for col in req_cols):
                messagebox.showerror("Erreur", "Colonnes manquantes. Attendu: Omega_rad_s, Magnitude_dB, Phase_deg")
                return
            
            self.plot_raw()
            messagebox.showinfo("Info", "Fichier chargé.")
            
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            
    def plot_raw(self):
        if self.df is None: return
        
        w = self.df["Omega_rad_s"]
        mag = self.df["Magnitude_dB"]
        phase = self.df["Phase_deg"]
        
        # Filter by Band if possible
        try:
            w_min = float(self.ent_wmin.get())
            w_max = float(self.ent_wmax.get())
            mask = (w >= w_min) & (w <= w_max)
            w = w[mask]
            mag = mag[mask]
            phase = phase[mask]
        except: pass
        
        self.ax_mag.clear()
        self.ax_mag.semilogx(w, mag, 'b.', label='Mesure', alpha=0.3)
        self.ax_mag.set_ylabel("Gain (dB)")
        self.ax_mag.grid(True, which="both")
        self.ax_mag.legend()
        
        self.ax_phase.clear()
        self.ax_phase.semilogx(w, phase, 'r.', label='Mesure', alpha=0.3)
        self.ax_phase.set_ylabel("Phase (deg)")
        self.ax_phase.grid(True, which="both")
        
        self.canvas.draw()
        
    def get_data_in_band(self):
        try:
            w_min = float(self.ent_wmin.get())
            w_max = float(self.ent_wmax.get())
            
            mask = (self.df["Omega_rad_s"] >= w_min) & (self.df["Omega_rad_s"] <= w_max)
            data = self.df[mask]
            
            w = data["Omega_rad_s"].values
            mag = data["Magnitude_dB"].values
            phase = data["Phase_deg"].values
            
            # Complex response
            # H = 10^(dB/20) * exp(j * deg * pi/180)
            H = (10**(mag/20.0)) * np.exp(1j * np.radians(phase))
            
            return w, H
        except Exception as e:
            messagebox.showerror("Erreur Data", str(e))
            return None, None

    def model_resp(self, w, params, n_p, n_z, integrator=False, delay=0.0):
        # Params: [Gain, z1, z2..., p1, p2...]
        # Assume form: K * (s-z1)... / (s-p1)...
        # Wait, fitting real coefficients is better for stability than raw poles/zeros usually.
        # Form: K * (b0 s^m + ... + 1) / (a0 s^n + ... + 1)
        # Or K * (s^m + ... ) / (s^n + ...)
        
        # Let's use polynomial coefficients form:
        # Num = b_m s^m + ... + b_0
        # Den = s^n + a_{n-1} s^{n-1} + ... + a_0
        # If integrator: Den = s(s^n + ... + a_0) -> Den has factor s.
        
        # Mapping params to coeffs:
        # Params vector size: (n_z + 1) + (n_p)  if Den monic. 
        # Actually Gain is usually separate.
        
        # Let's stick to standard scipy signal convention: num, den
        # We need to enforce real coefficients.
        
        # Pack params:
        # [b_mz, ..., b_0, a_{np-1}, ..., a_0] 
        # (Assuming monic denominator highest power)
        
        # Extract Num coeffs
        # Num order n_z. Coeffs: n_z+1.
        b = params[:n_z+1]
        
        # Extract Den coeffs (monic highest)
        # Den order n_p. we need n_p coeffs for lower terms.
        # s^n + a_{n-1}s^{n-1} + ... a_0
        a_rest = params[n_z+1:]
        a = np.concatenate(([1.0], a_rest))
        
        s = 1j * w
        
        if integrator:
            # Multiply Den by s
            # poly mul: equivalent to shifting coeffs or appending 0
            a = np.concatenate((a, [0.0]))
            
        N = np.polyval(b, s)
        D = np.polyval(a, s)
        
        # Plant Transfer Function
        H_plant = N / D
        
        # Add Delay: e^(-j * w * delay)
        # delay in seconds
        H_total = H_plant * np.exp(-1j * w * delay)
        
        return H_total, b, a

    def cost_func(self, params, w, H_meas, n_p, n_z, integrator, delay):
        H_est, _, _ = self.model_resp(w, params, n_p, n_z, integrator, delay)
        # Logarithmic error usually better for Bode
        # Error in log magnitude + Error in phase (rad)
        
        err_mag = np.log10(np.abs(H_est)) - np.log10(np.abs(H_meas))
        # Phase diff in complex plane to avoid wrapping issues:
        # dist = |H_est - H_meas|^2 / |H_meas|^2?
        # Or standard vector fitting weight: |H_est - H_meas| / |H_meas|
        
        # Robust cost:
        # sum of squared relative error
        res = (H_est - H_meas) / np.abs(H_meas) # Relative vector error
        return np.sum(np.abs(res)**2)

    def fit_model(self, w, H_meas, n_p, n_z, integrator, delay):
        # Init params
        # b: usually just Gain [1, 0, 0...]
        # a: [0, 0... 1] (stable poles?)
        
        # Heuristic init:
        # Gain K approx median magnitude
        K = np.median(np.abs(H_meas))
        # If integrator, remove 1/w effect from K estimation?
        if integrator:
             K = np.median(np.abs(H_meas) * w)
             
        # Init b: [0...0, K] (Constant term = K) ???
        # If n_z=0, b has 1 elem.
        b0 = np.zeros(n_z+1)
        b0[-1] = K
        
        # Init a: [0...0] ??
        # Choosing stable poles? e.g. (s+1)^n
        # (s+1) -> s+1 -> coeffs [1, 1] -> a_rest [1]
        # (s+10)... 
        # Random small initial coeffs usually work with Levenberg-Marquardt
        a0 = np.zeros(n_p) 
        # Let's set some damping
        if n_p > 0: a0[-1] = 1.0 # s^n + ... + 1
        
        x0 = np.concatenate((b0, a0))
        
        res = minimize(self.cost_func, x0, args=(w, H_meas, n_p, n_z, integrator, delay), method='L-BFGS-B') # or Nelder-Mead
        
        return res.x, res.fun

    def estimate_tf(self):
        w, H_meas = self.get_data_in_band()
        if w is None: return
        
        try:
            n_p = int(self.ent_nb_poles.get())
            n_z = int(self.ent_nb_zeros.get())
            integrator = self.var_integrator.get()
            
            # Calculate delay
            try:
                k = int(self.ent_k_filter.get())
            except: k = 5
            fs = 50.0 # Hz (Hardcoded assumption matching loop)
            delay = 0.0
            if k > 1:
                # Delay tau = (k-1)/2 samples * dt
                # dt = 1/fs
                delay = ((k - 1) / 2) * (1.0 / fs)
            
            params, cost = self.fit_model(w, H_meas, n_p, n_z, integrator, delay)
            self.display_results(w, H_meas, params, n_p, n_z, integrator, delay)
            
        except Exception as e:
            messagebox.showerror("Erreur Est", str(e))

    def auto_fit(self):
        w, H_meas = self.get_data_in_band()
        if w is None: return
        
        try:
            w_max_band = float(self.ent_wmax.get())
        except: w_max_band = 100.0
        
        best_score = float('inf')
        best_cfg = None
        
        self.txt_res.delete(1.0, tk.END)
        self.txt_res.insert(tk.END, "Scanning models (Auto-Fit smart)...\n")
        print("Scanning models...")
        self.root.update()
        
        self.root.update()
        
        integrator = self.var_integrator.get()
        
        # Calculate delay
        try:
            k = int(self.ent_k_filter.get())
        except: k = 5
        fs = 50.0
        delay = 0.0
        if k > 1:
            delay = ((k - 1) / 2) * (1.0 / fs)
        
        # Scan low orders first
        configs = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (1, 1), (2, 1), (3, 1),
            (2, 2)
        ]
        
        for p, z in configs:
            if p < z: continue # Improper TF
            
            try:
                params, raw_cost = self.fit_model(w, H_meas, p, z, integrator, delay)
                
                # Analyze Poles/Zeros location
                _, b_poly, a_poly = self.model_resp(w, params, p, z, integrator, delay) # Get polys
                poles = np.roots(a_poly)
                zeros = np.roots(b_poly)
                
                # Check for Out-of-Band artifacts
                # A pole/zero is "irrelevant" if its magnitude is >> w_max_band (e.g. 5x)
                # or if it is very close to 0 (unless we wanted an integrator, but here we check additional poles)
                
                max_relevant_freq = 5.0 * w_max_band
                
                has_oob = False
                for pole in poles:
                    if np.abs(pole) > max_relevant_freq: has_oob = True
                    
                for zero in zeros:
                    if np.abs(zero) > max_relevant_freq: has_oob = True
                
                # Check Stability (No Energy Creation)
                has_unstable = False
                for pole in poles:
                    if np.real(pole) >= 1e-6: # Tolerance for numerical noise
                         has_unstable = True

                # Calculate Score (BIC-like)
                n_params = p + z + 1
                score = raw_cost * (1 + 0.2 * n_params)
                
                status = "OK"
                if has_oob: 
                    score = score * 50.0 # Banish High Freq
                    status = "OOB"
                
                if has_unstable:
                    score = score * 1000.0 # BANISH Unstable
                    status = "Unstable"
                
                self.txt_res.insert(tk.END, f"Order P={p} Z={z} | Cost={raw_cost:.4f} | Score={score:.4f} [{status}]\n")
                print(f"Order P={p} Z={z} | Cost={raw_cost:.4f} | Score={score:.4f} [{status}]")
                
                if score < best_score:
                    best_score = score
                    best_cfg = (params, p, z)
            except Exception as e: 
                print(f"Err {p}/{z}: {e}")
            
        if best_cfg:
            self.txt_res.insert(tk.END, f"\n>>> BEST MODEL: P={best_cfg[1]} Z={best_cfg[2]}\n")
            print(f">>> BEST MODEL: P={best_cfg[1]} Z={best_cfg[2]}")
            self.display_results(w, H_meas, best_cfg[0], best_cfg[1], best_cfg[2], integrator, delay)

    def display_results(self, w, H_meas, params, n_p, n_z, integrator, delay):
        H_est, b, a = self.model_resp(w, params, n_p, n_z, integrator, delay)
        
        # Plot Plot
        self.plot_raw() # Refresh raw
        
        # Mag
        mag_est = 20 * np.log10(np.abs(H_est))
        self.ax_mag.semilogx(w, mag_est, 'r-', linewidth=2, label='Modèle')
        self.ax_mag.legend()
        
        # Phase
        phase_est = np.degrees(np.unwrap(np.angle(H_est)))
        self.ax_phase.semilogx(w, phase_est, 'r-', linewidth=2, label='Modèle')
        self.ax_phase.legend()
        
        # Poles/Zeros
        zeros = np.roots(b)
        poles = np.roots(a)
        
        self.ax_pz.clear()
        self.ax_pz.set_title("Carte des Pôles et Zéros")
        self.ax_pz.set_xlabel("Réel")
        self.ax_pz.set_ylabel("Imaginaire")
        self.ax_pz.grid(True)
        self.ax_pz.axhline(0, color='k', lw=1)
        self.ax_pz.axvline(0, color='k', lw=1)
        
        self.ax_pz.plot(np.real(poles), np.imag(poles), 'x', markersize=10, label='Pôles')
        self.ax_pz.plot(np.real(zeros), np.imag(zeros), 'o', markersize=10, fillstyle='none', label='Zéros')
        self.ax_pz.legend()
        
        self.canvas.draw()
        
        # Text Output
        res_str = f"--- Estimated Transfer Function ---\n"
        
        # Format H(s) string
        num_str = " + ".join([f"{c:.4f}s^{len(b)-1-i}" for i, c in enumerate(b)])
        den_str = " + ".join([f"{c:.4f}s^{len(a)-1-i}" for i, c in enumerate(a)])
        
        # Clean up s^0 and s^1
        num_str = num_str.replace("s^0", "").replace("s^1 ", "s ")
        den_str = den_str.replace("s^0", "").replace("s^1 ", "s ")
        
        res_str += f"H(s) = \n( {num_str} ) \n-------------------\n( {den_str} )\n"
        
        res_str += f"\nNum coeffs: {np.array2string(b, precision=4)}\n"
        res_str += f"Den coeffs: {np.array2string(a, precision=4)}\n"
        res_str += "\nPoles:\n" + str(poles) + "\n"
        res_str += "Zeros:\n" + str(zeros) + "\n"
        res_str += f"Relative Degree (P - Z): {len(a)-len(b)} (Physical delay/lag)\n"
        res_str += f" Modeled Filter Delay: {delay*1000:.1f} ms (k={int(delay*50*2+1) if delay>0 else 1})\n"
        
        # Bandwidth check
        w_max_band = float(self.ent_wmax.get())
        
        res_str += "\n--- Pertinence Check ---\n"
        for p in poles:
            freq = np.abs(p)
            if freq > 10 * w_max_band:
                res_str += f"[INFO] Pole at {p:.2f} is > 10x Bandwidth (High freq artifact?)\n"
            elif freq < 0.1 * float(self.ent_wmin.get()) and abs(freq) > 1e-6:
                res_str += f"[INFO] Pole at {p:.2f} is very slow/DC (Integrator?)\n"
                
        print(res_str) # Force print to console for debug
        self.txt_res.delete(1.0, tk.END)
        self.txt_res.insert(tk.END, res_str)

if __name__ == "__main__":
    root = tk.Tk()
    app = TFEstimatorApp(root)
    root.mainloop()
