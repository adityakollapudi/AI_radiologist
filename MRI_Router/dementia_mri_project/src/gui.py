"""Modern Cascade GUI Window for Brain MRI Diagnosis:
Stage 1: Brain Tumor Screening (Custom CNN with Multi-Layer Grad-CAM).
Stage 2: Reflex Dementia Assessment (EfficientNetV2-S) - triggered ONLY if Stage 1 is Normal (No Tumor).
         If a Brain Tumor is detected, Dementia assessment is bypassed as per cascade protocol.

Thread-safe architecture:
- Non-blocking background inference worker with queue.Queue communication.
- Matplotlib 'Agg' headless backend to prevent Tcl/Tkinter thread collisions.
"""
import hashlib
import os
import queue
import re
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("Agg")

from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import DementiaPredictor
from src.tumor_detector import BrainTumorClassifier, find_tumor_model_path


class CascadeMRIApp(tk.Tk):
    def __init__(self, initial_image=None, checkpoint=None):
        super().__init__()
        self.title("Cascade Brain MRI Diagnostic System (Tumor Screening & Reflex Dementia Assessment)")
        self.geometry("1060x980")
        self.minsize(920, 780)
        self.configure(bg="#0f172a")

        self.checkpoint_path = Path(checkpoint) if checkpoint else ROOT / "checkpoints" / "best_model.pth"
        self.tumor_detector = None
        self.predictor = None
        self.current_image_path = initial_image
        self.img_input_tk = None
        self.img_gradcam_tk = None
        self.msg_queue = queue.Queue()

        self.active_tumor_gradcam = None
        self.active_tumor_panel = None
        self.active_dementia_gradcam = None
        self.current_shown_gradcam = "tumor"

        self._init_styles()
        self._build_ui()

        # Initialize models in background
        self._load_models()

        # Start thread-safe queue poller in the main GUI thread
        self.after(50, self._poll_queue)

        # If an initial image was provided, load and run it
        if self.current_image_path:
            clean_path = str(self.current_image_path).strip().strip('"').strip("'")
            if Path(clean_path).exists():
                self.image_entry.delete(0, tk.END)
                self.image_entry.insert(0, clean_path)
                self.after(300, self.run_prediction)

    def _init_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TProgressbar", thickness=14, troughcolor="#334155", background="#38bdf8")
        style.configure("Tumor.Horizontal.TProgressbar", thickness=14, troughcolor="#334155", background="#ef4444")

    def _build_ui(self):
        # Top Address & Control Bar
        control_frame = tk.Frame(self, bg="#1e293b", padx=16, pady=12, highlightthickness=1, highlightbackground="#334155")
        control_frame.pack(fill="x", side="top", padx=16, pady=(12, 6))

        tk.Label(control_frame, text="Input MRI Address / Path:", font=("Segoe UI", 10, "bold"),
                 fg="#94a3b8", bg="#1e293b").pack(side="left", padx=(0, 8))

        self.image_entry = tk.Entry(control_frame, font=("Segoe UI", 10), bg="#0f172a", fg="#f8fafc",
                                    insertbackground="#38bdf8", relief="flat", highlightthickness=1, highlightbackground="#475569")
        self.image_entry.pack(side="left", fill="x", expand=True, padx=4, ipady=4)
        self.image_entry.bind("<Return>", lambda e: self.run_prediction())

        browse_btn = tk.Button(control_frame, text="Browse Image...", font=("Segoe UI", 9, "bold"),
                               bg="#334155", fg="#f8fafc", activebackground="#475569", activeforeground="#ffffff",
                               relief="flat", padx=12, pady=4, cursor="hand2", command=self.browse_file)
        browse_btn.pack(side="left", padx=6)

        self.analyze_btn = tk.Button(control_frame, text="Run Cascade Diagnosis", font=("Segoe UI", 9, "bold"),
                                     bg="#2563eb", fg="#ffffff", activebackground="#1d4ed8", activeforeground="#ffffff",
                                     relief="flat", padx=16, pady=4, cursor="hand2", command=self.run_prediction)
        self.analyze_btn.pack(side="left", padx=4)

        # Main scrollable canvas container
        canvas_container = tk.Frame(self, bg="#0f172a")
        canvas_container.pack(fill="both", expand=True, padx=16, pady=6)

        self.canvas = tk.Canvas(canvas_container, bg="#0f172a", highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#0f172a")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas_window_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.canvas_window_id, width=e.width))
        self.canvas.configure(xscrollcommand=None, yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.bind_all("<MouseWheel>", lambda event: self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

        # --- HEADING ---
        heading_frame = tk.Frame(self.scrollable_frame, bg="#0f172a", pady=4)
        heading_frame.pack(fill="x")

        tk.Label(heading_frame, text="Cascade Brain MRI Diagnostic Analysis",
                 font=("Segoe UI", 20, "bold"), fg="#f8fafc", bg="#0f172a").pack(anchor="w")
        tk.Label(heading_frame,
                 text="Workflow: Stage 1 Brain Tumor Screening  ➔  If Normal, reflex Stage 2 Dementia Assessment (Bypassed if Tumor Detected)",
                 font=("Segoe UI", 10), fg="#38bdf8", bg="#0f172a").pack(anchor="w", pady=(2, 6))

        # --- OVERALL CONSOLIDATED DIAGNOSIS BANNER ---
        self.overall_banner = tk.Frame(self.scrollable_frame, bg="#1e293b", padx=16, pady=10,
                                       highlightthickness=1, highlightbackground="#334155")
        self.overall_banner.pack(fill="x", pady=(2, 6))

        tk.Label(self.overall_banner, text="Consolidated Diagnosis:", font=("Segoe UI", 10, "bold"),
                 fg="#94a3b8", bg="#1e293b").pack(side="left", padx=(0, 10))
        self.lbl_overall = tk.Label(self.overall_banner, text="Awaiting Image Selection...", font=("Segoe UI", 13, "bold"),
                                    fg="#cbd5e1", bg="#1e293b")
        self.lbl_overall.pack(side="left")

        # --- 2 STAGES GRID (STAGE 1: BRAIN TUMOR & STAGE 2: DEMENTIA) ---
        stages_container = tk.Frame(self.scrollable_frame, bg="#0f172a")
        stages_container.pack(fill="x", pady=4)

        # Card 1: Stage 1 Brain Tumor Screening
        self.stage1_card = tk.Frame(stages_container, bg="#1e293b", padx=16, pady=12,
                                    highlightthickness=1, highlightbackground="#334155")
        self.stage1_card.pack(side="left", fill="both", expand=True, padx=(0, 6))

        tk.Label(self.stage1_card, text="STAGE 1: BRAIN TUMOR SCREENING", font=("Segoe UI", 11, "bold"),
                 fg="#38bdf8", bg="#1e293b").pack(anchor="w")
        tk.Label(self.stage1_card, text="Primary Screener (Custom 5-Layer CNN with Multi-Layer Grad-CAM)", font=("Segoe UI", 8),
                 fg="#64748b", bg="#1e293b").pack(anchor="w", pady=(0, 6))

        # Tumor Result line
        row_t_res = tk.Frame(self.stage1_card, bg="#1e293b")
        row_t_res.pack(fill="x", pady=2)
        tk.Label(row_t_res, text="Tumor Result:", font=("Segoe UI", 10, "bold"), fg="#94a3b8", bg="#1e293b", width=14, anchor="w").pack(side="left")
        self.lbl_tumor_res = tk.Label(row_t_res, text="--", font=("Segoe UI", 12, "bold"), fg="#f8fafc", bg="#1e293b")
        self.lbl_tumor_res.pack(side="left")

        # Tumor Probability line
        row_t_prob = tk.Frame(self.stage1_card, bg="#1e293b")
        row_t_prob.pack(fill="x", pady=2)
        tk.Label(row_t_prob, text="Probability:", font=("Segoe UI", 10, "bold"), fg="#94a3b8", bg="#1e293b", width=14, anchor="w").pack(side="left")
        self.lbl_tumor_prob = tk.Label(row_t_prob, text="--%", font=("Segoe UI", 11, "bold"), fg="#fbbf24", bg="#1e293b")
        self.lbl_tumor_prob.pack(side="left")

        # Tumor Finding / Status line
        row_t_stat = tk.Frame(self.stage1_card, bg="#1e293b")
        row_t_stat.pack(fill="x", pady=4)
        tk.Label(row_t_stat, text="Finding:", font=("Segoe UI", 10, "bold"), fg="#94a3b8", bg="#1e293b", width=14, anchor="w").pack(side="left")
        self.lbl_tumor_status = tk.Label(row_t_stat, text="--", font=("Segoe UI", 10, "bold"), fg="#94a3b8", bg="#1e293b", padx=8, pady=2)
        self.lbl_tumor_status.pack(side="left")

        # Card 2: Stage 2 Reflex Dementia Assessment
        self.stage2_card = tk.Frame(stages_container, bg="#1e293b", padx=16, pady=12,
                                    highlightthickness=1, highlightbackground="#334155")
        self.stage2_card.pack(side="right", fill="both", expand=True, padx=(6, 0))

        tk.Label(self.stage2_card, text="STAGE 2: REFLEX DEMENTIA ASSESSMENT", font=("Segoe UI", 11, "bold"),
                 fg="#a855f7", bg="#1e293b").pack(anchor="w")
        tk.Label(self.stage2_card, text="Secondary Screener (Activated ONLY if Brain Tumor is Normal)", font=("Segoe UI", 8),
                 fg="#64748b", bg="#1e293b").pack(anchor="w", pady=(0, 6))

        # Dementia trigger status line
        row_d_state = tk.Frame(self.stage2_card, bg="#1e293b")
        row_d_state.pack(fill="x", pady=2)
        tk.Label(row_d_state, text="Trigger Status:", font=("Segoe UI", 10, "bold"), fg="#94a3b8", bg="#1e293b", width=14, anchor="w").pack(side="left")
        self.lbl_dementia_trigger = tk.Label(row_d_state, text="Standby", font=("Segoe UI", 10, "bold"), fg="#94a3b8", bg="#1e293b", padx=6, pady=1)
        self.lbl_dementia_trigger.pack(side="left")

        # Category line
        row_d = tk.Frame(self.stage2_card, bg="#1e293b")
        row_d.pack(fill="x", pady=2)
        tk.Label(row_d, text="Category:", font=("Segoe UI", 10, "bold"), fg="#94a3b8", bg="#1e293b", width=14, anchor="w").pack(side="left")
        self.lbl_disease = tk.Label(row_d, text="--", font=("Segoe UI", 13, "bold"), fg="#f8fafc", bg="#1e293b")
        self.lbl_disease.pack(side="left")

        # Confidence line
        row_c = tk.Frame(self.stage2_card, bg="#1e293b")
        row_c.pack(fill="x", pady=2)
        tk.Label(row_c, text="Confidence:", font=("Segoe UI", 10, "bold"), fg="#94a3b8", bg="#1e293b", width=14, anchor="w").pack(side="left")
        self.lbl_conf = tk.Label(row_c, text="--%", font=("Segoe UI", 11, "bold"), fg="#fbbf24", bg="#1e293b")
        self.lbl_conf.pack(side="left")

        # Dementia Finding line
        row_s = tk.Frame(self.stage2_card, bg="#1e293b")
        row_s.pack(fill="x", pady=4)
        tk.Label(row_s, text="Finding:", font=("Segoe UI", 10, "bold"), fg="#94a3b8", bg="#1e293b", width=14, anchor="w").pack(side="left")
        self.lbl_dementia_status = tk.Label(row_s, text="--", font=("Segoe UI", 10, "bold"), fg="#94a3b8", bg="#1e293b", padx=8, pady=2)
        self.lbl_dementia_status.pack(side="left")

        # Quality warning banner (hidden by default)
        self.warn_frame = tk.Frame(self.scrollable_frame, bg="#451a03", padx=12, pady=6,
                                   highlightthickness=1, highlightbackground="#d97706")
        self.lbl_warning = tk.Label(self.warn_frame, text="", font=("Segoe UI", 9), fg="#fef3c7", bg="#451a03")
        self.lbl_warning.pack(anchor="w")

        # --- IMAGES SECTION (INPUT SCAN & GRAD-CAM HEATMAP) ---
        images_section = tk.Frame(self.scrollable_frame, bg="#0f172a", pady=8)
        images_section.pack(fill="x")

        # Left Card: Input MRI Image (Given)
        self.card_orig = tk.Frame(images_section, bg="#1e293b", padx=16, pady=12,
                                  highlightthickness=1, highlightbackground="#334155")
        self.card_orig.pack(side="left", fill="both", expand=True, padx=(0, 6))

        tk.Label(self.card_orig, text="Input MRI Slice (Given)", font=("Segoe UI", 11, "bold"),
                 fg="#f8fafc", bg="#1e293b").pack(anchor="w")
        tk.Label(self.card_orig, text="Original axial brain scan slice provided", font=("Segoe UI", 8),
                 fg="#94a3b8", bg="#1e293b").pack(anchor="w", pady=(0, 6))

        orig_box = tk.Frame(self.card_orig, width=280, height=280, bg="#0b1120",
                            highlightthickness=1, highlightbackground="#334155")
        orig_box.pack(pady=4)
        orig_box.pack_propagate(False)

        self.canvas_orig = tk.Label(orig_box, text="No Image Loaded\n\nClick 'Browse Image...'\nor enter path above",
                                    bg="#0b1120", fg="#64748b", font=("Segoe UI", 10), justify="center")
        self.canvas_orig.pack(fill="both", expand=True)

        # Right Card: Explainable Grad-CAM Heatmap
        self.card_grad = tk.Frame(images_section, bg="#1e293b", padx=16, pady=12,
                                  highlightthickness=1, highlightbackground="#334155")
        self.card_grad.pack(side="right", fill="both", expand=True, padx=(6, 0))

        grad_header = tk.Frame(self.card_grad, bg="#1e293b")
        grad_header.pack(fill="x")

        tk.Label(grad_header, text="Explainable Grad-CAM Attention Map", font=("Segoe UI", 11, "bold"),
                 fg="#f8fafc", bg="#1e293b").pack(side="left")

        # Action Buttons
        self.btn_open_panel = tk.Button(grad_header, text="Open 4-Panel View", font=("Segoe UI", 8, "bold"),
                                        bg="#1e40af", fg="#ffffff", relief="flat", padx=6, pady=1,
                                        cursor="hand2", command=self._open_panel_view)

        self.btn_toggle_cam = tk.Button(grad_header, text="View Dementia CAM", font=("Segoe UI", 8, "bold"),
                                        bg="#334155", fg="#f8fafc", relief="flat", padx=6, pady=1,
                                        cursor="hand2", command=self._toggle_gradcam_view)

        self.lbl_cam_desc = tk.Label(self.card_grad, text="Gradient-weighted class attribution localization", font=("Segoe UI", 8),
                                     fg="#38bdf8", bg="#1e293b")
        self.lbl_cam_desc.pack(anchor="w", pady=(0, 6))

        grad_box = tk.Frame(self.card_grad, width=280, height=280, bg="#0b1120",
                            highlightthickness=1, highlightbackground="#334155")
        grad_box.pack(pady=4)
        grad_box.pack_propagate(False)

        self.canvas_grad = tk.Label(grad_box, text="Grad-CAM Attention Map\n\nWill appear here\nafter analysis",
                                    bg="#0b1120", fg="#64748b", font=("Segoe UI", 10), justify="center")
        self.canvas_grad.pack(fill="both", expand=True)

        # --- IMPORTANT METRICS SECTION ---
        metrics_card = tk.Frame(self.scrollable_frame, bg="#1e293b", padx=20, pady=14,
                                highlightthickness=1, highlightbackground="#334155")
        metrics_card.pack(fill="x", pady=6)

        tk.Label(metrics_card, text="Detailed Diagnostic Probability Distributions",
                 font=("Segoe UI", 12, "bold"), fg="#f8fafc", bg="#1e293b").pack(anchor="w", pady=(0, 8))

        probs_frame = tk.Frame(metrics_card, bg="#1e293b")
        probs_frame.pack(fill="x", pady=2)

        # Stage 1 Brain Tumor Probability Row (Primary)
        row_tumor_bar = tk.Frame(probs_frame, bg="#1e293b")
        row_tumor_bar.pack(fill="x", pady=(2, 8))

        tk.Label(row_tumor_bar, text="Stage 1: Brain Tumor Prob", font=("Segoe UI", 9, "bold"), fg="#c084fc", bg="#1e293b", width=24, anchor="w").pack(side="left")
        self.pb_tumor = ttk.Progressbar(row_tumor_bar, orient="horizontal", mode="determinate", length=300, style="Tumor.Horizontal.TProgressbar")
        self.pb_tumor.pack(side="left", padx=8)
        self.lbl_tumor_meter = tk.Label(row_tumor_bar, text="--%", font=("Segoe UI", 9, "bold"), fg="#c084fc", bg="#1e293b", width=8, anchor="e")
        self.lbl_tumor_meter.pack(side="left")

        # Stage 2 Dementia Class probabilities grid
        self.prob_bars = {}
        self.prob_labels = {}
        classes = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]

        for cname in classes:
            row = tk.Frame(probs_frame, bg="#1e293b")
            row.pack(fill="x", pady=2)

            tk.Label(row, text=f"Stage 2: {cname}", font=("Segoe UI", 9), fg="#cbd5e1", bg="#1e293b", width=24, anchor="w").pack(side="left")

            pb = ttk.Progressbar(row, orient="horizontal", mode="determinate", length=300, style="TProgressbar")
            pb.pack(side="left", padx=8)
            self.prob_bars[cname] = pb

            lbl_val = tk.Label(row, text="--%", font=("Segoe UI", 9, "bold"), fg="#38bdf8", bg="#1e293b", width=8, anchor="e")
            lbl_val.pack(side="left")
            self.prob_labels[cname] = lbl_val

        # Diagnostic Details Grid
        details_grid = tk.Frame(metrics_card, bg="#1e293b", pady=8)
        details_grid.pack(fill="x")

        self.lbl_uncertainty = tk.Label(details_grid, text="Dementia Uncertainty: --", font=("Segoe UI", 9), fg="#94a3b8", bg="#1e293b")
        self.lbl_uncertainty.grid(row=0, column=0, sticky="w", padx=(0, 24), pady=2)

        self.lbl_latency = tk.Label(details_grid, text="Total Inference Latency: --", font=("Segoe UI", 9), fg="#94a3b8", bg="#1e293b")
        self.lbl_latency.grid(row=0, column=1, sticky="w", padx=12, pady=2)

        self.lbl_calibration = tk.Label(details_grid, text="Temperature Calibration: Active", font=("Segoe UI", 9), fg="#94a3b8", bg="#1e293b")
        self.lbl_calibration.grid(row=1, column=0, sticky="w", padx=(0, 24), pady=2)

        self.lbl_model_arch = tk.Label(details_grid, text="Models: Custom CNN + EfficientNetV2-S", font=("Segoe UI", 9), fg="#94a3b8", bg="#1e293b")
        self.lbl_model_arch.grid(row=1, column=1, sticky="w", padx=12, pady=2)

        # Protocol Disclaimer
        disclaimer_frame = tk.Frame(self.scrollable_frame, bg="#172554", padx=12, pady=8, highlightthickness=1, highlightbackground="#1e40af")
        disclaimer_frame.pack(fill="x", pady=(6, 12))

        tk.Label(disclaimer_frame,
                 text="CASCADE PROTOCOL: Stage 1 screens for Brain Tumors. When No Tumor is detected (Normal), Stage 2 automatically evaluates Dementia severity. "
                      "If a Brain Tumor is detected, Dementia screening is bypassed to prevent acute structural distortion from confounding subtle atrophy analysis. Research prototype; not a clinically validated standalone diagnostic device.",
                 font=("Segoe UI", 8, "italic"), fg="#93c5fd", bg="#172554", wraplength=980, justify="left").pack(anchor="w")

    def _load_models(self):
        """Loads both Brain Tumor and Dementia models."""
        try:
            self.tumor_detector = BrainTumorClassifier.get_instance()
        except Exception as e:
            print(f"Tumor detector warning: {e}")

        try:
            self.predictor = DementiaPredictor(self.checkpoint_path)
            self.lbl_calibration.config(text=f"Temperature Calibration: Active (T = {self.predictor.T:.3f})")
        except Exception as e:
            print(f"Dementia model warning: {e}")

    def browse_file(self):
        fpath = filedialog.askopenfilename(
            title="Select Brain MRI Image Slice",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.webp"), ("All files", "*.*")]
        )
        if fpath:
            clean_path = str(fpath).strip().strip('"').strip("'")
            self.image_entry.delete(0, tk.END)
            self.image_entry.insert(0, clean_path)
            self.run_prediction()

    def run_prediction(self):
        raw_text = self.image_entry.get().strip()
        clean_path = raw_text.strip('"').strip("'")

        if not clean_path:
            messagebox.showwarning("Input Required", "Please enter or browse for an MRI image address.")
            return

        p = Path(clean_path)
        if not p.exists() or not p.is_file():
            messagebox.showerror("File Not Found", f"The image path does not exist or is not a valid file:\n{clean_path}")
            return

        if self.tumor_detector is None:
            messagebox.showwarning("Initializing", "Tumor screening model is still loading. Please wait a moment.")
            return

        self.analyze_btn.config(state="disabled", text="Analyzing...")
        self.lbl_overall.config(text="Running Cascade Diagnostic Analysis...", fg="#38bdf8")
        self.lbl_tumor_res.config(text="Screening Stage 1...", fg="#38bdf8")
        self.lbl_tumor_prob.config(text="--%", fg="#fbbf24")
        self.lbl_tumor_status.config(text="Analyzing...", fg="#38bdf8", bg="#1e293b")
        self.lbl_dementia_trigger.config(text="Pending Stage 1...", fg="#94a3b8", bg="#1e293b")
        self.lbl_disease.config(text="--", fg="#f8fafc")
        self.lbl_conf.config(text="--%", fg="#fbbf24")
        self.lbl_dementia_status.config(text="--", fg="#94a3b8", bg="#1e293b")
        self.update_idletasks()

        # Run inference in worker thread
        threading.Thread(target=self._worker_cascade, args=(p,), daemon=True).start()

    def _worker_cascade(self, p):
        """Worker executing Brain Tumor screening first, then conditionally Dementia analysis."""
        import time
        t0 = time.time()
        try:
            stem_hash = hashlib.md5(p.stem.encode("utf-8")).hexdigest()[:8]
            safe_folder = f"{re.sub(r'[^a-zA-Z0-9_\-]', '_', p.stem)[:18]}_{stem_hash}"
            out_dir = ROOT / "predictions" / safe_folder
            out_dir.mkdir(parents=True, exist_ok=True)

            # Stage 1: Brain Tumor Screening
            if self.tumor_detector is None:
                self.tumor_detector = BrainTumorClassifier.get_instance()
            tumor_res = self.tumor_detector.predict(p, out_dir=out_dir)
            has_tumor = tumor_res["has_tumor"]

            dementia_res = None
            # Stage 2: Reflex Dementia assessment ONLY if Brain Tumor is Normal (No tumor)
            if not has_tumor:
                if self.predictor is None:
                    self.predictor = DementiaPredictor(self.checkpoint_path)
                dementia_res = self.predictor.predict(p, gradcam_dir=out_dir, mc_passes=0, multi_layer=True)

            total_ms = (time.time() - t0) * 1000

            self.msg_queue.put(("SUCCESS", (p, tumor_res, dementia_res, total_ms)))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.msg_queue.put(("ERROR", str(exc)))

    def _poll_queue(self):
        try:
            while True:
                msg_type, payload = self.msg_queue.get_nowait()
                if msg_type == "SUCCESS":
                    p, tumor_res, dementia_res, total_ms = payload
                    self._update_ui_cascade(p, tumor_res, dementia_res, total_ms)
                    self.analyze_btn.config(state="normal", text="Run Cascade Diagnosis")
                elif msg_type == "ERROR":
                    messagebox.showerror("Diagnostic Error", f"Error during analysis:\n{payload}")
                    self.lbl_overall.config(text="Analysis Error", fg="#ef4444")
                    self.analyze_btn.config(state="normal", text="Run Cascade Diagnosis")
        except queue.Empty:
            pass

        self.after(50, self._poll_queue)

    def _update_ui_cascade(self, img_path, t_res, d_res, total_ms):
        has_tumor = t_res["has_tumor"]
        tumor_prob = t_res["probability"] * 100.0

        # 1. Update Stage 1 Brain Tumor UI
        self.pb_tumor["value"] = tumor_prob
        self.lbl_tumor_meter.config(text=f"{tumor_prob:.2f}%")

        self.active_tumor_gradcam = t_res.get("gradcam_overlay")
        self.active_tumor_panel = t_res.get("gradcam_panel")

        if has_tumor:
            self.lbl_tumor_res.config(text="BRAIN TUMOR DETECTED (YES)", fg="#ef4444")
            self.lbl_tumor_prob.config(text=f"{tumor_prob:.2f}% (High Confidence)", fg="#ef4444")
            self.lbl_tumor_status.config(text="ABNORMAL (Brain Tumor Detected)", fg="#ef4444", bg="#450a0a")

            # Stage 2 Dementia was bypassed
            self.lbl_dementia_trigger.config(
                text="SKIPPED (Brain Tumor Detected)",
                fg="#94a3b8", bg="#1e293b"
            )
            self.lbl_disease.config(text="Reflex Check Bypassed", fg="#64748b")
            self.lbl_conf.config(text="N/A", fg="#64748b")
            self.lbl_dementia_status.config(text="BYPASSED (Gross Lesion Present)", fg="#64748b", bg="#1e293b")

            # Overall banner
            self.overall_banner.config(highlightbackground="#ef4444")
            self.lbl_overall.config(
                text=f"ABNORMAL: Brain Tumor Detected ({tumor_prob:.2f}%) | Dementia Assessment Bypassed",
                fg="#f87171"
            )

            # Reset dementia probability bars
            for cname in self.prob_bars:
                self.prob_bars[cname]["value"] = 0
                self.prob_labels[cname].config(text="--%")

            self.lbl_uncertainty.config(text="Dementia Uncertainty: N/A (Bypassed)")

            # Set tumor Grad-CAM as primary
            self.active_dementia_gradcam = None
            self.current_shown_gradcam = "tumor"
            self._set_gradcam_display(self.active_tumor_gradcam, "Multi-Layer Fused Brain Tumor Localization (5 Layers)")
            self.btn_toggle_cam.pack_forget()
            if self.active_tumor_panel:
                self.btn_open_panel.pack(side="right", padx=(0, 6))

        else:
            # Tumor is Normal -> Stage 2 Reflex Dementia check was executed
            self.lbl_tumor_res.config(text="NO BRAIN TUMOR (NORMAL BRAIN)", fg="#10b981")
            self.lbl_tumor_prob.config(text=f"{tumor_prob:.2f}% (Normal Brain)", fg="#10b981")
            self.lbl_tumor_status.config(text="NORMAL (No Tumor Detected)", fg="#10b981", bg="#064e3b")

            self.lbl_dementia_trigger.config(
                text="TRIGGERED (Tumor Normal)",
                fg="#38bdf8", bg="#0c4a6e"
            )

            disease = d_res["prediction"]
            conf = d_res["confidence"] * 100.0
            is_normal_dementia = (disease == "Non Demented")

            self.lbl_disease.config(text=disease, fg="#f8fafc")
            self.lbl_conf.config(text=f"{conf:.2f}% (Calibrated)", fg="#fbbf24")

            self.active_dementia_gradcam = d_res.get("gradcam")

            if is_normal_dementia:
                self.lbl_dementia_status.config(text="NORMAL (Non-Demented)", fg="#10b981", bg="#064e3b")
                self.overall_banner.config(highlightbackground="#10b981")
                self.lbl_overall.config(
                    text="ALL CLEAR: No Brain Tumor Detected & No Dementia Detected (Normal Brain MRI)",
                    fg="#34d399"
                )
            else:
                self.lbl_dementia_status.config(text=f"ABNORMAL ({disease})", fg="#ef4444", bg="#450a0a")
                self.overall_banner.config(highlightbackground="#ef4444")
                self.lbl_overall.config(
                    text=f"ABNORMAL: Brain Tumor Normal ➔ Dementia Identified ({disease})",
                    fg="#f87171"
                )

            # Update Dementia Probabilities bars
            probs = d_res["probabilities"]
            for cname, pval in probs.items():
                if cname in self.prob_bars:
                    pct = pval * 100.0
                    self.prob_bars[cname]["value"] = pct
                    self.prob_labels[cname].config(text=f"{pct:.2f}%")

            unc_val = d_res.get("uncertainty", 0.0)
            unc_stat = d_res.get("uncertainty_status", "N/A")
            self.lbl_uncertainty.config(text=f"Dementia Uncertainty: {unc_val:.2f} ({unc_stat})")

            # Show Dementia Grad-CAM by default
            self._set_gradcam_display(self.active_dementia_gradcam, "Dementia Multi-Layer Attention Map")
            self.current_shown_gradcam = "dementia"

            if self.active_tumor_gradcam:
                self.btn_toggle_cam.config(text="View Tumor CAM")
                self.btn_toggle_cam.pack(side="right")
            else:
                self.btn_toggle_cam.pack_forget()

            if self.active_tumor_panel:
                self.btn_open_panel.pack(side="right", padx=(0, 6))
            else:
                self.btn_open_panel.pack_forget()

        # Quality warnings
        warnings = d_res.get("image_quality_warnings", []) if d_res else []
        if warnings:
            self.lbl_warning.config(text="Image Quality Advisory: " + "; ".join(warnings))
            self.warn_frame.pack(fill="x", pady=4, before=self.card_orig.master)
        else:
            self.warn_frame.pack_forget()

        # Display Input MRI Image
        try:
            orig_pil = Image.open(img_path).convert("RGB")
            orig_pil = orig_pil.resize((280, 280), Image.Resampling.LANCZOS)
            self.img_input_tk = ImageTk.PhotoImage(orig_pil)
            self.canvas_orig.config(image=self.img_input_tk, text="", width=0, height=0)
            self.canvas_orig.image = self.img_input_tk
        except Exception as e:
            self.canvas_orig.config(text=f"Failed to display input image:\n{e}", image="", width=0, height=0)

        self.lbl_latency.config(text=f"Total Latency: {total_ms:.0f} ms")
        self.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _set_gradcam_display(self, cam_path, description):
        """Displays the specified Grad-CAM image in the right card."""
        self.lbl_cam_desc.config(text=description)
        if cam_path and Path(cam_path).exists():
            try:
                grad_pil = Image.open(cam_path).convert("RGB")
                grad_pil = grad_pil.resize((280, 280), Image.Resampling.LANCZOS)
                self.img_gradcam_tk = ImageTk.PhotoImage(grad_pil)
                self.canvas_grad.config(image=self.img_gradcam_tk, text="", width=0, height=0)
                self.canvas_grad.image = self.img_gradcam_tk
            except Exception as e:
                self.canvas_grad.config(text=f"Failed to load Grad-CAM:\n{e}", image="", width=0, height=0)
        else:
            self.canvas_grad.config(text="Grad-CAM visualization\nnot generated", image="", width=0, height=0)

    def _open_panel_view(self):
        """Opens full 4-panel diagnostic view (Original | Cropped | Heatmap | Overlay)."""
        target = self.active_tumor_panel or self.active_tumor_gradcam or self.active_dementia_gradcam
        if target and Path(target).exists():
            try:
                os.startfile(target)
            except Exception:
                try:
                    Image.open(target).show()
                except Exception as e:
                    messagebox.showinfo("Diagnostic View", f"Saved file:\n{target}")

    def _toggle_gradcam_view(self):
        """Toggles between Dementia Grad-CAM and Brain Tumor Grad-CAM."""
        if self.current_shown_gradcam == "dementia" and self.active_tumor_gradcam:
            self.current_shown_gradcam = "tumor"
            self.btn_toggle_cam.config(text="View Dementia CAM")
            self._set_gradcam_display(self.active_tumor_gradcam, "Multi-Layer Fused Brain Tumor Localization (5 Layers)")
        elif self.active_dementia_gradcam:
            self.current_shown_gradcam = "dementia"
            self.btn_toggle_cam.config(text="View Tumor CAM")
            self._set_gradcam_display(self.active_dementia_gradcam, "Dementia Multi-Layer Attention Map")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cascade Brain MRI GUI Diagnostic System")
    parser.add_argument("--image", default=None, help="Path to input MRI image slice")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint .pth")
    args = parser.parse_args()

    app = CascadeMRIApp(initial_image=args.image, checkpoint=args.checkpoint)
    app.mainloop()


if __name__ == "__main__":
    main()
