"""Medical CT Multi-Model Triage & Diagnosis GUI Dashboard.
Replicates the clinical research prototype interface in dark theme with live Grad-CAM.
"""

import os
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Union

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from .inference import MedicalCTPredictor

# Color Palette matching the clinical prototype interface
COLOR_BG = "#090d16"           # Main window background
COLOR_PANEL = "#0f172a"        # Card background
COLOR_PANEL_ALT = "#131f37"    # Primary result card background
COLOR_BORDER = "#1e293b"       # Card borders
COLOR_TEXT_PRIMARY = "#f8fafc" # High-contrast white
COLOR_TEXT_MUTED = "#94a3b8"   # Slate gray
COLOR_CYAN = "#38bdf8"         # Accent cyan
COLOR_BLUE_BTN = "#2563eb"     # Action button blue
COLOR_BTN_HOVER = "#1d4ed8"
COLOR_BAR_BG = "#1e293b"       # Progress bar track
COLOR_BAR_FILL = "#38bdf8"     # Progress bar fill
COLOR_ALERT_BG = "#450a0a"     # Abnormal badge background
COLOR_ALERT_FG = "#ef4444"     # Abnormal badge text
COLOR_SAFE_BG = "#052e16"      # Normal badge background
COLOR_SAFE_FG = "#22c55e"      # Normal badge text

CUR_DIR = Path(__file__).resolve().parent
BASE_DIR = CUR_DIR.parent
PREVIEWS_DIR = BASE_DIR / "predictions"
if not PREVIEWS_DIR.exists() and (BASE_DIR.parent / "predictions").exists():
    PREVIEWS_DIR = BASE_DIR.parent / "predictions"
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)


class MedicalCTGUI(tk.Tk):
    """Clinical Decision Support GUI Dashboard for Medical CT Multi-Model Triage."""

    def __init__(self, initial_image: Optional[str] = None, predictor: Optional[MedicalCTPredictor] = None):
        super().__init__()
        self.title("Medical CT Multi-Model Triage & Diagnosis Dashboard")
        self.geometry("1100x840")
        self.minsize(980, 720)
        self.configure(bg=COLOR_BG)

        self.predictor = predictor
        self.current_result: Optional[Dict[str, Any]] = None
        self.input_photo = None
        self.cam_photo = None

        self._build_layout()

        if initial_image and os.path.exists(initial_image):
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, str(Path(initial_image).resolve()))
            self.after(300, self.run_analysis)

    def _build_layout(self):
        # 1. Top Input Bar
        top_bar = tk.Frame(self, bg=COLOR_PANEL, bd=0, padx=16, pady=12)
        top_bar.pack(fill=tk.X, side=tk.TOP)

        tk.Label(
            top_bar,
            text="Image Address / Path:",
            bg=COLOR_PANEL,
            fg=COLOR_TEXT_MUTED,
            font=("Segoe UI", 10, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.path_entry = tk.Entry(
            top_bar,
            bg="#090e1a",
            fg=COLOR_TEXT_PRIMARY,
            insertbackground="white",
            relief=tk.FLAT,
            font=("Segoe UI", 10),
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
            highlightcolor=COLOR_CYAN,
        )
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 12), ipady=4)

        browse_btn = tk.Button(
            top_bar,
            text="Browse Image...",
            bg=COLOR_BORDER,
            fg=COLOR_TEXT_PRIMARY,
            activebackground="#334155",
            activeforeground="white",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=4,
            cursor="hand2",
            command=self.browse_file,
        )
        browse_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.run_btn = tk.Button(
            top_bar,
            text="Run Analysis",
            bg=COLOR_BLUE_BTN,
            fg="white",
            activebackground=COLOR_BTN_HOVER,
            activeforeground="white",
            relief=tk.FLAT,
            font=("Segoe UI", 10, "bold"),
            padx=18,
            pady=4,
            cursor="hand2",
            command=self.run_analysis,
        )
        self.run_btn.pack(side=tk.LEFT)

        # 2. Main Scrollable Canvas Container
        container = tk.Frame(self, bg=COLOR_BG)
        container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(container, bg=COLOR_BG, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.scroll_content = tk.Frame(self.canvas, bg=COLOR_BG, padx=24, pady=16)

        self.scroll_content.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scroll_content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width - 10),
        )

        # Mousewheel scroll support
        self.bind_all("<MouseWheel>", lambda event: self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

        # 3. Header Title & Subtitle
        header_frame = tk.Frame(self.scroll_content, bg=COLOR_BG)
        header_frame.pack(fill=tk.X, pady=(0, 14))

        tk.Label(
            header_frame,
            text="Medical CT Multi-Model Triage & Diagnosis",
            bg=COLOR_BG,
            fg=COLOR_TEXT_PRIMARY,
            font=("Segoe UI", 18, "bold"),
            anchor="w",
        ).pack(fill=tk.X)

        tk.Label(
            header_frame,
            text="Deep Learning Decision Support Prototype | Multi-Model Routing & Explainable Grad-CAM",
            bg=COLOR_BG,
            fg=COLOR_TEXT_MUTED,
            font=("Segoe UI", 9),
            anchor="w",
        ).pack(fill=tk.X, pady=(2, 0))

        # 4. Primary Results Card
        self.card_result = tk.Frame(self.scroll_content, bg=COLOR_PANEL_ALT, bd=1, relief=tk.SOLID, padx=20, pady=16)
        self.card_result.pack(fill=tk.X, pady=(0, 14))

        self.lbl_anatomy_val = self._create_result_row(self.card_result, "Anatomy / Body Part:", "-- (Awaiting Analysis)")
        self.lbl_disease_val = self._create_result_row(self.card_result, "Disease Type / Category:", "--")
        self.lbl_conf_val = self._create_result_row(self.card_result, "Model Confidence:", "--")

        row_badge = tk.Frame(self.card_result, bg=COLOR_PANEL_ALT)
        row_badge.pack(fill=tk.X, pady=(4, 0))
        tk.Label(
            row_badge,
            text="Clinical Classification:",
            bg=COLOR_PANEL_ALT,
            fg=COLOR_TEXT_MUTED,
            font=("Segoe UI", 10, "bold"),
            width=22,
            anchor="w",
        ).pack(side=tk.LEFT)

        self.lbl_badge = tk.Label(
            row_badge,
            text="AWAITING ANALYSIS",
            bg=COLOR_BORDER,
            fg=COLOR_TEXT_MUTED,
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=3,
        )
        self.lbl_badge.pack(side=tk.LEFT)

        # 5. Dual Image Previews (Given Image vs Grad-CAM)
        images_frame = tk.Frame(self.scroll_content, bg=COLOR_BG)
        images_frame.pack(fill=tk.X, pady=(0, 14))
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)

        # Left: Input Image Card
        card_input = tk.Frame(images_frame, bg=COLOR_PANEL, bd=1, relief=tk.SOLID, padx=14, pady=12)
        card_input.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        tk.Label(card_input, text="Input CT Image (Given)", bg=COLOR_PANEL, fg=COLOR_TEXT_PRIMARY, font=("Segoe UI", 11, "bold"), anchor="w").pack(fill=tk.X)
        tk.Label(card_input, text="Original axial CT scan slice", bg=COLOR_PANEL, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 8), anchor="w").pack(fill=tk.X, pady=(0, 8))

        self.canvas_input = tk.Canvas(card_input, bg="#050811", height=230, highlightthickness=0)
        self.canvas_input.pack(fill=tk.BOTH, expand=True)

        # Right: Grad-CAM Card
        card_cam = tk.Frame(images_frame, bg=COLOR_PANEL, bd=1, relief=tk.SOLID, padx=14, pady=12)
        card_cam.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        cam_header = tk.Frame(card_cam, bg=COLOR_PANEL)
        cam_header.pack(fill=tk.X)

        tk.Label(
            cam_header,
            text="Grad-CAM Based Image",
            bg=COLOR_PANEL,
            fg=COLOR_TEXT_PRIMARY,
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        ).pack(side=tk.LEFT)

        # Toggle buttons for Fused Overlay vs Multi-Layer Projection Grid
        toggle_frame = tk.Frame(cam_header, bg=COLOR_PANEL)
        toggle_frame.pack(side=tk.RIGHT)

        self.btn_view_fused = tk.Button(
            toggle_frame,
            text="Fused Overlay",
            bg=COLOR_BLUE_BTN,
            fg="white",
            relief=tk.FLAT,
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=2,
            cursor="hand2",
            command=self._show_fused_cam,
        )
        self.btn_view_fused.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_view_grid = tk.Button(
            toggle_frame,
            text="Multi-Layer Grid",
            bg=COLOR_BORDER,
            fg=COLOR_TEXT_MUTED,
            relief=tk.FLAT,
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=2,
            cursor="hand2",
            command=self._show_grid_cam,
        )
        self.btn_view_grid.pack(side=tk.LEFT)

        self.lbl_cam_subtitle = tk.Label(
            card_cam,
            text="Hierarchical multi-layer gradient projection attention",
            bg=COLOR_PANEL,
            fg=COLOR_TEXT_MUTED,
            font=("Segoe UI", 8),
            anchor="w",
        )
        self.lbl_cam_subtitle.pack(fill=tk.X, pady=(0, 8))

        self.canvas_cam = tk.Canvas(card_cam, bg="#050811", height=230, highlightthickness=0)
        self.canvas_cam.pack(fill=tk.BOTH, expand=True)

        # 6. Diagnostic & Uncertainty Metrics Card
        self.card_metrics = tk.Frame(self.scroll_content, bg=COLOR_PANEL, bd=1, relief=tk.SOLID, padx=20, pady=16)
        self.card_metrics.pack(fill=tk.X, pady=(0, 14))

        tk.Label(
            self.card_metrics,
            text="Important Diagnostic & Uncertainty Metrics",
            bg=COLOR_PANEL,
            fg=COLOR_TEXT_PRIMARY,
            font=("Segoe UI", 12, "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 12))

        self.bars_container = tk.Frame(self.card_metrics, bg=COLOR_PANEL)
        self.bars_container.pack(fill=tk.X)

        # Footer Metrics Row
        self.footer_metrics = tk.Frame(self.card_metrics, bg=COLOR_PANEL)
        self.footer_metrics.pack(fill=tk.X, pady=(16, 0))

        self.lbl_meta_uncertainty = tk.Label(self.footer_metrics, text="Router Confidence: --", bg=COLOR_PANEL, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 8))
        self.lbl_meta_uncertainty.pack(side=tk.LEFT, padx=(0, 24))

        self.lbl_meta_latency = tk.Label(self.footer_metrics, text="Inference Latency: --", bg=COLOR_PANEL, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 8))
        self.lbl_meta_latency.pack(side=tk.LEFT, padx=(0, 24))

        self.lbl_meta_arch = tk.Label(self.footer_metrics, text="Architecture: EfficientNet / SE-ResNeXt", bg=COLOR_PANEL, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 8))
        self.lbl_meta_arch.pack(side=tk.LEFT, padx=(0, 24))

        self.lbl_meta_status = tk.Label(self.footer_metrics, text="Status: Ready", bg=COLOR_PANEL, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 8))
        self.lbl_meta_status.pack(side=tk.RIGHT)

        # 7. Bottom Legal / Medical Prototype Notice
        bottom_notice = tk.Frame(self, bg="#070c18", padx=16, pady=8, bd=1, relief=tk.SOLID)
        bottom_notice.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Label(
            bottom_notice,
            text="NOTICE: Research prototype. Model confidence is not clinical certainty, and Grad-CAM is an attribution heatmap, not a lesion segmentation or medical diagnosis.",
            bg="#070c18",
            fg="#64748b",
            font=("Segoe UI", 8),
            anchor="center",
        ).pack(fill=tk.X)

        self._render_placeholder_bars()

    def _create_result_row(self, parent: tk.Widget, label_text: str, default_val: str) -> tk.Label:
        row = tk.Frame(parent, bg=COLOR_PANEL_ALT)
        row.pack(fill=tk.X, pady=(2, 6))

        tk.Label(
            row,
            text=label_text,
            bg=COLOR_PANEL_ALT,
            fg=COLOR_TEXT_MUTED,
            font=("Segoe UI", 10, "bold"),
            width=22,
            anchor="w",
        ).pack(side=tk.LEFT)

        val_lbl = tk.Label(
            row,
            text=default_val,
            bg=COLOR_PANEL_ALT,
            fg=COLOR_CYAN,
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        )
        val_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return val_lbl

    def _render_placeholder_bars(self):
        for widget in self.bars_container.winfo_children():
            widget.destroy()

        dummy_classes = [("Class A", 0.0), ("Class B", 0.0), ("Class C", 0.0)]
        for name, pct in dummy_classes:
            self._add_progress_bar_row(self.bars_container, name, pct)

    def _add_progress_bar_row(
        self,
        parent: tk.Widget,
        name: str,
        pct: float,
        bar_fill: str = COLOR_BAR_FILL,
        label_color: str = COLOR_TEXT_PRIMARY,
    ):
        row = tk.Frame(parent, bg=COLOR_PANEL)
        row.pack(fill=tk.X, pady=3)

        tk.Label(
            row,
            text=name,
            bg=COLOR_PANEL,
            fg=label_color,
            font=("Segoe UI", 9, "bold"),
            width=22,
            anchor="w",
        ).pack(side=tk.LEFT)

        bar_canvas = tk.Canvas(row, bg=COLOR_PANEL, height=14, width=320, highlightthickness=0)
        bar_canvas.pack(side=tk.LEFT, padx=(0, 16))

        bar_canvas.create_rectangle(0, 2, 320, 12, fill=COLOR_BAR_BG, outline="")
        fill_w = int((pct / 100.0) * 320)
        if fill_w > 0:
            bar_canvas.create_rectangle(0, 2, fill_w, 12, fill=bar_fill, outline="")

        tk.Label(
            row,
            text=f"{pct:.2f}%",
            bg=COLOR_PANEL,
            fg=label_color,
            font=("Segoe UI", 9, "bold"),
            width=8,
            anchor="e",
        ).pack(side=tk.LEFT)

    def _show_fused_cam(self):
        if not self.current_result:
            return
        self.btn_view_fused.config(bg=COLOR_BLUE_BTN, fg="white")
        self.btn_view_grid.config(bg=COLOR_BORDER, fg=COLOR_TEXT_MUTED)
        self.lbl_cam_subtitle.config(text="Fused multi-layer gradient projection overlay")
        cam_pil = self.current_result.get("cam_overlay_pil")
        if cam_pil:
            self._display_image_on_canvas(cam_pil, self.canvas_cam, is_cam=True)

    def _show_grid_cam(self):
        if not self.current_result:
            return
        grid_pil = self.current_result.get("cam_grid_pil")
        if not grid_pil:
            grid_file = self.current_result.get("gradcam_files", {}).get("multilayer_grid")
            if grid_file and os.path.exists(grid_file):
                try:
                    grid_pil = Image.open(grid_file)
                except Exception:
                    pass
        if grid_pil:
            self.btn_view_grid.config(bg=COLOR_BLUE_BTN, fg="white")
            self.btn_view_fused.config(bg=COLOR_BORDER, fg=COLOR_TEXT_MUTED)
            self.lbl_cam_subtitle.config(text="Hierarchical layer progression: Input -> Stages -> Fused")
            self._display_image_on_canvas(grid_pil, self.canvas_cam, is_cam=True)

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select CT Slice Image",
            filetypes=[
                ("Image & DICOM Files", "*.png;*.jpg;*.jpeg;*.dcm;*.bmp;*.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if filename:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, filename)

    def run_analysis(self):
        raw_path = self.path_entry.get().strip().strip('"\'')
        if not raw_path:
            messagebox.showwarning("Input Required", "Please browse or enter a CT scan image address.")
            return

        img_path = Path(raw_path)
        if not img_path.exists():
            messagebox.showerror("File Not Found", f"Specified image does not exist:\n{raw_path}")
            return

        self.run_btn.config(state=tk.DISABLED, text="Analyzing...")
        self.lbl_meta_status.config(text="Status: Running neural network inference & Grad-CAM...")

        # Clear previous cam image and show loading notice
        self.canvas_cam.delete("all")
        self.canvas_cam.create_text(
            180, 115,
            text="Generating Multi-Layer Grad-CAM...",
            fill=COLOR_CYAN,
            font=("Segoe UI", 10, "italic"),
        )

        threading.Thread(target=self._async_predict, args=(img_path,), daemon=True).start()

    def _async_predict(self, img_path: Path):
        try:
            if self.predictor is None:
                self.predictor = MedicalCTPredictor()

            PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
            import hashlib, re
            safe_name = f"{re.sub(r'[^a-zA-Z0-9_\-]', '_', img_path.stem)[:18]}_{hashlib.md5(img_path.stem.encode('utf-8')).hexdigest()[:8]}"
            out_dir = PREVIEWS_DIR / safe_name

            result = self.predictor.predict(img_path, gradcam_dir=out_dir)
            self.after(0, self._update_ui_with_result, result, img_path)
        except Exception as e:
            self.after(0, self._handle_error, str(e))

    def _handle_error(self, err_msg: str):
        self.run_btn.config(state=tk.NORMAL, text="Run Analysis")
        self.lbl_meta_status.config(text="Status: Error encountered")
        messagebox.showerror("Inference Error", f"Failed to complete CT analysis:\n{err_msg}")

    def _update_ui_with_result(self, r: Dict[str, Any], img_path: Path):
        self.current_result = r
        self.run_btn.config(state=tk.NORMAL, text="Run Analysis")

        router_res = r.get("router", {})
        expert_res = r.get("expert_result", {})
        anatomy = router_res.get("prediction", "UNKNOWN")
        router_conf = router_res.get("confidence_pct", "100.0%")
        latency_ms = int(r.get("latency_seconds", 0.0) * 1000)

        # 1. Update Result Card Labels
        anatomy_label = "Chest CT (Thoracic)" if anatomy == "CHEST_CT" else ("Brain CT (Cranial)" if anatomy == "BRAIN_CT" else anatomy)
        self.lbl_anatomy_val.config(text=f"{anatomy_label}  |  Router Confidence: {router_conf}")

        diagnosis = expert_res.get("diagnosis", "Awaiting Review")
        conf_pct = expert_res.get("confidence_pct", "--")
        self.lbl_disease_val.config(text=diagnosis)
        self.lbl_conf_val.config(text=f"{conf_pct} (Calibrated)")

        # 2. Update Classification Badge
        is_safe = expert_res.get("is_safe", False) or diagnosis == "Normal" or ("No Acute" in diagnosis)
        if router_res.get("uncertain"):
            self.lbl_badge.config(
                text="UNCERTAIN (Radiologist Review Required)",
                bg="#78350f",
                fg="#fde047",
            )
        elif is_safe:
            self.lbl_badge.config(
                text=f"NORMAL ({diagnosis.upper()})",
                bg=COLOR_SAFE_BG,
                fg=COLOR_SAFE_FG,
            )
        else:
            self.lbl_badge.config(
                text=f"ABNORMAL ({diagnosis.upper()})",
                bg=COLOR_ALERT_BG,
                fg=COLOR_ALERT_FG,
            )

        # 3. Render Progress Bars (Class Probabilities + Multi-Layer Weights)
        for widget in self.bars_container.winfo_children():
            widget.destroy()

        # Section A: Diagnostic Class Probabilities
        tk.Label(
            self.bars_container,
            text="Class Probabilities:",
            bg=COLOR_PANEL,
            fg=COLOR_TEXT_MUTED,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 2))

        probs = expert_res.get("probabilities", {})
        for name, prob in probs.items():
            label_name = "Any Hemorrhage" if name == "any" else name.capitalize()
            self._add_progress_bar_row(self.bars_container, label_name, prob * 100.0, bar_fill=COLOR_BAR_FILL)

        # Section B: Multi-Layer Gradient Projection Weights
        layer_weights = r.get("gradcam_layer_weights", {})
        if layer_weights:
            sep = tk.Frame(self.bars_container, bg=COLOR_BORDER, height=1)
            sep.pack(fill=tk.X, pady=(10, 6))

            tk.Label(
                self.bars_container,
                text="Multi-Layer Gradient Projection Contributions:",
                bg=COLOR_PANEL,
                fg=COLOR_CYAN,
                font=("Segoe UI", 9, "bold"),
                anchor="w",
            ).pack(fill=tk.X, pady=(0, 4))

            for l_name, l_pct in layer_weights.items():
                self._add_progress_bar_row(
                    self.bars_container,
                    l_name,
                    l_pct,
                    bar_fill="#60a5fa",
                    label_color="#cbd5e1",
                )

        # 4. Update Footer Info
        self.lbl_meta_uncertainty.config(text=f"Router Confidence: {router_conf}")
        self.lbl_meta_latency.config(text=f"Inference Latency: {latency_ms} ms")
        arch_name = expert_res.get("model_name", "EfficientNet-B0")
        self.lbl_meta_arch.config(text=f"Architecture: {arch_name}")
        self.lbl_meta_status.config(text="Status: Completed")

        # 5. Display Input Image
        self._display_image_on_canvas(img_path, self.canvas_input, is_cam=False)

        # 6. Display Grad-CAM Image (Reset to Fused view)
        self.btn_view_fused.config(bg=COLOR_BLUE_BTN, fg="white")
        self.btn_view_grid.config(bg=COLOR_BORDER, fg=COLOR_TEXT_MUTED)
        self.lbl_cam_subtitle.config(text="Fused multi-layer gradient projection overlay")

        cam_pil = r.get("cam_overlay_pil")
        if cam_pil is not None:
            self._display_image_on_canvas(cam_pil, self.canvas_cam, is_cam=True)
        else:
            gradcam_files = r.get("gradcam_files", {})
            cam_path = gradcam_files.get("overlay") or gradcam_files.get("dashboard") or gradcam_files.get("heatmap")
            if cam_path and os.path.exists(cam_path):
                self._display_image_on_canvas(cam_path, self.canvas_cam, is_cam=True)
            else:
                self.canvas_cam.delete("all")
                self.canvas_cam.create_text(
                    180, 115,
                    text="Grad-CAM Generated in predictions/",
                    fill=COLOR_TEXT_MUTED,
                    font=("Segoe UI", 10),
                )

    def _display_image_on_canvas(self, img_input: Union[str, Path, Image.Image], canvas: tk.Canvas, is_cam: bool = False):
        try:
            if isinstance(img_input, Image.Image):
                im = img_input.copy()
            else:
                im = Image.open(str(img_input))

            canvas.update_idletasks()
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw < 50:
                cw = 380
            if ch < 50:
                ch = 230

            # Scale preserving aspect ratio
            im.thumbnail((cw - 16, ch - 16), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(im)

            if is_cam:
                self.cam_photo = photo
            else:
                self.input_photo = photo

            canvas.delete("all")
            canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=photo)
        except Exception as e:
            canvas.delete("all")
            canvas.create_text(
                180, 115,
                text=f"Preview Unavailable\n{e}",
                fill=COLOR_TEXT_MUTED,
                font=("Segoe UI", 9),
            )

