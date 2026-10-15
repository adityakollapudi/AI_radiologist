import io
import base64
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable
)
from reportlab.pdfgen import canvas

from ..schemas.report import ReportRequest
from ..core.errors import MedicalAIException

# Color Palette: Clinical Dark Navy with Restrained Cyan Accents
NAVY_PRIMARY = colors.HexColor("#0A192F")
NAVY_SECONDARY = colors.HexColor("#1E293B")
CYAN_ACCENT = colors.HexColor("#0891B2")
CYAN_LIGHT = colors.HexColor("#06B6D4")
CYAN_TINT = colors.HexColor("#ECFEFF")
BG_CARD = colors.HexColor("#F8FAFC")
BORDER_LIGHT = colors.HexColor("#E2E8F0")
BORDER_SUBTLE = colors.HexColor("#CBD5E1")
TEXT_MUTED = colors.HexColor("#64748B")
TEXT_DARK = colors.HexColor("#0F172A")
ALERT_RED = colors.HexColor("#DC2626")
ALERT_GREEN = colors.HexColor("#059669")
ALERT_AMBER = colors.HexColor("#D97706")


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute total pages and draw
    consistent header/footer with 'Page X of Y' numbering.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        page_width, page_height = A4
        margin = 36  # 0.5 inch

        # Footer divider line
        self.setStrokeColor(BORDER_LIGHT)
        self.setLineWidth(0.6)
        self.line(margin, 30, page_width - margin, 30)

        # Footer text
        self.setFont("Helvetica", 7.5)
        self.setFillColor(TEXT_MUTED)
        self.drawString(
            margin,
            20,
            "MEDVISION AI — Clinical AI Research Prototype • Not for Standalone Clinical Diagnosis"
        )
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(page_width - margin, 20, page_str)

        self.restoreState()


def _decode_base64_image(b64_str: Optional[str]) -> Optional[PILImage.Image]:
    """Safely decodes a base64 or data URL image string into a PIL Image."""
    if not b64_str or not isinstance(b64_str, str):
        return None
    try:
        raw = b64_str
        if "," in raw and raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        raw_bytes = base64.b64decode(raw)
        img = PILImage.open(io.BytesIO(raw_bytes))
        # Ensure image is loaded into memory
        img.load()
        return img
    except Exception as e:
        return None


def _scale_image_proportional(
    pil_img: PILImage.Image,
    max_w: float = 230.0,
    max_h: float = 210.0
) -> Tuple[float, float, io.BytesIO]:
    """
    Calculates exact proportional scaling preserving aspect ratio.
    Returns (scaled_width, scaled_height, io_buffer).
    """
    orig_w, orig_h = pil_img.size
    if orig_w <= 0 or orig_h <= 0:
        orig_w, orig_h = 100, 100

    scale = min(max_w / orig_w, max_h / orig_h)
    target_w = orig_w * scale
    target_h = orig_h * scale

    buf = io.BytesIO()
    # Save as PNG for lossless representation in PDF
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return target_w, target_h, buf


def _format_task_title(modality: str, routed_to: Optional[str] = None) -> str:
    """Returns the descriptive modality-specific task title."""
    mod = (modality or "").lower()
    r = (routed_to or "").lower()

    if mod == "xray":
        if "mura" in r or "musculo" in r:
            return "Musculoskeletal Radiograph Abnormality Detection"
        return "Chest Radiograph Pneumonia Detection"
    elif mod == "ct":
        if "bleed" in r or "brain" in r:
            return "Non-Contrast Head CT Hemorrhage Detection"
        return "Chest CT Axial Slice COVID-19 / Pneumonia Detection"
    elif mod == "mri":
        if "dementia" in r:
            return "Brain MRI Neurodegenerative Dementia Assessment"
        return "Brain MRI Neoplasm / Tumor Detection"
    
    return f"{mod.upper()} Medical Imaging Analysis"


def _is_disease_positive(prediction: str) -> bool:
    """Helper to detect whether prediction is a positive disease/abnormal finding."""
    if not prediction or not isinstance(prediction, str):
        return False
    p = prediction.upper()
    if any(neg in p for neg in [
        "NO TUMOR", "NO BLEED", "NO HEMORRHAGE", "NON-DEMENTED",
        "BRAIN IS SAFE", "SAFE", "HEALTHY", "BENIGN"
    ]):
        return False
    if "NORMAL" in p and "ABNORMAL" not in p:
        return False
    return any(pos in p for pos in [
        "TUMOR", "HEMORRHAGE", "BLEED", "PNEUMONIA", "COVID",
        "ABNORMAL", "DEMENTIA", "ALERT", "CRITICAL", "POSITIVE"
    ])


def _deterministic_model_explanation(
    modality: str,
    prediction: str,
    confidence: Optional[float] = None
) -> str:
    """
    Generates a neutral, deterministic clinical description of the classifier output.
    Does not use LLMs and does not fabricate predictions or diagnoses.
    """
    pred_upper = (prediction or "").upper()
    mod_upper = (modality or "").upper()
    conf_clause = f" with a model probability score of {confidence * 100:.1f}%" if confidence is not None else ""

    if (
        "SAFE" in pred_upper
        or "NO HEMORRHAGE" in pred_upper
        or "NO BLEED" in pred_upper
        or "NO TUMOR" in pred_upper
        or "NON-DEMENTED" in pred_upper
        or "NORMAL" in pred_upper
    ):
        return (
            f"The {mod_upper} classifier evaluated the input image and identified no dominant radiographic "
            f"patterns of acute focal pathology{conf_clause}, categorizing the scan as {prediction}."
        )
    elif "PNEUMONIA" in pred_upper:
        return (
            f"The {mod_upper} deep learning classifier evaluated thoracic radiographic features and identified "
            f"pulmonary infiltrates/consolidation patterns indicative of {prediction}{conf_clause}."
        )
    elif "COVID" in pred_upper:
        return (
            f"The {mod_upper} volumetric axial slice classifier evaluated lung parenchymal attenuation and "
            f"identified ground-glass/consolidation patterns characteristic of {prediction}{conf_clause}."
        )
    elif "BLEED" in pred_upper or "HEMORRHAGE" in pred_upper:
        return (
            f"The {mod_upper} neural network evaluated non-contrast intracranial density features and "
            f"identified localized hyperdensity indicative of {prediction}{conf_clause}."
        )
    elif "TUMOR" in pred_upper:
        return (
            f"The {mod_upper} diagnostic pipeline evaluated intracranial structural morphology and "
            f"identified tissue contrast characteristics consistent with {prediction}{conf_clause}."
        )
    elif "DEMENTIA" in pred_upper:
        return (
            f"The {mod_upper} classifier evaluated cerebral atrophy and ventricular enlargement biomarkers, "
            f"yielding a category of {prediction}{conf_clause}."
        )
    else:
        return (
            f"The {mod_upper} model evaluated imaging features and classified the findings "
            f"as {prediction}{conf_clause}."
        )


def generate_medical_pdf_report(report_data: ReportRequest) -> bytes:
    """
    Generates a professional, A4-sized PDF medical analysis report following
    the MEDVISION AI clinical document specifications.
    """
    # 1. Validate original image
    orig_pil = _decode_base64_image(report_data.original_png_base64)
    if orig_pil is None:
        raise MedicalAIException(
            "Original scan image is missing or invalid. An authentic original scan is required to generate the diagnostic report.",
            status_code=400
        )

    # Decode Grad-CAM if available
    gradcam_pil = _decode_base64_image(report_data.gradcam_png_base64)

    # Document Geometry
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=42
    )

    usable_width = A4[0] - 72  # 523.27 pt

    # Styles Setup
    styles = getSampleStyleSheet()

    title_brand_style = ParagraphStyle(
        "BrandTitle",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=21,
        textColor=NAVY_PRIMARY
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=CYAN_ACCENT
    )
    header_meta_style = ParagraphStyle(
        "HeaderMeta",
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=TEXT_MUTED,
        alignment=2  # Right-aligned
    )
    header_meta_val_style = ParagraphStyle(
        "HeaderMetaVal",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=NAVY_SECONDARY,
        alignment=2
    )

    section_heading_style = ParagraphStyle(
        "SectionHeading",
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        textColor=NAVY_PRIMARY,
        spaceAfter=5
    )

    label_style = ParagraphStyle(
        "CardLabel",
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=TEXT_MUTED
    )
    is_positive = _is_disease_positive(report_data.prediction)
    val_pred_style = ParagraphStyle(
        "CardValuePred",
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=15,
        textColor=ALERT_RED if is_positive else (ALERT_GREEN if not is_positive and report_data.prediction != "UNKNOWN" else NAVY_PRIMARY)
    )
    val_normal_style = ParagraphStyle(
        "CardValueNormal",
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=NAVY_SECONDARY
    )
    body_text_style = ParagraphStyle(
        "BodyTextRegular",
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=TEXT_DARK
    )

    image_caption_title = ParagraphStyle(
        "ImageCaptionTitle",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=10.5,
        textColor=NAVY_PRIMARY,
        alignment=1  # Centered
    )
    image_caption_sub = ParagraphStyle(
        "ImageCaptionSub",
        fontName="Helvetica",
        fontSize=7,
        leading=8.5,
        textColor=TEXT_MUTED,
        alignment=1  # Centered
    )

    disclaimer_text_style = ParagraphStyle(
        "DisclaimerText",
        fontName="Helvetica",
        fontSize=7.5,
        leading=10.5,
        textColor=NAVY_SECONDARY
    )

    meta_key_style = ParagraphStyle(
        "MetaKey",
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=TEXT_MUTED
    )
    meta_val_style = ParagraphStyle(
        "MetaVal",
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=NAVY_SECONDARY
    )

    story = []

    # ---------------------------------------------------------
    # HEADER
    # ---------------------------------------------------------
    report_id = report_data.request_id or f"MEDV-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    gen_time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    modality_name = report_data.modality.upper()
    task_name = _format_task_title(report_data.modality, report_data.routed_to)

    header_left = [
        Paragraph("MEDVISION AI", title_brand_style),
        Spacer(1, 2),
        Paragraph("AI-Assisted Medical Imaging Analysis Report", subtitle_style),
        Spacer(1, 2),
        Paragraph(f"<b>Modality:</b> {modality_name} &nbsp;|&nbsp; <b>Task:</b> {task_name}", val_normal_style)
    ]

    header_right = [
        Paragraph(f"<b>Report ID:</b> {report_id}", header_meta_val_style),
        Paragraph(f"<b>Generated:</b> {gen_time_str}", header_meta_style),
        Paragraph("<b>Status:</b> Official AI Inference Summary", header_meta_style)
    ]

    header_table = Table(
        [[header_left, header_right]],
        colWidths=[usable_width * 0.62, usable_width * 0.38]
    )
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 4))

    # Cyan Accent Separator
    story.append(HRFlowable(
        width="100%",
        thickness=1.5,
        color=CYAN_ACCENT,
        spaceBefore=1,
        spaceAfter=10
    ))

    # ---------------------------------------------------------
    # SECTION 1 — PREDICTION SUMMARY
    # ---------------------------------------------------------
    story.append(Paragraph("1. PREDICTION SUMMARY", section_heading_style))

    # Confidence display
    conf_display = "Not Provided"
    if report_data.confidence is not None and 0.0 <= report_data.confidence <= 1.0:
        conf_display = f"{report_data.confidence * 100:.1f}%"
    elif report_data.probabilities and report_data.prediction in report_data.probabilities:
        prob = report_data.probabilities[report_data.prediction]
        conf_display = f"{prob * 100:.1f}%"

    model_name_display = report_data.model_name or "Unified Multi-Architecture Ensemble"
    explanation_text = _deterministic_model_explanation(
        report_data.modality,
        report_data.prediction,
        report_data.confidence
    )

    badge_html = f'<font color="#DC2626"><b>[POSITIVE FINDING DETECTED]</b></font><br/>' if is_positive else ""
    pred_cell = [
        Paragraph(f"{badge_html}PREDICTED CLASS", label_style),
        Spacer(1, 1),
        Paragraph(f"<b>{report_data.prediction}</b>", val_pred_style),
        Spacer(1, 4),
        Paragraph("CONFIDENCE SCORE", label_style),
        Spacer(1, 1),
        Paragraph(f"<b>{conf_display}</b>", val_normal_style),
    ]

    task_cell = [
        Paragraph("ANALYSIS TASK", label_style),
        Spacer(1, 1),
        Paragraph(f"<b>{task_name}</b>", val_normal_style),
        Spacer(1, 4),
        Paragraph("INFERENCE PIPELINE / MODEL", label_style),
        Spacer(1, 1),
        Paragraph(f"<b>{model_name_display} ({report_data.framework or 'PyTorch'})</b>", val_normal_style),
    ]

    summary_table = Table(
        [
            [pred_cell, task_cell],
            [
                [
                    Paragraph("MODEL OUTPUT INTERPRETATION", label_style),
                    Spacer(1, 1),
                    Paragraph(explanation_text, body_text_style)
                ],
                ""
            ]
        ],
        colWidths=[usable_width * 0.48, usable_width * 0.52]
    )
    summary_table.setStyle(TableStyle([
        ("SPAN", (0, 1), (1, 1)),
        ("BACKGROUND", (0, 0), (-1, -1), BG_CARD),
        ("BOX", (0, 0), (-1, -1), 0.75, BORDER_SUBTLE),
        ("LINEBELOW", (0, 0), (1, 0), 0.5, BORDER_LIGHT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # SECTION 2 — SCAN COMPARISON: SIDE BY SIDE
    # ---------------------------------------------------------
    # Available width per column: usable_width is ~523.27
    col_w = usable_width / 2.0  # 261.63 pt
    # Max image box within column
    max_img_w = col_w - 24.0   # ~237 pt
    max_img_h = 205.0          # 205 pt

    # Scale Original Image
    orig_w, orig_h, orig_buf = _scale_image_proportional(orig_pil, max_w=max_img_w, max_h=max_img_h)
    orig_rl_img = RLImage(orig_buf, width=orig_w, height=orig_h)

    # Frame for Original Image
    orig_frame = Table(
        [[orig_rl_img]],
        colWidths=[col_w - 12]
    )
    orig_frame.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, -1), BG_CARD),
        ("BOX", (0, 0), (-1, -1), 0.75, BORDER_SUBTLE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))

    orig_col_elements = [
        Paragraph("ORIGINAL INPUT SCAN", image_caption_title),
        Paragraph("Original image uploaded by user", image_caption_sub),
        Spacer(1, 4),
        orig_frame
    ]

    # Handle Grad-CAM image or missing message
    if gradcam_pil is not None:
        gc_w, gc_h, gc_buf = _scale_image_proportional(gradcam_pil, max_w=max_img_w, max_h=max_img_h)
        gc_rl_img = RLImage(gc_buf, width=gc_w, height=gc_h)

        gc_frame = Table(
            [[gc_rl_img]],
            colWidths=[col_w - 12]
        )
        gc_frame.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, -1), BG_CARD),
            ("BOX", (0, 0), (-1, -1), 0.75, BORDER_SUBTLE),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))

        gradcam_col_elements = [
            Paragraph("GRAD-CAM VISUALIZATION", image_caption_title),
            Paragraph("Grad-CAM activation overlay (model attention)", image_caption_sub),
            Spacer(1, 4),
            gc_frame
        ]
    else:
        # Graceful handling for missing Grad-CAM
        unavailable_text_style = ParagraphStyle(
            "UnavailableText",
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11.5,
            textColor=TEXT_MUTED,
            alignment=1
        )
        unavailable_note_style = ParagraphStyle(
            "UnavailableNote",
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.5,
            textColor=TEXT_MUTED,
            alignment=1
        )

        unavailable_card = Table(
            [
                [
                    [
                        Spacer(1, 75),
                        Paragraph("Grad-CAM visualization is unavailable for this analysis.", unavailable_text_style),
                        Spacer(1, 4),
                        Paragraph("The model prediction was executed directly; visual activation maps were not generated for this scan.", unavailable_note_style),
                        Spacer(1, 75)
                    ]
                ]
            ],
            colWidths=[col_w - 12]
        )
        unavailable_card.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, -1), BG_CARD),
            ("BOX", (0, 0), (-1, -1), 0.75, BORDER_SUBTLE),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))

        gradcam_col_elements = [
            Paragraph("GRAD-CAM VISUALIZATION", image_caption_title),
            Paragraph("Grad-CAM activation overlay (model attention)", image_caption_sub),
            Spacer(1, 4),
            unavailable_card
        ]

    comparison_table = Table(
        [[orig_col_elements, gradcam_col_elements]],
        colWidths=[col_w, col_w]
    )
    comparison_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    # Keep Section 2 together on the same page
    section_2_elements = [
        Paragraph("2. SCAN COMPARISON — SIDE BY SIDE", section_heading_style),
        Spacer(1, 2),
        comparison_table
    ]
    story.append(KeepTogether(section_2_elements))
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # SECTION 3 — LIMITATIONS AND DISCLAIMER
    # ---------------------------------------------------------
    disclaimer_text = (
        "This report is generated by an AI research prototype for educational and demonstration purposes. "
        "The model output is not a standalone medical diagnosis and must not be used as the sole basis for "
        "clinical decisions. Grad-CAM highlights image regions relevant to the model's output; it does not "
        "prove the presence or absence of disease. Results require interpretation by a qualified healthcare professional."
    )

    disclaimer_table = Table(
        [[
            Paragraph(
                f"<b>CLINICAL RESEARCH DISCLAIMER:</b> {disclaimer_text}",
                disclaimer_text_style
            )
        ]],
        colWidths=[usable_width]
    )
    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_SUBTLE),
        ("LINEBEFORE", (0, 0), (0, 0), 3.0, CYAN_ACCENT),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))

    section_3_elements = [
        Paragraph("3. LIMITATIONS AND DISCLAIMER", section_heading_style),
        Spacer(1, 2),
        disclaimer_table
    ]
    story.append(KeepTogether(section_3_elements))
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # SECTION 4 — REPORT METADATA
    # ---------------------------------------------------------
    dataset_name = report_data.dataset or "Curated Multi-Institution Medical Imaging Cohort"
    input_size_display = report_data.input_size or (f"{orig_pil.width}x{orig_pil.height} px" if orig_pil else "Standard 224x224")

    meta_row1 = [
        [Paragraph("REPORT ID", meta_key_style), Paragraph(report_id, meta_val_style)],
        [Paragraph("IMAGING MODALITY", meta_key_style), Paragraph(modality_name, meta_val_style)],
        [Paragraph("ANALYSIS TASK", meta_key_style), Paragraph(task_name, meta_val_style)],
    ]
    meta_row2 = [
        [Paragraph("GENERATION TIME", meta_key_style), Paragraph(gen_time_str, meta_val_style)],
        [Paragraph("AI ARCHITECTURE", meta_key_style), Paragraph(f"{report_data.framework or 'PyTorch'} ({model_name_display})", meta_val_style)],
        [Paragraph("IMAGE GEOMETRY", meta_key_style), Paragraph(input_size_display, meta_val_style)],
    ]

    metadata_table = Table(
        [
            [
                meta_row1[0][0], meta_row1[0][1],
                meta_row1[1][0], meta_row1[1][1],
                meta_row1[2][0], meta_row1[2][1],
            ],
            [
                meta_row2[0][0], meta_row2[0][1],
                meta_row2[1][0], meta_row2[1][1],
                meta_row2[2][0], meta_row2[2][1],
            ]
        ],
        colWidths=[
            usable_width * 0.16, usable_width * 0.17,
            usable_width * 0.16, usable_width * 0.17,
            usable_width * 0.16, usable_width * 0.18
        ]
    )
    metadata_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG_CARD),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_LIGHT),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#F1F5F9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))

    section_4_elements = [
        Paragraph("4. REPORT METADATA", section_heading_style),
        Spacer(1, 2),
        metadata_table
    ]
    story.append(KeepTogether(section_4_elements))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes
