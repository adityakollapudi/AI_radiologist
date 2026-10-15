import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import httpx

from ..core.config import settings
from ..schemas.assistant import ChatQueryRequest, ChatQueryResponse, ChatMessage

SYSTEM_INSTRUCTION = """You are MedVision AI Assistant, a friendly, compassionate, and clear medical imaging guide for everyday patients and non-technical users.

CRITICAL INSTRUCTION — PLAIN, SIMPLE LANGUAGE:
1. NEVER use dense medical jargon or complex engineering buzzwords (e.g., avoid words like "alveolar opacification", "posterior softmax probability", "parenchymal attenuation", "hyperdensity", "atrophy indices").
2. ALWAYS explain findings in simple, everyday English that a patient, parent, or family member can easily understand.
3. Use friendly, clear analogies:
   - Explain Grad-CAM as a "highlighter pen" showing where the AI focused its eyes.
   - Explain lungs on X-rays like "air-filled balloons that should look clear and dark, but get cloudy when fluid or infection builds up".
   - Explain CT as "looking at the body in 3D slices, like slices of bread".
   - Explain MRI as "using safe magnets with no radiation to take super-detailed pictures of soft tissue".
   - Explain confidence score as "how closely the pattern in this picture matches thousands of verified cases the AI studied".
4. Reassure the user, explain what symptoms might feel like, and clearly list practical, easy next steps.
5. Remind users in a kind way that this AI is a research helper, and their real doctor will make the final decision.
"""

def _extract_context_summary(context: Optional[Dict[str, Any]]) -> str:
    if not context:
        return "No scan has been uploaded yet. General friendly imaging guidance mode."
    
    modality = context.get("modality", "Unknown").upper()
    pred = context.get("prediction", "Unknown")
    conf = context.get("confidence")
    conf_str = f"{conf * 100:.1f}%" if isinstance(conf, (int, float)) else "N/A"
    task = context.get("routed_to") or context.get("task") or "Diagnostic Screening"

    return (
        f"Active Scan Summary:\n"
        f"- Scan Type: {modality}\n"
        f"- What the AI Found: {pred}\n"
        f"- How Sure the AI Is: {conf_str}\n"
        f"- Analysis Focus: {task}\n"
    )

def _has_keyword(text: str, keywords: List[str]) -> bool:
    """Checks if any keyword matches as a distinct word or multi-word phrase."""
    for k in keywords:
        if " " in k or "-" in k:
            if k in text:
                return True
        else:
            if re.search(r'\b' + re.escape(k) + r'\b', text, re.IGNORECASE):
                return True
    return False

def _generate_rule_based_reply(
    message: str,
    context: Optional[Dict[str, Any]]
) -> Tuple[str, List[str]]:
    """
    Intelligent Clinical Knowledge Engine providing clear, layperson-friendly,
    easy-to-understand explanations without complicated medical jargon.
    """
    msg = message.lower().strip()
    
    # Context variables
    modality = (context.get("modality") or "").lower() if context else ""
    prediction = (context.get("prediction") or "").upper() if context else ""
    confidence = context.get("confidence") if context else None
    conf_str = f"{confidence * 100:.1f}%" if isinstance(confidence, (int, float)) else "N/A"

    # 1. Questions about Grad-CAM / Colors / Heatmaps
    if _has_keyword(msg, ["gradcam", "grad-cam", "heatmap", "heat map", "color", "colors", "red", "yellow", "blue", "overlay", "activation", "highlight"]):
        reply = (
            "### What Do the Colors on the Scan Mean?\n\n"
            "Think of this colored picture like a **highlighter pen** that the AI used to show you where its 'eyes' were looking!\n\n"
            "- **Red, Orange & Yellow (Warm Colors):** This is where the AI focused its **highest attention**. It spotted something here—like a cloudy patch, a shadow, or an unusual spot—that made it come to its conclusion.\n"
            "- **Blue & Purple (Cool Colors):** These are areas the AI looked at and found completely normal and healthy.\n\n"
            "> **In simple words:** The colors show you *where* and *why* the AI noticed something in your scan, so you and your doctor can see exactly what caught its attention."
        )
        suggestions = [
            "Explain my scan result in simple words",
            "How sure is the AI about this?",
            "What should I do next?"
        ]
        return reply, suggestions

    # 2. Questions about Prediction / Scan Findings
    if _has_keyword(msg, [
        "prediction", "result", "finding", "findings", "diagnos", "pneumonia", "covid", "tumor",
        "bleed", "hemorrhage", "dementia", "what does this mean", "explain", "why", "dangerous",
        "mean", "bad", "serious", "problem"
    ]):
        if prediction:
            reply = (
                f"### What Your Scan Means in Simple Words\n\n"
                f"- **Type of Scan:** **{modality.upper()}**\n"
                f"- **What the AI Detected:** **{prediction}**\n"
                f"- **How Sure the AI Is:** **{conf_str}**\n\n"
            )
            if "PNEUMONIA" in prediction:
                reply += (
                    "**What the AI Saw:**\n"
                    "In a healthy chest X-ray, lungs look dark because they are filled with air. The AI noticed **fuzzy, cloudy white patches** in your lungs. When there is a lung infection like pneumonia, fluid or mucus builds up and shows up as these hazy spots.\n\n"
                    "**Common Signs:**\n"
                    "People with pneumonia often have a cough, fever, feeling short of breath, or chest discomfort.\n\n"
                    "**What You Should Do Next:**\n"
                    "1. Share this report with your family doctor or visit a clinic.\n"
                    "2. Your doctor can listen to your breathing with a stethoscope and may give you medicine (like antibiotics or breathing treatments) to help you heal quickly.\n"
                    "3. Drink plenty of warm fluids and get lots of rest."
                )
            elif "COVID" in prediction:
                reply += (
                    "**What the AI Saw:**\n"
                    "On this chest CT scan, the AI noticed faint, cloudy patches in both lungs. Doctors often call these 'ground-glass' spots because they look like frosted bathroom glass instead of clear air.\n\n"
                    "**What It Means:**\n"
                    "This is a very common pattern seen when a viral infection like COVID-19 irritates the lung lining.\n\n"
                    "**What You Should Do Next:**\n"
                    "1. Take a quick home COVID-19 test to confirm.\n"
                    "2. Check your oxygen level using a finger pulse oximeter.\n"
                    "3. Rest, stay hydrated, and reach out to a healthcare professional if you feel breathless."
                )
            elif "BLEED" in prediction or "HEMORRHAGE" in prediction:
                reply += (
                    "**What the AI Saw:**\n"
                    "On this head CT scan, the AI detected an unusually bright white spot inside the skull. Because fresh blood is thicker than surrounding brain fluid, it shows up as a bright white patch on the scan.\n\n"
                    "**What It Means:**\n"
                    "It suggests there might be some bleeding inside or around the brain.\n\n"
                    "**Important Action:**\n"
                    "Any suspected head bleeding requires prompt evaluation by a hospital doctor or emergency team. If you or the person had a fall, sudden severe headache, or dizziness, visit an emergency room right away."
                )
            elif "TUMOR" in prediction:
                reply += (
                    "**What the AI Saw:**\n"
                    "On this brain MRI, the AI spotted an unusual lump or cluster of tissue that looks different from the surrounding healthy brain areas.\n\n"
                    "**What It Means:**\n"
                    "This is an initial AI flag showing a spot that needs closer doctor inspection. Many growths can be benign (non-cancerous) or simple harmless fluid pockets, so do not panic.\n\n"
                    "**What You Should Do Next:**\n"
                    "1. Have a brain specialist (neurologist) review the full scan.\n"
                    "2. They may order a contrast scan to see the exact tissue type and discuss the best next steps."
                )
            elif "DEMENTIA" in prediction:
                reply += (
                    "**What the AI Saw:**\n"
                    "On this brain MRI, the AI examined areas of the brain involved in memory and noticed subtle shrinkage or slightly larger fluid spaces than typical for younger adults.\n\n"
                    "**What It Means:**\n"
                    "This can be part of natural aging, or an early sign of mild memory changes.\n\n"
                    "**What You Should Do Next:**\n"
                    "1. Discuss your daily thinking and memory habits with your doctor.\n"
                    "2. They can run routine checks (including Vitamin B12 and thyroid levels) that can easily fix reversible memory issues."
                )
            else:
                reply += (
                    "**Great News:**\n"
                    "The AI examined this scan and found **no obvious signs of disease, damage, or bleeding**. The tissues appear clean, clear, and healthy!\n\n"
                    "**Remember:** A clear AI scan is very encouraging, but if you are experiencing any pain or symptoms, always trust how you feel and speak with your doctor."
                )
            suggestions = [
                "What do the colors on the scan mean?",
                "How do I download my PDF report?",
                "How accurate is this AI?"
            ]
            return reply, suggestions

    # 3. Questions about Confidence / Accuracy
    if _has_keyword(msg, ["confidence", "accuracy", "probability", "certain", "reliable", "how accurate", "sure", "trust"]):
        reply = (
            "### How Sure Is the AI About This Result?\n\n"
            f"The **confidence score ({conf_str})** is a percentage showing how closely your scan matches thousands of verified medical examples the AI learned from:\n\n"
            "- **High Score (Above 80%):** The pattern in your scan looks very clear and very similar to textbook examples of this condition.\n"
            "- **Moderate Score (60% to 80%):** The AI sees suggestive clues, but some features might be faint or overlap with normal variations.\n"
            "- **Low Score (Below 60%):** The AI is less sure. In these cases, a doctor's personal examination is especially important.\n\n"
            "> **Simple Rule:** The AI is a helpful smart assistant that highlights clues, but a licensed physician will always have the final say."
        )
        suggestions = [
            "Explain my scan result in simple words",
            "What do the colors on the scan mean?",
            "How do I download the PDF report?"
        ]
        return reply, suggestions

    # 4. Modality Differences (X-Ray vs CT vs MRI)
    if _has_keyword(msg, ["difference", "ct vs mri", "xray vs ct", "mri vs ct", "what is ct", "what is mri", "what is x-ray", "what is xray", "scan types"]):
        reply = (
            "### What Is the Difference Between X-Ray, CT, and MRI?\n\n"
            "Here is a simple breakdown of the three types of scans:\n\n"
            "- **X-Ray (The Quick 2D Photo):**\n"
            "  - Like taking a fast shadow photo of your bones and chest.\n"
            "  - Takes just a few seconds and uses very low radiation.\n"
            "  - Best for checking broken bones and chest infections like pneumonia.\n\n"
            "- **CT Scan (The 3D Slices):**\n"
            "  - Takes lots of cross-sectional pictures from all angles, like slicing a loaf of bread.\n"
            "  - Shows deep internal details of organs, lung tissue, and emergency brain bleeding.\n\n"
            "- **MRI (The Super-Detailed Magnet):**\n"
            "  - Uses safe magnetic fields and radio waves (zero radiation!).\n"
            "  - Gives the sharpest, clearest view of soft tissues, nerves, brain tumors, and memory areas."
        )
        suggestions = [
            "How does X-ray detect pneumonia?",
            "How does CT find brain bleeds?",
            "How does MRI detect tumors?"
        ]
        return reply, suggestions

    # 5. How to Download Report / Next Steps
    if _has_keyword(msg, ["pdf", "report", "download", "export", "print", "save"]):
        reply = (
            "### How to Download Your Medical Report\n\n"
            "You can save and print an official report in just a couple of clicks:\n\n"
            "1. Go to the top of the **Results** page.\n"
            "2. Click the button that says **'DOWNLOAD PDF REPORT'** (next to 'Analyze Another Scan').\n"
            "3. Your computer or phone will download a clean, easy-to-read document with your original scan and the AI highlight side-by-side.\n\n"
            "**Tip:** You can take this printed PDF directly to your doctor's appointment so they can review the AI's findings with you!"
        )
        suggestions = [
            "Explain my scan result in simple words",
            "What do the colors on the scan mean?",
            "Analyze another scan"
        ]
        return reply, suggestions

    # 6. Default / Friendly Welcome Response
    greeting = "Hello! I am your **MedVision AI Radiology Guide**.\n\n"
    context_note = ""
    if context and prediction:
        context_note = f"I am currently looking at your **{modality.upper()} scan**, which was flagged as **{prediction}** ({conf_str} confidence).\n\n"
    
    reply = (
        greeting + context_note +
        "I am here to explain everything in plain, simple English — with no confusing medical words!\n\n"
        "Here are a few things you can ask me:\n"
        "- *'What does my result mean?'*\n"
        "- *'What do the colored highlights on my scan show?'*\n"
        "- *'How sure is the AI about this?'*\n"
        "- *'What should my next steps be?'*\n"
        "- *'How do I download the PDF report?'*\n\n"
        "**What would you like to know?**"
    )
    suggestions = [
        "What does my scan result mean?",
        "What do the colors on the scan mean?",
        "How sure is the AI about this?",
        "How do I download the PDF report?"
    ]
    return reply, suggestions


async def generate_assistant_reply(request: ChatQueryRequest) -> ChatQueryResponse:
    """
    Handles user chat query. Attempts Gemini API if configured; seamlessly falls back
    to the Smart Clinical Knowledge Engine for offline, private, zero-latency responses.
    Both modes strictly adhere to simple, layperson-friendly, compassionate language.
    """
    message = request.message.strip()
    context = request.context
    history = request.history or []
    
    api_key = settings.GEMINI_API_KEY.strip()
    gemini_reply = None

    if api_key:
        try:
            # Build conversation turns for Gemini
            context_summary = _extract_context_summary(context)
            system_prompt = f"{SYSTEM_INSTRUCTION}\n\n{context_summary}"
            
            contents = []
            for h in history[-6:]:  # Keep recent history turns
                role = "user" if h.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": h.content}]})
            
            # Append current turn
            contents.append({"role": "user", "parts": [{"text": message}]})

            payload = {
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": contents,
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 800,
                    "topP": 0.95
                }
            }

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.GEMINI_MODEL}:generateContent?key={api_key}"
            async with httpx.AsyncClient(timeout=12.0) as client:
                res = await client.post(url, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            gemini_reply = parts[0].get("text", "").strip()
        except Exception as e:
            # Fall back gracefully to the built-in clinical engine
            gemini_reply = None

    if gemini_reply:
        suggestions = [
            "Can you explain the scan highlights?",
            "What should I ask my doctor?",
            "How do I download the PDF report?"
        ]
        return ChatQueryResponse(
            reply=gemini_reply,
            source="gemini",
            suggested_queries=suggestions,
            timestamp=datetime.now(timezone.utc).isoformat(),
            disclaimer=settings.DEFAULT_DISCLAIMER
        )

    # Use built-in Smart Clinical Knowledge Engine
    reply, suggestions = _generate_rule_based_reply(message, context)
    return ChatQueryResponse(
        reply=reply,
        source="clinical_engine",
        suggested_queries=suggestions,
        timestamp=datetime.now(timezone.utc).isoformat(),
        disclaimer=settings.DEFAULT_DISCLAIMER
    )
