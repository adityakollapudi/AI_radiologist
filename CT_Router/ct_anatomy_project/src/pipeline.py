"""Master Medical CT Multi-Model Triage Pipeline.

Execution Flow:
  1. Input CT Scan (DICOM, PNG, JPG, etc.)
       │
       ▼
  2. Model 1: CT Anatomy Router (best_router.pt)
       ├──> CHEST_CT  ──► Model 2: Chest Expert (COVID-19 / Pneumonia Detection)
       ├──> BRAIN_CT  ──► Model 3: Brain Expert (Intracranial Hemorrhage Detection)
       └──> UNCERTAIN ──► Manual Clinician Triage Flag (Ambiguous Scan)
"""

import time
from pathlib import Path
from typing import Union, Dict, Any, Optional
from PIL import Image

from .router import CTAnatomyRouter
from .chest_expert import ChestExpertModel
from .brain_expert import BrainExpertModel


class MedicalCTPipeline:
    """Master intelligent triage pipeline integrating all three models."""

    def __init__(self, router_path: Optional[Union[str, Path]] = None, device: Optional[str] = None):
        print("[Pipeline] Initializing Model 1 (CT Anatomy Router)...")
        self.router = CTAnatomyRouter(model_path=router_path) if router_path else CTAnatomyRouter(device=device)

        print("[Pipeline] Initializing Model 2 (Chest CT Expert)...")
        self.chest_expert = ChestExpertModel(device=device)

        print("[Pipeline] Initializing Model 3 (Brain CT Expert)...")
        self.brain_expert = BrainExpertModel(device=device)
        print("[Pipeline] All models initialized successfully!")

    def process(self, image_input: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """Runs end-to-end routing and downstream diagnosis."""
        t0 = time.time()

        # Step 1: Run Anatomy Router (Model 1)
        router_result = self.router.predict(image_input)
        anatomy = router_result["prediction"]
        is_uncertain = router_result["uncertain"]

        expert_result = None

        # Step 2: Route dynamically based on anatomy
        if is_uncertain:
            routing_action = "Sent to Manual Radiologist Review (Ambiguous Anatomy)"
        elif anatomy == "CHEST_CT":
            routing_action = "Dispatched to Chest CT Expert (COVID-19 / Pneumonia Detection)"
            expert_result = self.chest_expert.predict(image_input)
        elif anatomy == "BRAIN_CT":
            routing_action = "Dispatched to Brain CT Expert (Intracranial Hemorrhage Detection)"
            expert_result = self.brain_expert.predict(image_input)
        else:
            routing_action = "Unknown Anatomy"

        elapsed = round(time.time() - t0, 3)

        return {
            "image_path": router_result["image_path"],
            "router": router_result,
            "routing_action": routing_action,
            "expert_result": expert_result,
            "latency_seconds": elapsed,
        }
