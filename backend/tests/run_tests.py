"""Automated test runner for MEDVISION AI Backend.
Supports executing tests without external test runner dependencies.
"""
import sys
import traceback
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.tests.test_health_and_capabilities import (
    test_health_endpoint,
    test_api_health_endpoint,
    test_modalities_endpoint,
    test_models_specs_endpoint
)
from backend.tests.test_validation_and_errors import (
    test_unsupported_modality,
    test_invalid_file_extension,
    test_empty_file,
    test_corrupted_image_content
)
from backend.tests.test_modality_adapters_mocked import (
    test_xray_mocked_adapter,
    test_ct_mocked_adapter,
    test_mri_mocked_adapter
)
from backend.tests.test_pdf_report import (
    test_pdf_endpoint_missing_original_image,
    test_pdf_endpoint_with_missing_gradcam,
    test_real_xray_pdf_generation,
    test_real_ct_pdf_generation,
    test_real_mri_pdf_generation
)
from backend.tests.test_assistant import (
    test_assistant_general_query,
    test_assistant_contextual_scan_query,
    test_assistant_gradcam_query
)

UNIT_TESTS = [
    ("Health: /health", test_health_endpoint),
    ("Health: /api/health", test_api_health_endpoint),
    ("Modalities: /modalities", test_modalities_endpoint),
    ("Models: /models specs", test_models_specs_endpoint),
    ("Validation: Unsupported Modality", test_unsupported_modality),
    ("Validation: Invalid Extension", test_invalid_file_extension),
    ("Validation: Empty File", test_empty_file),
    ("Validation: Corrupted Image", test_corrupted_image_content),
    ("Adapter Mock: X-Ray", test_xray_mocked_adapter),
    ("Adapter Mock: CT Scan", test_ct_mocked_adapter),
    ("Adapter Mock: Brain MRI", test_mri_mocked_adapter),
    ("PDF Report: Missing Original Image Validation", test_pdf_endpoint_missing_original_image),
    ("PDF Report: Missing Grad-CAM Graceful Handling", test_pdf_endpoint_with_missing_gradcam),
    ("AI Assistant: General Medical Query", test_assistant_general_query),
    ("AI Assistant: Contextual Scan Query", test_assistant_contextual_scan_query),
    ("AI Assistant: Grad-CAM Explainability Query", test_assistant_gradcam_query),
]

def run_all(include_smoke=False):
    print("=" * 70)
    print("MEDVISION AI — AUTOMATED BACKEND TEST SUITE")
    print("=" * 70)
    passed = 0
    failed = 0

    for name, test_fn in UNIT_TESTS:
        try:
            test_fn()
            print(f" [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f" [FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1

    if include_smoke:
        print("\n" + "-" * 70)
        print("REAL MODEL INFERENCE SMOKE TESTS (LIVE CHECKPOINTS)")
        print("-" * 70)
        from backend.tests.test_real_inference_smoke import (
            test_real_xray_inference_smoke,
            test_real_ct_inference_smoke,
            test_real_mri_inference_smoke
        )
        SMOKE_TESTS = [
            ("Real Inference: X-Ray Pneumonia Ensemble", test_real_xray_inference_smoke),
            ("Real Inference: CT Scan Multi-Model Pipeline", test_real_ct_inference_smoke),
            ("Real Inference: Brain MRI Dementia/Tumor Pipeline", test_real_mri_inference_smoke),
            ("Real PDF Report: Real X-Ray Scan to Downloadable PDF", test_real_xray_pdf_generation),
            ("Real PDF Report: Real CT Scan to Downloadable PDF", test_real_ct_pdf_generation),
            ("Real PDF Report: Real MRI Scan to Downloadable PDF", test_real_mri_pdf_generation),
        ]
        for name, test_fn in SMOKE_TESTS:
            try:
                print(f" [RUNNING] {name}...")
                test_fn()
                print(f" [PASS] {name}")
                passed += 1
            except Exception as e:
                print(f" [FAIL] {name}: {e}")
                traceback.print_exc()
                failed += 1

    print("=" * 70)
    print(f"SUMMARY: {passed} PASSED, {failed} FAILED")
    print("=" * 70)
    return failed == 0

if __name__ == "__main__":
    include_smoke = "--smoke" in sys.argv
    success = run_all(include_smoke=include_smoke)
    sys.exit(0 if success else 1)
