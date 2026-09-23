"""Maps raw model output onto the Android app's result contract.

The trained model has 2 classes (see
../../GanoScan Model/results/01_model_info.json): "Healthy", "Infected".
Severity mirrors those 2 classes directly (none / infected).
"""

from app.core.config import settings

CLASS_LABEL_ID = {
    "Healthy": "Pohon Sehat",
    "Infected": "Terinfeksi Ganoderma",
}

SEVERITY_OF = {
    "Healthy": "none",
    "Infected": "infected",
}

RECOMMENDATIONS = {
    "none": [],
    "infected": [
        "Tumbang & musnahkan pohon terinfeksi segera",
        "Bongkar tunggul dan akar, sanitasi total lahan",
        "Karantina & pantau seluruh pohon radius 9 m",
    ],
}


def build_result(prob_map, predicted, confidence, image_bytes, model, pipeline):
    """Shape raw model output into the JSON fields the app reads.

    ``pipeline`` is the dict returned by GanodermaModel.predict() alongside
    the probabilities — None in random/mock mode (no real preprocessing ran,
    so there's nothing to show), otherwise it carries the gamma value that
    was actually used plus the 3 base64-encoded stage images.
    """
    severity = SEVERITY_OF.get(predicted, "infected")
    infected = predicted != "Healthy"
    verdict = "INFECTED" if infected else "HEALTHY"

    dims = model.image_dimensions(image_bytes)
    resolution = f"{dims[0]}×{dims[1]} px" if dims else "—"
    outcome_sub = f"output: {'Terinfeksi' if infected else 'Sehat'} · {confidence:.3f}"

    if pipeline is not None:
        gamma = pipeline["gamma"]
        ae_sub = "detail tekstur dipertajam" if pipeline["ae_ran"] else "dilewati — model AE tidak tersedia"
        stages = [
            {"index": "1", "title": "Citra Asli", "sub": f"input {resolution}", "image": pipeline["stage1_b64"]},
            {"index": "2", "title": "Gamma Correction", "sub": f"γ = {gamma} · kontras diacak", "image": pipeline["stage2_b64"]},
            {"index": "3", "title": "CNN-Based Enhancement", "sub": ae_sub, "image": pipeline["stage3_b64"]},
            {"index": "✓", "title": "Klasifikasi (CNN)", "sub": outcome_sub, "done": True},
        ]
    else:
        gamma = settings.gamma_values[0]
        stages = [
            {"index": "1", "title": "Citra Asli", "sub": f"input {resolution}"},
            {"index": "2", "title": "Gamma Correction", "sub": "mode acak (mock) — tidak diproses"},
            {"index": "3", "title": "CNN-Based Enhancement", "sub": "mode acak (mock) — tidak diproses"},
            {"index": "✓", "title": "Klasifikasi (CNN)", "sub": outcome_sub, "done": True},
        ]

    return {
        "verdict": verdict,
        "label": CLASS_LABEL_ID.get(predicted, predicted),
        "predictedClass": predicted,
        "severity": severity,
        "confidence": round(confidence, 4),
        "probabilities": {k: round(v, 4) for k, v in prob_map.items()},
        "gamma": gamma,
        "inputResolution": resolution,
        "recommendations": RECOMMENDATIONS.get(severity, []),
        "stages": stages,
    }
