"""Maps raw model output onto the Android app's result contract.

The trained model has 3 classes (see ../../GanoScan Model/results/*_model_info.json):
"Healthy", "Infected", "Initial Infection". Severity mirrors those 3 classes
directly (none / early / infected) rather than an invented finer-grained scale
the model has no data to support.
"""

from app.core.config import settings

CLASS_LABEL_ID = {
    "Healthy": "Pohon Sehat",
    "Initial Infection": "Terinfeksi Ganoderma (Awal)",
    "Infected": "Terinfeksi Ganoderma",
}

SEVERITY_OF = {
    "Healthy": "none",
    "Initial Infection": "early",
    "Infected": "infected",
}

RECOMMENDATIONS = {
    "none": [],
    "early": [
        "Isolasi & tandai pohon terinfeksi",
        "Tingkatkan pemantauan setiap 2 minggu",
        "Perbaiki drainase & sanitasi sekitar pangkal batang",
    ],
    "infected": [
        "Tumbang & musnahkan pohon terinfeksi segera",
        "Bongkar tunggul dan akar, sanitasi total lahan",
        "Karantina & pantau seluruh pohon radius 9 m",
    ],
}


def build_result(prob_map, predicted, confidence, image_bytes, model):
    """Shape raw model output into the JSON fields the app reads."""
    severity = SEVERITY_OF.get(predicted, "infected")
    infected = predicted != "Healthy"
    verdict = "INFECTED" if infected else "HEALTHY"

    dims = model.image_dimensions(image_bytes)
    resolution = f"{dims[0]}×{dims[1]} px" if dims else "—"
    gamma = settings.gamma

    stages = [
        {"index": "1", "title": "Citra Asli", "sub": f"input {resolution}"},
        {"index": "2", "title": "Gamma Correction", "sub": f"γ = {gamma} · kontras dinaikkan"},
        {"index": "3", "title": "CNN-Based Enhancement", "sub": "detail tekstur dipertajam"},
        {"index": "✓", "title": "Klasifikasi (CNN)",
         "sub": f"output: {'Terinfeksi' if infected else 'Sehat'} · {confidence:.3f}", "done": True},
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
