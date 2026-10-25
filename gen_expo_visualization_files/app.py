from pathlib import Path
import json
import time
from flask import Flask, jsonify, render_template, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = DATA_DIR / "assets"
LATEST_PATH = DATA_DIR / "latest_match.json"
CONTROL_PATH = DATA_DIR / "control_command.json"
ALLOWED_CONTROLS = {"q", "a", "s", "r", "esc"}

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)


def _find_image_url(images: list, keyword: str) -> str | None:
    kw = (keyword or "").lower()
    for item in images or []:
        label = str((item or {}).get("label", "")).lower()
        url = (item or {}).get("url")
        if kw in label and url:
            return url
    return None


def _normalize_payload(payload: dict) -> dict:
    # Already in expected API envelope.
    if isinstance(payload, dict) and ("success" in payload) and ("data" in payload):
        return payload

    # New expo payload format written by ExFCC publisher.
    card = (payload or {}).get("card") if isinstance(payload, dict) else None
    if not isinstance(card, dict):
        return {
            "success": False,
            "message": "No matched cards yet. Run identification and click Update Example.",
        }

    hash_details = (payload or {}).get("hash_details") or {}
    confidence_details = (payload or {}).get("confidence_details") or {}
    localization_details = (payload or {}).get("localization_details") or {}
    images = (payload or {}).get("images") or []

    camera_image_url = _find_image_url(images, "camera")
    scryfall_image_url = _find_image_url(images, "scryfall") or card.get("scryfall_image_url")

    stages = []
    for item in images:
        label = (item or {}).get("label")
        url = (item or {}).get("url")
        if not label or not url:
            continue
        stages.append({
            "label": str(label),
            "url": str(url),
            "caption": "",
        })

    data = {
        "card_name": card.get("name", "-"),
        "set_name": card.get("set", "-"),
        "orientation_fixed": str(card.get("orientation", "")).lower() == "upright",
        "orientation_mode": card.get("orientation", ""),
        "orientation_angle_deg": card.get("orientation_angle_deg"),
        "best_distance": card.get("distance"),
        "card_cad_price": card.get("cad_price"),
        "scryfall_url": card.get("scryfall_uri") or "#",
        "camera_image_url": camera_image_url,
        "scryfall_image_url": scryfall_image_url,
        "camera_hash_hex": hash_details.get("camera_hash_hex"),
        "scryfall_hash_hex": hash_details.get("scryfall_hash_hex"),
        "hamming_distance": hash_details.get("hamming_distance"),
        "similarity_percent": hash_details.get("similarity_percent"),
        "camera_hash_bits": hash_details.get("camera_hash_binary_64"),
        "scryfall_hash_bits": hash_details.get("scryfall_hash_binary_64"),
        "confidence": {
            "score_10": confidence_details.get("confidence_score"),
            "final_score": confidence_details.get("final_score"),
            "weighted_matches": confidence_details.get("weighted_matches"),
            "matched_regions": confidence_details.get("matched_regions"),
            "stage1_distance": confidence_details.get("stage1_distance"),
            "explanation": confidence_details.get("explanation"),
        },
        "price_payload": card.get("prices") or {},
        "localization": {
            "yolo_confidence": localization_details.get("yolo_confidence", card.get("yolo_detector_confidence")),
        },
        "stages": stages,
    }

    return {
        "success": True,
        "data": data,
        "timestamp": payload.get("timestamp"),
    }


def _load_latest_payload() -> dict:
    if not LATEST_PATH.exists():
        return {
            "success": False,
            "message": "No matched cards yet. Run identification and click Update Example.",
        }
    try:
        with open(LATEST_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("latest_match.json must contain a JSON object")
        return _normalize_payload(payload)
    except Exception as exc:
        return {
            "success": False,
            "message": f"Could not read latest match payload: {exc}",
        }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/kids")
def kids():
    return render_template("kids.html")


@app.route("/detection-localization")
def detection_localization():
    return render_template("detection_localization.html")


@app.route("/api/latest-match", methods=["GET"])
def latest_match():
    return jsonify(_load_latest_payload())


@app.route("/api/control", methods=["POST"])
def control():
    payload = request.get_json(silent=True) or {}
    command = str(payload.get("command", "")).strip().lower()
    if command not in ALLOWED_CONTROLS:
        return jsonify({"success": False, "message": "Unsupported command."}), 400

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONTROL_PATH, "w", encoding="utf-8") as f:
        json.dump({"command": command, "timestamp": time.time()}, f)
    return jsonify({"success": True, "command": command})


@app.route("/api/assets/<path:filename>")
def get_asset(filename: str):
    return send_from_directory(str(ASSETS_DIR), filename)


def run_server(host: str = "127.0.0.1", port: int = 5050):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    run_server()
