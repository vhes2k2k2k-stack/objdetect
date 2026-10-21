"""
app.py - Servidor Web Flask para la aplicación Web / PWA de Detección de Objetos
Permite instalar la aplicación directamente en Android desde Chrome / Edge.
"""

import os
import sys
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify

# Agregar el directorio raíz al path para importar detector
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from detector import ObjectDetector

app = Flask(__name__, template_folder="templates", static_folder="static")

# Inicializar detector
model_path = os.path.join(parent_dir, "models", "nanodet.onnx")
detector = ObjectDetector(model_path=model_path, prob_threshold=0.35, language="es")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Decodificar imagen Base64 enviada desde el navegador del móvil
        image_bytes = base64.b64decode(data["image"])
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # Parámetros opcionales
        threshold = float(data.get("threshold", 0.35))
        lang = data.get("lang", "es")
        detector.set_threshold(threshold)
        detector.language = lang

        orig_w = int(data.get("orig_w", frame_bgr.shape[1]))
        orig_h = int(data.get("orig_h", frame_bgr.shape[0]))

        # Inferencia
        detections = detector.detect(frame_bgr)

        # Escalar cajas al tamaño del canvas del móvil si difiere
        scale_x = orig_w / frame_bgr.shape[1]
        scale_y = orig_h / frame_bgr.shape[0]

        formatted = []
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            formatted.append({
                "box": [int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)],
                "confidence": round(d["confidence"], 2),
                "label": d["label"],
                "color": d["color"] # (B, G, R)
            })

        return jsonify({
            "status": "ok",
            "fps": round(detector.fps, 1),
            "detections": formatted
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Escuchar en todas las interfaces de red para que el celular Android en la misma WiFi pueda entrar
    app.run(host="0.0.0.0", port=5000, debug=False)
