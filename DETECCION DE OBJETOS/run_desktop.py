"""
run_desktop.py - Ejecutor de prueba local en Desktop (Windows/Linux/Mac)
Permite probar la detección de objetos con la cámara web del computador
antes de compilar e instalar la app en Android.
"""

import sys
import time
import cv2
import numpy as np
from detector import ObjectDetector

def main():
    print("=" * 60)
    print("   DETECTOR DE OBJETOS EN TIEMPO REAL (MODO DESKTOP / PC)")
    print("=" * 60)
    print("Iniciando cámara y cargando modelo de Inteligencia Artificial...")

    detector = ObjectDetector(model_path="models/nanodet.onnx", prob_threshold=0.35, language="es")

    # Intentar abrir cámara (índice 0 o 1)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] No se pudo abrir la cámara 0. Intentando con cámara 1...")
        cap = cv2.VideoCapture(1)

    has_camera = cap.isOpened()
    if not has_camera:
        print("[AVISO] No se detectó una cámara física conectada.")
        print("Activando modo de simulación con cuadro de prueba animado.")

    print("\n[CONTROLES DEL TECLADO]:")
    print("  'q' o ESC  : Salir de la aplicación")
    print("  '+' / '-'  : Aumentar / Disminuir umbral de confianza")
    print("  'l'        : Alternar idioma (Español / Inglés)")
    print("  'p'        : Pausar / Reanudar detección")
    print("  'c'        : Capturar foto con detecciones")
    print("-" * 60)

    paused = False
    threshold = 0.35
    lang = "es"
    sim_t = 0

    while True:
        if has_camera:
            ret, frame = cap.read()
            if not ret:
                print("Error al leer el cuadro de la cámara.")
                break
        else:
            # Cuadro sintético para pruebas sin cámara
            sim_t += 0.05
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (30, 30, 30)
            
            # Dibujar un objeto simulado que se mueve (ej. una persona o botella)
            cx = int(320 + 150 * np.sin(sim_t))
            cy = int(240 + 60 * np.cos(sim_t))
            cv2.circle(frame, (cx, cy - 60), 40, (180, 180, 180), -1)
            cv2.rectangle(frame, (cx - 50, cy - 20), (cx + 50, cy + 120), (120, 120, 200), -1)
            cv2.putText(frame, "Modo Simulacion: Conecte una camara web", (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Inferencia
        if not paused:
            detections = detector.detect(frame)
        else:
            detections = []

        # Dibujar resultados
        annotated = detector.draw_detections(frame, detections, show_fps=True)

        # Barra de estado inferior
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, h - 35), (w, h), (20, 20, 20), -1)
        status_text = f"Confianza: {int(threshold*100)}% | Idioma: {lang.upper()} | {'PAUSADO' if paused else 'EN VIVO'}"
        cv2.putText(annotated, status_text, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Mostrar ventana
        cv2.imshow("Detector de Objetos AI - Camara", annotated)

        # Manejo de teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key in (ord('+'), ord('=')):
            threshold = min(0.95, threshold + 0.05)
            detector.set_threshold(threshold)
            print(f"[+] Umbral de confianza ajustado a: {int(threshold * 100)}%")
        elif key in (ord('-'), ord('_')):
            threshold = max(0.10, threshold - 0.05)
            detector.set_threshold(threshold)
            print(f"[-] Umbral de confianza ajustado a: {int(threshold * 100)}%")
        elif key == ord('l'):
            lang = "en" if lang == "es" else "es"
            detector.language = lang
            print(f"[i] Idioma cambiado a: {lang.upper()}")
        elif key == ord('p'):
            paused = not paused
            print(f"[i] Detección {'Pausada' if paused else 'Reanudada'}")
        elif key == ord('c'):
            filename = f"captura_{int(time.time())}.jpg"
            cv2.imwrite(filename, annotated)
            print(f"[v] Captura guardada como: {filename}")

    if has_camera:
        cap.release()
    cv2.destroyAllWindows()
    print("Aplicación finalizada.")

if __name__ == "__main__":
    main()
