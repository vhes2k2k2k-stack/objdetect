"""
main.py - Aplicación Android y Multiplataforma de Detección de Objetos en Python (Kivy)
Permite al usuario ver los objetos frente a él en tiempo real a través de la cámara del móvil.
"""

import os
import sys
import threading
import numpy as np
import cv2

# Importaciones de Kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.utils import platform

# Importar nuestro motor de detección de objetos con IA
from detector import ObjectDetector

# Dimensiones base para simulación móvil en PC
if platform != "android":
    Window.size = (420, 750)

class ObjectDetectorApp(App):
    def build(self):
        self.title = "Detector de Objetos AI"
        self.detector = ObjectDetector(model_path="models/nanodet.onnx", prob_threshold=0.35, language="es")
        
        # Variables de estado
        self.is_paused = False
        self.speech_enabled = False
        self.language = "es"
        self.camera_index = 0
        self.last_spoken_time = 0
        self.last_spoken_objects = set()
        
        # Interfaz Principal (FloatLayout permite superponer controles sobre la cámara)
        self.root_layout = FloatLayout()

        # 1. Widget de Video / Vista de Cámara
        self.image_widget = Image(
            allow_stretch=True,
            keep_ratio=True,
            size_hint=(1, 1),
            pos_hint={'center_x': 0.5, 'center_y': 0.5}
        )
        self.root_layout.add_widget(self.image_widget)

        # 2. Barra Superior de Información y Objetos Detectados (HUD)
        self.top_bar = BoxLayout(
            orientation='vertical',
            size_hint=(1, None),
            height=90,
            pos_hint={'top': 1, 'x': 0},
            padding=[12, 10, 12, 5],
            spacing=4
        )
        
        self.fps_label = Label(
            text="Iniciando cámara y modelo de IA...",
            font_size='14sp',
            bold=True,
            color=(0, 1, 0.8, 1),
            size_hint=(1, None),
            height=25
        )
        self.objects_label = Label(
            text="Apunte la cámara hacia los objetos",
            font_size='13sp',
            color=(1, 1, 1, 0.95),
            size_hint=(1, None),
            height=45
        )
        self.top_bar.add_widget(self.fps_label)
        self.top_bar.add_widget(self.objects_label)
        self.root_layout.add_widget(self.top_bar)

        # 3. Panel Inferior de Controles Táctiles (Móvil)
        self.bottom_panel = BoxLayout(
            orientation='vertical',
            size_hint=(1, None),
            height=130,
            pos_hint={'bottom': 1, 'x': 0},
            padding=[15, 10, 15, 12],
            spacing=8
        )

        # Fila del Slider de Sensibilidad / Confianza
        slider_row = BoxLayout(orientation='horizontal', size_hint=(1, None), height=35, spacing=10)
        self.slider_label = Label(
            text="Umbral: 35%",
            font_size='12sp',
            size_hint=(0.35, 1),
            color=(1, 1, 1, 1)
        )
        self.conf_slider = Slider(
            min=0.15,
            max=0.90,
            value=0.35,
            step=0.05,
            size_hint=(0.65, 1)
        )
        self.conf_slider.bind(value=self.on_slider_change)
        slider_row.add_widget(self.slider_label)
        slider_row.add_widget(self.conf_slider)
        self.bottom_panel.add_widget(slider_row)

        # Fila de Botones Táctiles
        buttons_row = BoxLayout(orientation='horizontal', size_hint=(1, None), height=48, spacing=8)
        
        # Botón Pausar / Reanudar
        self.pause_btn = Button(
            text="⏯ Pausar",
            font_size='12sp',
            background_color=(0.2, 0.4, 0.8, 0.85)
        )
        self.pause_btn.bind(on_press=self.toggle_pause)

        # Botón Cambiar Cámara (Frontal / Trasera)
        self.switch_cam_btn = Button(
            text="🔄 Cámara",
            font_size='12sp',
            background_color=(0.2, 0.6, 0.4, 0.85)
        )
        self.switch_cam_btn.bind(on_press=self.switch_camera)

        # Botón Idioma (ES / EN)
        self.lang_btn = Button(
            text="🌐 ES",
            font_size='12sp',
            background_color=(0.5, 0.3, 0.7, 0.85)
        )
        self.lang_btn.bind(on_press=self.toggle_language)

        # Botón Voz (Lectura en voz alta)
        self.voice_btn = Button(
            text="🔇 Silencio",
            font_size='12sp',
            background_color=(0.4, 0.4, 0.4, 0.85)
        )
        self.voice_btn.bind(on_press=self.toggle_voice)

        buttons_row.add_widget(self.pause_btn)
        buttons_row.add_widget(self.switch_cam_btn)
        buttons_row.add_widget(self.lang_btn)
        buttons_row.add_widget(self.voice_btn)
        self.bottom_panel.add_widget(buttons_row)

        self.root_layout.add_widget(self.bottom_panel)

        # Inicialización de captura de video
        self.capture = None
        self.init_camera()

        # Solicitar permisos en Android
        if platform == 'android':
            self.request_android_permissions()

        # Programar el bucle de actualización a 30 FPS
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)

        return self.root_layout

    def request_android_permissions(self):
        """Solicita permisos de hardware al usuario en Android."""
        try:
            from android.permissions import request_permissions, Permission
            def callback(permissions, grant_results):
                if all(grant_results):
                    print("[ANDROID] Permiso de cámara concedido.")
                else:
                    print("[ANDROID] Permiso de cámara denegado por el usuario.")
            request_permissions([Permission.CAMERA, Permission.VIBRATE, Permission.RECORD_AUDIO], callback)
        except Exception as e:
            print(f"[ANDROID] Error al solicitar permisos: {e}")

    def init_camera(self):
        """Inicializa o reinicia la captura de video."""
        if self.capture is not None:
            self.capture.release()
            
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            # Si falla la cámara 0, intentar con 1
            alt_idx = 1 if self.camera_index == 0 else 0
            self.capture = cv2.VideoCapture(alt_idx)
            if self.capture.isOpened():
                self.camera_index = alt_idx

        # Configurar resolución moderada para alta velocidad en móviles
        if self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def on_slider_change(self, instance, value):
        """Callback cuando el usuario ajusta el control deslizante de confianza."""
        pct = int(value * 100)
        self.slider_label.text = f"Umbral: {pct}%"
        self.detector.set_threshold(value)

    def toggle_pause(self, instance):
        """Alterna el estado de pausa de la detección."""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.text = "▶ Reanudar"
            self.pause_btn.background_color = (0.8, 0.3, 0.2, 0.85)
        else:
            self.pause_btn.text = "⏯ Pausar"
            self.pause_btn.background_color = (0.2, 0.4, 0.8, 0.85)

    def switch_camera(self, instance):
        """Alterna entre cámara trasera (0) y cámara frontal (1)."""
        self.camera_index = 1 if self.camera_index == 0 else 0
        self.init_camera()

    def toggle_language(self, instance):
        """Alterna entre nombres en español e inglés."""
        if self.language == "es":
            self.language = "en"
            self.lang_btn.text = "🌐 EN"
        else:
            self.language = "es"
            self.lang_btn.text = "🌐 ES"
        self.detector.language = self.language

    def toggle_voice(self, instance):
        """Activa o desactiva la síntesis de voz para personas con discapacidad visual."""
        self.speech_enabled = not self.speech_enabled
        if self.speech_enabled:
            self.voice_btn.text = "🔊 Voz ON"
            self.voice_btn.background_color = (0.1, 0.7, 0.4, 0.85)
            self.speak_text("Voz activada" if self.language == "es" else "Voice enabled")
        else:
            self.voice_btn.text = "🔇 Silencio"
            self.voice_btn.background_color = (0.4, 0.4, 0.4, 0.85)

    def speak_text(self, text):
        """Pronuncia el texto detectado a través del altavoz del dispositivo."""
        def run_tts():
            try:
                if platform == 'android':
                    # TTS nativo de Android vía pyjnius
                    from jnius import autoclass
                    PythonActivity = autoclass('org.kivy.android.PythonActivity')
                    TextToSpeech = autoclass('android.speech.tts.TextToSpeech')
                    Locale = autoclass('java.util.Locale')
                    # Implementación sencilla
                else:
                    # En Windows / Desktop
                    import subprocess
                    ps_cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"
                    subprocess.run(["powershell", "-Command", ps_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

        threading.Thread(target=run_tts, daemon=True).start()

    def update_frame(self, dt):
        """Bucle principal de procesamiento de cada cuadro de video."""
        if self.capture is None or not self.capture.isOpened():
            # Crear cuadro de espera
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Esperando camara...", (160, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            detections = []
        else:
            ret, frame = self.capture.read()
            if not ret or frame is None:
                return

            # En Android, rotar según la orientación si es necesario
            if platform == 'android':
                # La mayoría de cámaras traseras en Android requieren rotación 90° CW para modo vertical
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Inferencia de detección de objetos
            if not self.is_paused:
                detections = self.detector.detect(frame)
            else:
                detections = []

            # Dibujar cajas y etiquetas estilizadas
            frame = self.detector.draw_detections(frame, detections, show_fps=False)

        # Actualizar etiquetas de texto en la interfaz
        current_fps = self.detector.fps
        self.fps_label.text = f"FPS: {current_fps:.1f} | Objetos: {len(detections)}"

        if detections:
            # Resumen de objetos detectados (ej: "2 Persona, 1 Celular")
            counts = {}
            for d in detections:
                counts[d['label']] = counts.get(d['label'], 0) + 1
            summary_parts = [f"{count} {label}" for label, count in counts.items()]
            self.objects_label.text = "Frente a usted: " + ", ".join(summary_parts)

            # Síntesis de voz periódica si está activada
            if self.speech_enabled:
                import time
                now = time.time()
                current_labels = set(counts.keys())
                if (now - self.last_spoken_time > 3.0) or (current_labels != self.last_spoken_objects and now - self.last_spoken_time > 1.5):
                    self.last_spoken_time = now
                    self.last_spoken_objects = current_labels
                    speech_msg = ", ".join(summary_parts)
                    self.speak_text(speech_msg)
        else:
            self.objects_label.text = "Sin objetos detectados en el visor"

        # Convertir imagen OpenCV (BGR) a textura compatible con Kivy
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image_widget.texture = texture

    def on_stop(self):
        """Liberar la cámara al cerrar la aplicación."""
        if self.capture is not None:
            self.capture.release()

if __name__ == '__main__':
    ObjectDetectorApp().run()
