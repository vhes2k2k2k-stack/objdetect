"""
detector.py - Motor de Detección de Objetos con NanoDet (ONNX)
Compatible con Android (Kivy / Buildozer), Desktop (OpenCV) y Web.
"""

import os
import time
import numpy as np
import cv2

# Diccionario de 80 clases COCO con traducción al español
COCO_CLASSES = [
    ("person", "Persona"),
    ("bicycle", "Bicicleta"),
    ("car", "Carro / Auto"),
    ("motorcycle", "Motocicleta"),
    ("airplane", "Avión"),
    ("bus", "Autobús"),
    ("train", "Tren"),
    ("truck", "Camión"),
    ("boat", "Barco"),
    ("traffic light", "Semáforo"),
    ("fire hydrant", "Hidrante"),
    ("stop sign", "Señal de Alto"),
    ("parking meter", "Parquímetro"),
    ("bench", "Banca"),
    ("bird", "Pájaro"),
    ("cat", "Gato"),
    ("dog", "Perro"),
    ("horse", "Caballo"),
    ("sheep", "Oveja"),
    ("cow", "Vaca"),
    ("elephant", "Elefante"),
    ("bear", "Oso"),
    ("zebra", "Cebra"),
    ("giraffe", "Jirafa"),
    ("backpack", "Mochila"),
    ("umbrella", "Paraguas"),
    ("handbag", "Bolso"),
    ("tie", "Corbata"),
    ("suitcase", "Maleta"),
    ("frisbee", "Frisbee"),
    ("skis", "Esquís"),
    ("snowboard", "Snowboard"),
    ("sports ball", "Pelota"),
    ("kite", "Cometa"),
    ("baseball bat", "Bate"),
    ("baseball glove", "Guante"),
    ("skateboard", "Patineta"),
    ("surfboard", "Tabla de Surf"),
    ("tennis racket", "Raqueta"),
    ("bottle", "Botella"),
    ("wine glass", "Copa"),
    ("cup", "Taza / Vaso"),
    ("fork", "Tenedor"),
    ("knife", "Cuchillo"),
    ("spoon", "Cuchara"),
    ("bowl", "Tazón"),
    ("banana", "Plátano"),
    ("apple", "Manzana"),
    ("sandwich", "Sándwich"),
    ("orange", "Naranja"),
    ("broccoli", "Brócoli"),
    ("carrot", "Zanahoria"),
    ("hot dog", "Hot Dog"),
    ("pizza", "Pizza"),
    ("donut", "Dona"),
    ("cake", "Pastel"),
    ("chair", "Silla"),
    ("couch", "Sofá"),
    ("potted plant", "Planta"),
    ("bed", "Cama"),
    ("dining table", "Mesa"),
    ("toilet", "Inodoro"),
    ("tv", "Televisor"),
    ("laptop", "Laptop"),
    ("mouse", "Mouse"),
    ("remote", "Control Remoto"),
    ("keyboard", "Teclado"),
    ("cell phone", "Celular"),
    ("microwave", "Microondas"),
    ("oven", "Horno"),
    ("toaster", "Tostadora"),
    ("sink", "Lavamanos"),
    ("refrigerator", "Refrigerador"),
    ("book", "Libro"),
    ("clock", "Reloj"),
    ("vase", "Florero"),
    ("scissors", "Tijeras"),
    ("teddy bear", "Oso de Peluche"),
    ("hair drier", "Secador"),
    ("toothbrush", "Cepillo de Dientes")
]

# Paleta de colores atractiva para las cajas de detección
COLOR_PALETTE = [
    (46, 204, 113),   # Esmeralda
    (52, 152, 219),   # Azul
    (155, 89, 182),   # Amatista
    (241, 196, 15),   # Amarillo
    (230, 126, 34),   # Naranja
    (231, 76, 60),    # Rojo coral
    (26, 188, 156),   # Turquesa
    (243, 156, 18),   # Ámbar
    (211, 84, 0),     # Calabaza
    (192, 57, 43),    # Granate
    (142, 68, 173),   # Morado oscuro
    (41, 128, 185),   # Azul cobalto
    (39, 174, 96),    # Verde bosque
]

class ObjectDetector:
    """
    Detector de objetos en tiempo real basado en NanoDet ONNX.
    Optimizado para dispositivos móviles y procesamiento en CPU.
    """
    def __init__(self, model_path=None, prob_threshold=0.35, iou_threshold=0.5, language="es"):
        if model_path is None:
            # Buscar en directorio local o en 'models'
            base_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(base_dir, "models", "nanodet.onnx")
            if os.path.exists(candidate):
                model_path = candidate
            else:
                model_path = "models/nanodet.onnx"

        self.model_path = model_path
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.language = language

        # Parámetros del modelo NanoDet
        self.strides = (8, 16, 32)
        self.image_shape = (416, 416)
        self.reg_max = 7
        self.project = np.arange(self.reg_max + 1)
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)

        # Cargar red neuronal con OpenCV DNN
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No se encontró el modelo ONNX en: {self.model_path}")

        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        
        # Precomputar anclas
        self.anchors_mlvl = []
        for stride in self.strides:
            feat_h = int(self.image_shape[0] / stride)
            feat_w = int(self.image_shape[1] / stride)
            shift_x = np.arange(0, feat_w) * stride
            shift_y = np.arange(0, feat_h) * stride
            xv, yv = np.meshgrid(shift_x, shift_y)
            cx = xv.flatten() + 0.5 * (stride - 1)
            cy = yv.flatten() + 0.5 * (stride - 1)
            self.anchors_mlvl.append(np.column_stack((cx, cy)))

        # Control de FPS
        self.prev_time = time.time()
        self.fps = 0.0

    def set_threshold(self, threshold: float):
        """Ajusta el umbral de confianza dinámicamente."""
        self.prob_threshold = max(0.1, min(0.95, threshold))

    def get_label(self, class_id: int) -> str:
        """Devuelve el nombre de la clase según el idioma seleccionado."""
        if 0 <= class_id < len(COCO_CLASSES):
            en_name, es_name = COCO_CLASSES[class_id]
            return es_name if self.language == "es" else en_name
        return f"Obj_{class_id}"

    def get_color(self, class_id: int):
        """Devuelve un color BGR único y consistente para cada clase."""
        return COLOR_PALETTE[class_id % len(COLOR_PALETTE)]

    def detect(self, frame_bgr: np.ndarray):
        """
        Ejecuta la inferencia sobre un cuadro (imagen BGR de OpenCV o numpy array).
        Retorna:
            detections (list): Lista de dicts con {'box', 'confidence', 'class_id', 'label', 'color'}
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        # Actualizar FPS
        now = time.time()
        dt = now - self.prev_time
        if dt > 0:
            current_fps = 1.0 / dt
            self.fps = 0.9 * self.fps + 0.1 * current_fps if self.fps > 0 else current_fps
        self.prev_time = now

        h_orig, w_orig = frame_bgr.shape[:2]

        # Preprocesamiento NanoDet
        resized = cv2.resize(frame_bgr, self.image_shape)
        img_norm = (resized.astype(np.float32) - self.mean) / self.std
        blob = cv2.dnn.blobFromImage(img_norm)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        cls_scores = outs[:3]
        bbox_preds = outs[3:]

        bboxes_mlvl, scores_mlvl = [], []
        for stride, cls_score, bbox_pred, anchors in zip(self.strides, cls_scores, bbox_preds, self.anchors_mlvl):
            cls_score = cls_score.squeeze(axis=0)
            bbox_pred = bbox_pred.squeeze(axis=0)

            x_exp = np.exp(bbox_pred.reshape(-1, self.reg_max + 1))
            bbox_pred = (x_exp / np.sum(x_exp, axis=1, keepdims=True)).dot(self.project).reshape(-1, 4) * stride

            points = anchors
            x1 = np.clip(points[:, 0] - bbox_pred[:, 0], 0, self.image_shape[1])
            y1 = np.clip(points[:, 1] - bbox_pred[:, 1], 0, self.image_shape[0])
            x2 = np.clip(points[:, 0] + bbox_pred[:, 2], 0, self.image_shape[1])
            y2 = np.clip(points[:, 1] + bbox_pred[:, 3], 0, self.image_shape[0])

            bboxes_mlvl.append(np.column_stack([x1, y1, x2, y2]))
            scores_mlvl.append(cls_score)

        bboxes = np.concatenate(bboxes_mlvl, axis=0)
        scores = np.concatenate(scores_mlvl, axis=0)

        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes_wh[:, 2:4] - bboxes_wh[:, 0:2]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.prob_threshold, self.iou_threshold)

        detections = []
        if len(indices) > 0:
            scale_x = w_orig / self.image_shape[1]
            scale_y = h_orig / self.image_shape[0]
            for idx in np.array(indices).flatten():
                box = bboxes[idx]
                x1 = max(0, int(box[0] * scale_x))
                y1 = max(0, int(box[1] * scale_y))
                x2 = min(w_orig, int(box[2] * scale_x))
                y2 = min(h_orig, int(box[3] * scale_y))
                conf = float(confidences[idx])
                cls_id = int(class_ids[idx])
                label = self.get_label(cls_id)
                color = self.get_color(cls_id)

                detections.append({
                    "box": (x1, y1, x2, y2),
                    "confidence": conf,
                    "class_id": cls_id,
                    "label": label,
                    "color": color
                })

        return detections

    def draw_detections(self, frame_bgr: np.ndarray, detections: list, show_fps=True):
        """
        Dibuja cajas delimitadoras estilizadas y etiquetas legibles en la imagen.
        """
        out_frame = frame_bgr.copy()
        h, w = out_frame.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            color = det["color"]
            label = det["label"]
            conf = det["confidence"]
            text = f"{label} {int(conf * 100)}%"

            # Rectángulo exterior con esquinas decorativas
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Pequeños bordes acentuados en las 4 esquinas (estilo visor táctico)
            corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
            if corner_len > 0:
                # Arriba izquierda
                cv2.line(out_frame, (x1, y1), (x1 + corner_len, y1), color, 4, cv2.LINE_AA)
                cv2.line(out_frame, (x1, y1), (x1, y1 + corner_len), color, 4, cv2.LINE_AA)
                # Arriba derecha
                cv2.line(out_frame, (x2, y1), (x2 - corner_len, y1), color, 4, cv2.LINE_AA)
                cv2.line(out_frame, (x2, y1), (x2, y1 + corner_len), color, 4, cv2.LINE_AA)
                # Abajo izquierda
                cv2.line(out_frame, (x1, y2), (x1 + corner_len, y2), color, 4, cv2.LINE_AA)
                cv2.line(out_frame, (x1, y2), (x1, y2 - corner_len), color, 4, cv2.LINE_AA)
                # Abajo derecha
                cv2.line(out_frame, (x2, y2), (x2 - corner_len, y2), color, 4, cv2.LINE_AA)
                cv2.line(out_frame, (x2, y2), (x2, y2 - corner_len), color, 4, cv2.LINE_AA)

            # Fondo para la etiqueta de texto
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.55
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            badge_y1 = max(0, y1 - text_h - 10)
            badge_y2 = y1
            badge_x1 = x1
            badge_x2 = min(w, x1 + text_w + 14)

            # Insignia de fondo
            cv2.rectangle(out_frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
            # Texto contrastado (blanco o negro según luminosidad)
            text_color = (255, 255, 255)
            cv2.putText(out_frame, text, (badge_x1 + 6, badge_y2 - 5), font, font_scale, text_color, thickness, cv2.LINE_AA)

        if show_fps:
            fps_text = f"FPS: {self.fps:.1f} | Objetos: {len(detections)}"
            cv2.putText(out_frame, fps_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out_frame, fps_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 1, cv2.LINE_AA)

        return out_frame
