# 👁️ Detector de Objetos AI para Android (Python)

Aplicación móvil desarrollada en **Python** diseñada para instalarse en dispositivos **Android**, permitiendo al usuario ver e identificar en tiempo real los objetos que tiene frente a él a través de la cámara del teléfono celular.

---

## 🌟 Características Principales

- **Detección en Tiempo Real:** Identifica hasta **80 tipos de objetos cotidianos** (personas, celulares, botellas, sillas, computadoras, autos, perros, mochilas, etc.) utilizando una red neuronal **NanoDet ONNX** ultra ligera (3.8 MB) optimizada para procesadores móviles.
- **Visor Táctico con Cajas Delimitadoras:** Dibuja recuadros con esquinas acentuadas, porcentaje de confianza y etiquetas legibles con colores contrastados.
- **Soporte Bilingüe:** Nombres de objetos traducidos al **Español** e **Inglés**.
- **Lectura en Voz Alta (Text-To-Speech):** Puede anunciar por el altavoz del teléfono los objetos detectados (ideal para asistencia visual o uso manos libres).
- **Controles Táctiles Integrados:**
  - ⏯ **Pausar / Reanudar:** Congelar la detección en cualquier instante.
  - 🔄 **Cambio de Cámara:** Alternar entre la cámara trasera y la cámara frontal.
  - 🎚 **Control de Sensibilidad:** Ajustar el umbral de confianza (de 15% a 90%).
  - 🌐 **Selector de Idioma:** Alternar entre Español e Inglés en vivo.
- **Múltiples Formas de Instalación en Android:** Compilación de archivo `.apk` nativo (con Buildozer / Kivy) e instalación instantánea PWA (Progressive Web App).

---

## 📂 Estructura del Proyecto

```text
DETECCION DE OBJETOS/
├── .github/
│   └── workflows/
│       └── build_apk.yml       # Compilación automática del APK en la nube con GitHub Actions
├── models/
│   └── nanodet.onnx            # Modelo de Inteligencia Artificial (NanoDet - 80 clases COCO, 3.8 MB)
├── web_app/                    # Aplicación PWA para instalación directa desde Chrome Android
│   ├── static/
│   │   ├── manifest.json       # Manifiesto de Android PWA
│   │   └── sw.js               # Service Worker para ejecución offline
│   ├── templates/
│   │   └── index.html          # Interfaz táctil móvil
│   └── app.py                  # Servidor backend de inferencia
├── build_apk_colab.ipynb       # Cuaderno de Google Colab para compilar el APK gratis en la nube
├── buildozer.spec              # Configuración de empaquetado Android (SDK, permisos, cámara)
├── detector.py                 # Motor de visión por computadora e inferencia ONNX
├── main.py                     # Aplicación Android nativa desarrollada en Kivy
├── run_desktop.py              # Ejecutor para probar la app en tu PC con tu cámara web
├── run_web.py                  # Inicia la versión web instalable en tu celular
└── requirements.txt            # Dependencias de Python
```

---

## 🚀 Cómo Probarlo Inmediatamente en tu Computadora

Antes de instalar en tu teléfono, puedes probar la detección con la cámara web de tu PC:

```powershell
# 1. Instalar dependencias necesarias
pip install opencv-python numpy pillow flask

# 2. Ejecutar la prueba en escritorio
python run_desktop.py
```

**Teclas de control en PC:**
- `q` o `ESC`: Salir.
- `+` / `-`: Subir o bajar el umbral de confianza.
- `l`: Cambiar idioma (Español / Inglés).
- `p`: Pausar / reanudar.
- `c`: Tomar una captura de pantalla con los objetos detectados.

---

## 📱 Cómo Instalar la Aplicación en Android

Tienes **dos métodos principales**: compilar el archivo nativo `.apk` o instalarla en 30 segundos como Progressive Web App (PWA).

---

### Opción 1: Compilar el APK Nativo con GitHub Actions (1 Clic, Recomendado)

Dado que las herramientas de Android (NDK/SDK de Python) requieren un entorno Linux para compilar, el repositorio incluye un flujo automatizado de **GitHub Actions** que compila el APK en servidores de Ubuntu de forma 100% gratuita.

1. Sube este proyecto a tu cuenta de **GitHub** (repositorio público o privado):
   ```bash
   git init
   git add .
   git commit -m "Detector de Objetos Android"
   git branch -M main
   git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git
   git push -u origin main
   ```
2. Ve a la pestaña **Actions** en tu repositorio de GitHub.
3. Verás la tarea **Compilar APK Android (Buildozer)** ejecutándose.
4. En ~8 a 10 minutos, entra en la ejecución completada y descarga el archivo en **Artifacts**:
   - `detector-objetos-android-apk.zip` -> Contiene tu archivo `.apk`.
5. Pasa el archivo `.apk` a tu teléfono Android (por WhatsApp, Telegram, Google Drive o cable USB) y tócalo para instalarlo.

---

### Opción 2: Compilar el APK Gratis en Google Colab

Si no deseas usar GitHub, puedes compilar el APK en la nube usando Google Colab:

1. Abre [Google Colab](https://colab.research.google.com/).
2. Haz clic en **Subir** y selecciona el archivo [`build_apk_colab.ipynb`](build_apk_colab.ipynb).
3. Sube la carpeta de este proyecto al panel izquierdo de archivos de Colab.
4. Ejecuta las celdas en orden.
5. Al finalizar, la última celda descargará automáticamente el archivo `.apk` a tu navegador.

---

### Opción 3: Compilar Localmente con WSL (Linux en Windows)

Si tienes WSL (Ubuntu) instalado en tu Windows:

```bash
# Dentro de Ubuntu WSL:
sudo apt update
sudo apt install -y build-essential ccache git libffi-dev libssl-dev openjdk-17-jdk zlib1g-dev
pip install --upgrade "cython<3.0.0" buildozer
buildozer android debug
```
El archivo `.apk` resultante se guardará en la carpeta `bin/`.

---

### Opción 4: Instalación Rápida como PWA en Android (Sin Esperas de Compilación)

Puedes instalar la aplicación directamente en la pantalla de inicio de tu teléfono en 1 minuto:

1. En tu computadora, ejecuta:
   ```powershell
   python run_web.py
   ```
2. Verás un mensaje en pantalla indicando la dirección IP local de tu PC, por ejemplo:
   `http://192.168.1.50:5000`
3. En tu celular Android (conectado a la misma red WiFi), abre **Google Chrome** y entra a esa dirección.
4. En Chrome, pulsa el botón **"Instalar"** que aparecerá en la pantalla (o abre el menú de 3 puntos de Chrome y selecciona **"Instalar aplicación"** / **"Agregar a la pantalla principal"**).
5. Se creará el icono de **Detector de Objetos AI** en tu pantalla de inicio de Android. Al abrirlo, funcionará en pantalla completa con acceso directo a tu cámara trasera y delantera.

---

## 🔒 Permisos en Android

Al abrir la aplicación en tu celular por primera vez, el sistema solicitará:
- **Cámara (`android.permission.CAMERA`):** Necesario para capturar el video en tiempo real.
- **Almacenamiento (`android.permission.WRITE_EXTERNAL_STORAGE`):** Para guardar capturas si lo deseas.
- **Audio / Vibración:** Para la respuesta háptica y lectura en voz alta.

> **Nota para instalar APKs:** Si instalas el `.apk` directamente, Android te preguntará: *"¿Permitir instalar aplicaciones de fuentes desconocidas?"*. Selecciona **Permitir** en la configuración de la app desde la cual lo abriste (Chrome o Administrador de Archivos).

---

## 🧠 Clases de Objetos que Puede Detectar (80 Clases COCO)

- **Personas y Accesorios:** Persona, Mochila, Paraguas, Bolso, Corbata, Maleta.
- **Vehículos:** Bicicleta, Carro/Auto, Motocicleta, Avión, Autobús, Tren, Camión, Barco.
- **Animales:** Pájaro, Gato, Perro, Caballo, Oveja, Vaca, Elefante, Oso, Cebra, Jirafa.
- **Dispositivos Electrónicos:** Celular, Laptop, Mouse, Control Remoto, Teclado, Televisor, Microondas, Horno, Tostadora, Refrigerador, Reloj.
- **Hogar y Muebles:** Silla, Sofá, Planta, Cama, Mesa de comedor, Inodoro, Lavamanos, Florero, Libro.
- **Comida y Utensilios:** Botella, Copa, Taza/Vaso, Tenedor, Cuchillo, Cuchara, Tazón, Plátano, Manzana, Sándwich, Naranja, Brócoli, Zanahoria, Hot Dog, Pizza, Dona, Pastel.
- **Deportes:** Pelota, Frisbee, Esquís, Snowboard, Cometa, Bate, Guante, Patineta, Tabla de Surf, Raqueta.
