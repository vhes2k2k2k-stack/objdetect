[app]

# (str) Title of your application
title = Detector Objetos AI

# (str) Package name
package.name = detectorobjetos

# (str) Package domain (needed for android packaging)
package.domain = org.ia.vision

# (str) Source code where the main.py lives
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,kv,atlas,onnx,txt

# (list) List of inclusions using pattern matching
source.include_patterns = models/*,assets/*

# (str) Application versioning
version = 1.0.0

# (list) Application requirements
# comma separated e.g. requirements = sqlite3,kivy
requirements = python3,kivy,numpy,opencv-python-headless,pillow,pyjnius

# (str) Supported orientation (one of landscape, sensorLandscape, portrait or all)
orientation = portrait

# (bool) Indicate if the application should be fullscreen
fullscreen = 0

# (list) Permissions
android.permissions = CAMERA, VIBRATE, INTERNET, WRITE_EXTERNAL_STORAGE, RECORD_AUDIO

# (list) Features
android.features = android.hardware.camera, android.hardware.camera.autofocus

# (int) Target Android API, should be as high as possible.
android.api = 33

# (int) Minimum API your APK will support.
android.minapi = 21

# (int) Android SDK version to use
android.sdk = 33

# (str) Android NDK version to use
android.ndk = 25b

# (bool) Use --private data storage (True) or --dir public storage (False)
android.private_storage = True

# (list) The Android architectures to build for.
# arm64-v8a is modern 64-bit Android (95%+ devices), armeabi-v7a is 32-bit legacy
android.archs = arm64-v8a, armeabi-v7a

# (bool) Allow backup of app data
android.allow_backup = True

# (bool) Automatically accept SDK license
android.accept_sdk_license = True

[buildozer]

# (int) Log level (0 = error only, 1 = info, 2 = debug with command output)
log_level = 2

# (int) Display warning if buildozer is run as root (0 = False, 1 = True)
warn_on_root = 0
