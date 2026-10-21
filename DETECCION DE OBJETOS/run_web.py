"""
run_web.py - Servidor para acceso e instalación móvil directa en Android.
Inicia el servidor y muestra la URL / IP local para abrirlo en el celular.
"""

import socket
import os
import sys

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No necesita conectarse realmente
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

if __name__ == '__main__':
    local_ip = get_local_ip()
    port = 5000

    print("=" * 65)
    print("   DETECTOR DE OBJETOS AI - SERVIDOR MOVIL / PWA ANDROID")
    print("=" * 65)
    print(f"\n[1] Para probar en esta computadora:")
    print(f"    👉 http://localhost:{port}")
    print(f"\n[2] Para probar e instalar en tu celular ANDROID:")
    print(f"    (Asegúrate de que tu celular esté conectado a la misma red WiFi)")
    print(f"    👉 Abre Google Chrome en tu celular y entra a:")
    print(f"       http://{local_ip}:{port}")
    print(f"\n[3] Para instalarla como App nativa en Android:")
    print(f"    En Google Chrome en tu celular:")
    print(f"    - Toca los 3 puntos (menú de Chrome)")
    print(f"    - Selecciona 'Instalar aplicación' o 'Agregar a la pantalla principal'")
    print(f"    - ¡Listo! Se creará un icono de app en tu móvil que abre en pantalla completa.")
    print("=" * 65)
    print("Iniciando servidor Flask...\n")

    sys.path.append(os.path.join(os.path.dirname(__file__), "web_app"))
    from web_app.app import app
    app.run(host="0.0.0.0", port=port, debug=False)
