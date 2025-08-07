# analizar_descargas.py

def analizar_log(debug=False):
    archivo = "notebooks/download_log.txt"
    
    # Inicializar contadores
    totales = {
        "bullet": 0,
        "blitz": 0,
        "rapid": 0,
        "classical": 0
    }
    total_general = 0
    conteo_lineas_validas = 0  # Para verificar cuántas líneas con éxito procesamos

    # Verificar que el archivo exista
    try:
        with open(archivo, 'r', encoding='utf-8') as file:
            lineas = file.readlines()
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{archivo}' en el directorio actual.")
        return
    except Exception as e:
        print(f"❌ Ocurrió un error al intentar leer el archivo: {e}")
        return

    # Procesar cada línea
    for num_linea, linea in enumerate(lineas, start=1):
        linea = linea.strip()
        
        # Ignorar líneas vacías
        if not linea:
            continue

        # Solo procesar líneas con "Éxito"
        if "Estado: Éxito" not in linea:
            continue

        # Extraer tipo de partida
        partida_tipo = None
        if "Partida: bullet" in linea:
            partida_tipo = "bullet"
        elif "Partida: blitz" in linea:
            partida_tipo = "blitz"
        elif "Partida: rapid" in linea:
            partida_tipo = "rapid"
        elif "Partida: classical" in linea:
            partida_tipo = "classical"

        if not partida_tipo:
            print(f"⚠️  Advertencia (línea {num_linea}): No se identificó el tipo de partida.")
            continue

        # Extraer número después de "Descargadas:"
        if "Descargadas:" not in linea:
            print(f"⚠️  Advertencia (línea {num_linea}): Estado Éxito pero no se encontró 'Descargadas'")
            continue

        try:
            # Extraer todo lo que sigue a "Descargadas:"
            parte = linea.split("Descargadas:")[1].strip()
            # Tomar solo los dígitos al inicio (por si hay texto adicional)
            numero_str = ""
            for char in parte:
                if char.isdigit():
                    numero_str += char
                else:
                    break  # Detener al primer carácter no numérico
            if not numero_str:
                raise ValueError("No se encontró número")
            descargadas = int(numero_str)
        except Exception as e:
            print(f"❌ Error al extraer número en línea {num_linea}: '{linea}' -> {e}")
            continue

        # Acumular
        totales[partida_tipo] += descargadas
        total_general += descargadas
        conteo_lineas_validas += 1

        # Modo depuración: mostrar lo que se procesó
        if debug:
            print(f"✅ Línea {num_linea}: {partida_tipo} += {descargadas}")

    # Mostrar resultados
    print("\n" + "="*50)
    print("📊 RESULTADOS FINALES")
    print("="*50)
    for tipo, cantidad in totales.items():
        print(f"  {tipo.capitalize():<10}: {cantidad:>8} partidas")
    print("-" * 50)
    print(f"📌 Total general de partidas descargadas: {total_general}")
    print(f"📁 Líneas con éxito procesadas: {conteo_lineas_validas}")

    return totales, total_general


# --- Ejecución ---
if __name__ == "__main__":
    # Cambia a True si quieres ver detalles de cada línea procesada
    analizar_log(debug=False)