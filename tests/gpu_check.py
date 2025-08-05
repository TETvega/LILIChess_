# test_gpu.py

# Generado por Gemini
"""
    Codigo para verificar la configuración de TensorFlow y GPU.
    Este script comprueba la versión de TensorFlow, la disponibilidad de GPU,
    y configura el crecimiento de memoria para las GPUs disponibles.
    
"""
import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("✅ Versión de TensorFlow:", tf.__version__)
print("✅ GPU Disponible:", tf.config.list_physical_devices('GPU'))

# Habilita crecimiento de memoria
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Crecimiento de memoria de GPU habilitado.")
    except RuntimeError as e:
        print("❌ Error:", e)
else:
    print("❌ GPU no detectada. Aún usando CPU.")

# Prueba en GPU
tf.debugging.set_log_device_placement(True)
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
c = tf.matmul(a, b)
print("Resultado:\n", c)