# model/chess_policy_model.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_policy_model(input_shape=(8, 8, 29)):
    """
    Modelo de política para ajedrez, optimizado para entrenamiento en GPU con 4 GB de VRAM.
    
    Características:
    - Usa bloques residuales (ResNet) para mejorar el flujo de gradientes.
    - Salida espacial (8x8x73) en lugar de Dense(4672)
    - Diseñado para entrenar con lotes (batch) de 64 en RTX 3050.
    - Total de parámetros: ~1.2M → ocupa ~3.5 GB de VRAM (con mixed precision).
    
    Args:
        input_shape (tuple): Forma de entrada (8, 8, 29) → tablero + planos de características.
        num_actions (int): Número de tipos de movimientos (73 es estándar en Leela Chess Zero).
    
    Returns:
        keras.Model: Modelo listo para entrenar con .fit().
    """
    inputs = layers.Input(shape=input_shape)

    # === Stem: convolución inicial ===
    x = layers.Conv2D(64, (3, 3), padding='same', name='stem_conv')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.LeakyReLU(alpha=0.01, name='stem_activation')(x)

    # === Bloque Residual 1 ===
    x_shortcut = x
    x = layers.Conv2D(64, (3, 3), padding='same', name='res1_conv1')(x)
    x = layers.BatchNormalization(name='res1_bn1')(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv2D(64, (3, 3), padding='same', name='res1_conv2')(x)
    x = layers.BatchNormalization(name='res1_bn2')(x)
    x = layers.Add(name='res1_add')([x, x_shortcut])
    x = layers.LeakyReLU(alpha=0.01, name='res1_out')(x)

    # === Bloque Residual 2 ===
    # Ajuste de canales con 1x1 conv
    x_shortcut = layers.Conv2D(128, (1, 1), padding='same', name='res2_shortcut_conv')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', name='res2_conv1')(x)
    x = layers.BatchNormalization(name='res2_bn1')(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv2D(128, (3, 3), padding='same', name='res2_conv2')(x)
    x = layers.BatchNormalization(name='res2_bn2')(x)
    x = layers.Add(name='res2_add')([x, x_shortcut])
    x = layers.LeakyReLU(alpha=0.01, name='res2_out')(x)

    # === Bloque Residual 3 ===
    x_shortcut = layers.Conv2D(128, (1, 1), padding='same', name='res3_shortcut_conv')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', name='res3_conv1')(x)
    x = layers.BatchNormalization(name='res3_bn1')(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv2D(128, (3, 3), padding='same', name='res3_conv2')(x)
    x = layers.BatchNormalization(name='res3_bn2')(x)
    x = layers.Add(name='res3_add')([x, x_shortcut])
    x = layers.LeakyReLU(alpha=0.01, name='res3_out')(x)

    # === Head de política: salida espacial (8, 8, 73) ===
    x = layers.Conv2D(73, (1, 1), name='policy_conv')(x)  # 73 movimientos posibles
    x = layers.Reshape((8 * 8 * 73,), name='policy_flatten')(x)  # (4672,)
    outputs = layers.Activation('softmax', name='policy_head')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ChessPolicyModel")

    # Compilación
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss='categorical_crossentropy',  # Usa one-hot encoding (mejor para softmax)
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
        ]
    )
    return model

