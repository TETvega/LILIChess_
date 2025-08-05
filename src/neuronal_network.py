# src/neural_network.py

import tensorflow as tf
from tensorflow.keras import layers, models, Model

def create_residual_block(x, filters):
    """
    Un bloque residual: x -> Conv -> BN -> ReLU -> Conv -> BN -> +x -> ReLU
    x es la entrada, filters es el número de filtros para las convoluciones.
    Conv es una convolución 2D 
    BN es Batch Normalization
    ReLU es la activación.
    """
    shortcut = x
    # Convolución 1
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # Convolución 2
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Atajo: si la entrada y salida no coinciden, ajusta la forma
    if shortcut.shape[-1] != filters:
        # Ajusta la forma del atajo para que coincida con la salida
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)
    # Suma residual
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def create_chess_network(input_shape=(8, 8, 22), num_policies=None, num_residual_blocks=5, filters=64):
    """
    Red tipo AlphaZero:
    - Input: (8, 8, 22)
    - 5 bloques residuales
    - Salida: política (movimiento) + valor (ganar/perder)
    """
    inputs = layers.Input(shape=input_shape)

    # Capa inicial
    x = layers.Conv2D(filters, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Bloques residuales
    for _ in range(num_residual_blocks):
        x = create_residual_block(x, filters)

    # --- Salida de POLÍTICA (movimiento) ---
    pol = layers.Conv2D(2, (1, 1), padding='same')(x)
    pol = layers.BatchNormalization()(pol)
    pol = layers.ReLU()(pol)
    pol = layers.Flatten()(pol)
    pol = layers.Dense(num_policies, name='policy', activation='softmax')(pol)

    # --- Salida de VALOR ---
    val = layers.Conv2D(1, (1, 1), padding='same')(x) # Conv para reducir a 1 canal
    val = layers.BatchNormalization()(val) 
    val = layers.ReLU()(val) 
    val = layers.Flatten()(val  )
    val = layers.Dense(256, activation='relu')(val)
    val = layers.Dense(1, name='value', activation='tanh')(val)  # -1 (pierde) a +1 (gana) y si empate es 0

    # Modelo
    model = Model(inputs=inputs, outputs=[pol, val])

    # Compilación del modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
        'policy': 'categorical_crossentropy',
        'value': 'mean_squared_error' 
        },
        loss_weights={
            'policy': 1.0, # Ponderación de la pérdida de política 
            'value': 1.0 # Ponderación de la pérdida de valor
        },
        metrics={
        'policy': 'accuracy',        # accuracy para política
        'value': 'mae'               #  mae para valor (mean absolute error)
    }
    )

    return model


