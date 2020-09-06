import numpy as np
import pandas as pd
import tensorflow as tf

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=786, activation='sidmoid'))
ann.add(tf.keras.layers.Dense(units=100, activation='sigmoid'))
ann.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
