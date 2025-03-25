# mnist_classifier.py

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# CHARGEMENT DU DATASET
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# NORMALISATION DES DONNÉES
x_train = x_train / 255.0
x_test = x_test / 255.0

# CONSTRUCTION DU MODÈLE
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# COMPILATION
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ENTRAÎNEMENT
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# ÉVALUATION
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nPrécision sur les données de test : {test_acc:.4f}")

# PREDICTION SUR DES EXEMPLES
predictions = model.predict(x_test)

# AFFICHAGE DE QUELQUES RÉSULTATS
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], cmap="gray")
    plt.title(f"Vrai: {y_test[i]}\nPrévu: {np.argmax(predictions[i])}")
    plt.axis("off")
plt.tight_layout()
plt.show()
