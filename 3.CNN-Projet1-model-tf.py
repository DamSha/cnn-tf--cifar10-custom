import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from matplotlib import pyplot

# Paramètre du jeu de données
taille_image = (32, 32)
# RVB
nombre_canaux = 3
# kernel_size
kernel_size = (3, 3)
# Classes
classes = ["OISEAU", "CHAT", "CHIEN"]

# v2 : Augmentations des images
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal", seed=42),
    layers.RandomRotation(0.2, seed=42),
    layers.RandomContrast(0.2, seed=42)
])

model = models.Sequential([
    layers.InputLayer(input_shape=(*taille_image, nombre_canaux)),

    # data_augmentation,

    layers.Conv2D(16, kernel_size, padding="same", activation="relu"),
    layers.Conv2D(16, kernel_size, activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(.25),
    layers.Conv2D(32, kernel_size, padding="same", activation="relu"),
    layers.Conv2D(32, kernel_size, activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(.25),
    layers.Conv2D(64, kernel_size, padding="same", activation="relu"),
    layers.Conv2D(64, kernel_size, activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(.25),
    layers.Conv2D(128, kernel_size, padding="same", activation="relu"),
    layers.Conv2D(128, kernel_size, activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(.25),
    layers.Conv2D(256, kernel_size, padding="same", activation="relu"),
    layers.Conv2D(256, kernel_size, activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(.25),

    # Vectorisation
    layers.Flatten(),

    # Neurones profonds
    layers.Dense(1024, activation="relu"),
    # layers.Dense(1024, activation="relu"),
    layers.Dropout(.4),
    # layers.Dense(128, activation="relu"),

    # Neurones de sortie
    layers.Dense(len(classes), activation="softmax")
])

# Modele
# Adam : learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, ema_momentum=0.99
model.compile(
    # optimizer=keras.optimizers.SGD(),
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.CategoricalCrossentropy(),
    # loss="categorical_crossentropy",
    metrics=[keras.metrics.CategoricalAccuracy(),
             # keras.metrics.F1Score(),
             # keras.metrics.FalsePositives(),
             # keras.metrics.FalseNegatives(),
             ]
    # metrics=["accuracy"]
)

model.summary()

# Generation du Dataset
# https://stackoverflow.com/questions/62547807/how-to-create-cifar-10-subset
training_path = "dataset/1.training"

train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=.2)

train_generator = train_datagen.flow_from_directory(
    training_path,
    target_size=taille_image,
    batch_size=32,
    class_mode="categorical",
    subset="training",
)

validation_generator = train_datagen.flow_from_directory(
    training_path,
    target_size=taille_image,
    batch_size=32,
    class_mode="categorical",
    subset="validation",
)

# Early Stopper Callback
# Au bout de "patience" fois où la valeur de "monitor" ne change plus trop
model_train_callback = EarlyStopping(restore_best_weights=True, patience=2, verbose=True, start_from_epoch=10)

# Entrainement du modele
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // 32),
    epochs=20,
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator.samples // 32),
    callbacks=[model_train_callback]
)

# Sauvegarde du modèle
model.save("tf_models/model_cnn_cifar10-custom-v2.keras")

# plot training history
# plot loss
pyplot.subplot(211)
pyplot.title('Cross Entropy Loss')
pyplot.plot(history.history['loss'], color='blue', label='train')
pyplot.plot(history.history['val_loss'], color='orange', label='test')
# plot accuracy
pyplot.subplot(212)
pyplot.title('Classification Accuracy')
pyplot.plot(history.history['categorical_accuracy'], color='blue', label='train')
pyplot.plot(history.history['val_categorical_accuracy'], color='orange', label='test')

pyplot.savefig("tf_models/history--model_cnn_cifar10-custom-v2.png")
pyplot.show()

# pandas.DataFrame(history.history).plot()

print("Model Saved! Bye.")
