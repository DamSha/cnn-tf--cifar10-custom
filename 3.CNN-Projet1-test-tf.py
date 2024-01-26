import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from keras.preprocessing import image

# Paramètre du jeu de données
taille_image = (32, 32)
# RVB
nombre_canaux = 3
# Classes
classes = ["OISEAU", "CHAT", "CHIEN"]

model = models.load_model("tf_models/model_cnn_cifar10-custom-v2.keras")
model.summary()

# Test prediction manuel
def charger_et_preparer_image(image_path):
    # Cgargement de l'image
    img = image.load_img(image_path, target_size=taille_image)
    # Transformation en NP Array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


images_test = [
    "dataset/2.testing/bird/0001.png",
    "dataset/2.testing/bird/0002.png",
    "dataset/2.testing/cat/0001.png",
    "dataset/2.testing/cat/0002.png",
    "dataset/2.testing/dog/0001.png",
    "dataset/2.testing/dog/0002.png",
]

for image_test_path in images_test:
    print("-- -- --")
    print(f"TEST AVEC IMAGE : {image_test_path}")
    image_test = charger_et_preparer_image(image_test_path)

    predictions = model.predict(image_test)
    print(f"Predictions : {np.round(predictions * 100.0, 2)}")

    classe_predite = classes[np.argmax(predictions)]
    precision = np.max(predictions)
    print(f"Résultat : {classe_predite} avec {precision * 100:.2f}% de probabilité.")

    print("-- -- --")

# Test prediction AUTO

# Dataset de test
testing_path = "dataset/2.testing"
test_datagen = image.ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    testing_path,
    target_size=taille_image,
    batch_size=32,
    class_mode="categorical"
)

# evaluate model
_, acc = model.evaluate(
    test_generator,
    steps=max(1, test_generator.samples // 32),
    verbose=1
)
print('> %.3f' % (acc * 100.0))
