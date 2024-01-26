import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from keras.preprocessing import image
from keras.src.utils import load_img, img_to_array
from matplotlib import pyplot

# Paramètre du jeu de données
taille_image = (32, 32)
# RVB
nombre_canaux = 3
# Classes
classes = ["OISEAU", "CHAT", "CHIEN"]

model = models.load_model("tf_models/model_cnn_cifar10-custom-v2.keras")
model.summary()


# load the image with the required shape
def charger_et_preparer_image(image_path, target_size=taille_image):
    # Cgargement de l'image
    img = image.load_img(image_path, target_size=target_size)
    # Transformation en NP Array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


# summarize filter shapes
for layer_id, conv_layer in enumerate(model.layers):
    # # check for convolutional layer
    layer_name = conv_layer.name
    if 'conv' not in layer_name:
        continue
    # get filter weights
    filters, biases = conv_layer.get_weights()
    print(layer_name, filters.shape)

    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    # 8 premiers filtres x 3 couches
    n_filters, ix = 8, 1
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]
        # plot each channel separately
        for j in range(3):
            # specify subplot and turn of axis
            ax = pyplot.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(f[:, :, j], cmap='gray')
            ix += 1
    # Visualisation
    pyplot.title(f"Filtres n°{layer_id}: {layer_name} (8 filtres x 3 couches)")
    pyplot.savefig(f"tf_models/maps--model_cnn_cifar10-{layer_name}.png")
    # show the figure
    pyplot.show()

    img = charger_et_preparer_image('dataset/2.testing/coq.png')

    # Model sur la 1ʳᵉ couche de Convolution
    model_convlayer_1 = keras.Model(inputs=model.inputs, outputs=model.layers[layer_id].output)

    # Prédiction
    feature_maps_convlayer_1 = model_convlayer_1.predict(img)

    # 224x224x32 feature_maps
    dimensions = [4, 8]
    ix = 1
    for _ in range(dimensions[0]):
        for _ in range(dimensions[1]):
            # specify subplot and turn of axis
            ax = pyplot.subplot(dimensions[0], dimensions[1], ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(feature_maps_convlayer_1[0, :, :, ix - 1], cmap='gray')
            ix += 1
    # show the figure
    pyplot.title(f"Feature Map n°{layer_id}: {layer_name} (32)")
    pyplot.savefig(f"tf_models/feature_maps--model_cnn_cifar10-{layer_name}.png")
    pyplot.show()
