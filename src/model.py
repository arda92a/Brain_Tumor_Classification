import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNetV2, DenseNet121
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall

def create_basic_cnn1(input_shape=[150,150,3]):    

    cnn = tf.keras.models.Sequential()

    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=input_shape))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")) 
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")) 
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

    cnn.add(tf.keras.layers.Flatten())

    cnn.add(tf.keras.layers.Dense(units=128,activation="relu"))

    cnn.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

    cnn.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

    return cnn

def create_basic_cnn2(input_shape=[150,150,3]):    

    cnn = tf.keras.models.Sequential()

    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=input_shape))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")) 
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")) 
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

    cnn.add(tf.keras.layers.Flatten())

    cnn.add(tf.keras.layers.Dense(units=128,activation="relu"))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Dropout(0.3))
    cnn.add(tf.keras.layers.Dense(units=64,activation="relu"))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

    cnn.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

    return cnn


def create_resnet_model(input_shape):
    """
    Creates a ResNet-based CNN model for image classification.
    
    Args:
        input_shape (tuple): Shape of the input images (e.g., (150, 150, 3)).
    
    Returns:
        Model: A compiled ResNet-based model.
    """

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = True
    
    X = base_model.output
    X = Flatten()(X)

    X = Dense(512, kernel_initializer='he_uniform')(X)
    X = Dropout(0.4)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Dense(128, kernel_initializer='he_uniform')(X)
    X = Dropout(0.4)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Dense(16, kernel_initializer='he_uniform')(X)
    X = Dropout(0.4)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    output = Dense(1, activation='sigmoid')(X)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def create_mobilenetv2_model(input_shape):
    """
    Creates a MobileNetV2-based CNN model for image classification.

    Args:
        input_shape (tuple): Shape of the input images (e.g., (150, 150, 3)).

    Returns:
        Model: A compiled MobileNetV2-based model.
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer="adam", loss = 'binary_crossentropy', metrics=['accuracy'])

    return model

def create_densenet_model(input_shape):
    """
    Creates a DenseNet121-based CNN model for image classification.

    Args:
        input_shape (tuple): Shape of the input images (e.g., (224, 224, 3)).

    Returns:
        Model: A compiled DenseNet121-based model.
    """
    basemodel = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape, pooling=None)

    x = Flatten()(basemodel.output)
    x = Dropout(0.7)(x)
    x = BatchNormalization()(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    output = Dense(1, activation= 'sigmoid')(x)

    model = Model(inputs=basemodel.input, outputs=output)

    loss = 'binary_crossentropy'
    model.compile(
        loss=loss,
        optimizer="adam",
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
    )

    return model