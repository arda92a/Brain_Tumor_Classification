import tensorflow as tf
import os
import cv2 
import random

def load_data(data_dir, img_size=(150, 150), batch_size=32):
    """
    Loads brain tumor data from the direction and prepares training/validation data using ImageDataGenerator.
    
    Args:
        data_dir (str): The directory where the images are located.
        img_size (tuple): Image size to resize (default 150x150).
        batch_size (int): Number of images to use per run (default 32).
    
    Returns:
        tuple: Training and validation datasets.
    """

    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,  
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
        )
    
    if (img_size is None) : 

        train_data = datagen.flow_from_directory(
            data_dir,
            batch_size=batch_size,
            class_mode="binary",
            subset='training'
            )

        validation_data = datagen.flow_from_directory(
            data_dir,
            batch_size=batch_size,
            class_mode="binary",
            subset='validation'  
            )
    else :
          
        train_data = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset='training'
        )

        validation_data = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset='validation'  
        )  
          
    return train_data, validation_data


def get_random_image(data_dir):
    
    """
    Selects a random image from the specified directory and applies image processing.
    
    Args:
        data_dir (str): Directory path containing images.
        
    Returns:
        numpy.ndarray: Original image
    """

    image_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        raise ValueError("No images found in the specified directory.")

    random_image_path = random.choice(image_paths)

    image = cv2.imread(random_image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image