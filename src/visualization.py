import matplotlib.pyplot as plt
import random
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix
import os
import cv2

def visualize_image(data, label=None):
    """
    The function selects a random image from the given dataset and visualizes it. 
    If a label is given, it selects a random image that matches that label.
    
    Args:
        data (ImageDataGenerator): Training or validation dataset.
        label (str): Label of the image to be visualized ("Health" or "Tumor"). 
                    If not entered, a random image is displayed.
    """
    
    target_label = 1 if label == "Health" else (0 if label == "Tumor" else None)
    
   
    images, labels = [], []
    for batch_images, batch_labels in data:
        for i in range(len(batch_labels)):
            if target_label is None or batch_labels[i] == target_label:
                images.append(batch_images[i])
                labels.append(batch_labels[i])
        
        if len(images) > 0:
            break
    
    
    idx = random.randint(0, len(images) - 1)
    
    
    img = images[idx]  

    selected_label = "Health" if labels[idx] == 1 else "Tumor"
    
    plt.imshow(img)
    plt.title(f"Label: {selected_label}")
    plt.axis("off")
    plt.show()

def visualize_images_by_label(data,num_images_per_label=5):

    """
    Visualizes random images with specified number of 
    'Health' and 'Tumor' labels from given dataset.
    
    Args:
        data (ImageDataGenerator): Training or validation dataset.
        num_images_per_label (int, optional): Number of images to display from each tag.
                                              By default it is taken as 5.
    """

    health_images, tumor_images = [], []

    for batch_images, batch_labels in data:
        for i in range(len(batch_labels)):
            if batch_labels[i] == 1 and len(health_images) < num_images_per_label:
                health_images.append(batch_images[i])
            elif batch_labels[i] == 0 and len(tumor_images) < num_images_per_label:
                tumor_images.append(batch_images[i])
        
        if len(health_images) == num_images_per_label and len(tumor_images) == num_images_per_label:
            break
    
    plt.figure(figsize=(num_images_per_label * 4, 12))

    for i, img in enumerate(health_images):
        plt.subplot(2, num_images_per_label, i + 1)
        plt.imshow(img)
        plt.title(f"Health {i+1}", fontsize=32)
        plt.axis("off")
    
    for i, img in enumerate(tumor_images):
        plt.subplot(2, num_images_per_label, num_images_per_label + i + 1)
        plt.imshow(img)
        plt.title(f"Tumor {i+1}", fontsize=32) 
        plt.axis("off")

    plt.suptitle('Healthy and Tumor Brain Images', fontsize=35)
    plt.tight_layout()
    plt.show()

def visualize_class_distribution(data):
    """
    Plots the distribution of classes in the training dataset.
    
    Args:
        data (ImageDataGenerator): Training or validation dataset.
    """

    labels = []
    if hasattr(data, 'subset'):
        graph_label = "Train" if data.subset == 'training' else "Validation"
    else:
        graph_label = "Train"

    for i in range(len(data)):
        images, batch_labels = next(data)
        labels.extend(batch_labels)  

    labels = np.array(labels)

    healthy_count = np.sum(labels == 1)
    tumor_count = np.sum(labels == 0)

    categories = ['Healthy', 'Tumor']
    counts = [healthy_count, tumor_count]

    plt.bar(categories, counts, color=['green', 'red'])
    plt.title(f'Distribution of Healthy and Tumor Images in {graph_label} Dataset')
    plt.ylabel('Number of Images')
    plt.show()


def plot_training_history(history):
    """
    Plots accuracy and loss graphs based on the model's training history.
    
    Args:
        history: The History object returned from the model's fit() method.
    """
    plt.figure(figsize=(14, 5))
    
    try:
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
    except:
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
    plt.show()

    
def display_confusionMatrix_and_report(model, validation_generator, step_size_valid, n_classes, class_names,model_name="Model"):
    """
    Generates a classification report and confusion matrix using a validation generator.
    
    Args:
        model (keras.Model): The trained model to evaluate.
        validation_generator (keras.utils.Sequence): The validation data generator.
        step_size_valid (int): Number of steps to iterate over the validation generator.
        n_classes (int): Number of classes in the classification problem.
        class_names (list): List of class names.
        model_name (str): Name of the model to display in the confusion matrix title.
    
    Returns:
        None. Displays a confusion matrix and prints the classification report.
    """
    y_pred, y_true = [], []
    
    for i in range(step_size_valid):
        X_batch, y_batch = next(validation_generator)  
        if X_batch.shape[0] != validation_generator.batch_size:
            continue
        y_pred_batch = model.predict(X_batch, verbose=0)  
        y_pred.append(y_pred_batch)
        y_true.append(y_batch)
    
    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)
    
    if n_classes > 2:
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
    else:
        y_pred_classes = (y_pred > 0.5).astype(int).flatten()
        y_true_classes = y_true.flatten().astype(int)
    
    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names,annot_kws={"fontsize": 16})
    plt.title(f"Confusion Matrix of {model_name}", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(fontsize=12) 
    plt.yticks(fontsize=12)
    plt.show()
    
    print(f"Classification Report of {model_name}:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

def display_image_properties(data_dir):
    """
    Displays image properties (height, width, and color channels) for images in the 'Brain Tumor' and 'Healthy' categories,
    including statistical summaries and histograms of image dimensions.

    Args:
        data_dir (str): Path to the directory containing subdirectories 'Brain Tumor' and 'Healthy',
                         each containing image files for analysis.

    Returns:
        None: This function prints the summary of image properties and displays histograms of image dimensions.
    """
    categories = ["Brain Tumor", "Healthy"]  
    valid_extensions = ('.jpeg', '.jpg', '.png', '.tiff')  
    
    for category in categories:
        category_path = os.path.join(data_dir, category)
        heights = []
        widths = []
        channels = []
        
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            if file_path.lower().endswith(valid_extensions): 
                img = cv2.imread(file_path)
                if img is not None:
                    h, w, c = img.shape
                    heights.append(h)
                    widths.append(w)
                    channels.append(c)

        print(f"\nCategory: {category}")
        print("Total Images Analyzed:", len(heights))
        print("Average Height:", np.mean(heights) if heights else 0)
        print("Average Width:", np.mean(widths) if widths else 0)
        print("Color Channels (3=RGB, 1=Grayscale):", set(channels))
        
        if heights and widths:
            plt.figure(figsize=(10, 5))
            plt.hist(heights, bins=20, alpha=0.7, label='Height')
            plt.hist(widths, bins=20, alpha=0.7, label='Width')
            plt.title(f'Distribution of Image Dimensions in {category}')
            plt.xlabel('Pixels')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()

def display_pixel_intensity(data_dir):
    """
    Displays pixel intensity histograms for images in the 'Brain Tumor' and 'Healthy' categories.
    The function reads images, computes their pixel intensity distributions, and visualizes them using histograms.

    Args:
        data_dir (str): Path to the directory containing subdirectories 'Brain Tumor' and 'Healthy',
                         each containing images for analysis.

    Returns:
        None: This function generates and displays pixel intensity histograms for each category.
    """
    categories = ["Brain Tumor", "Healthy"]
    valid_extensions = ('.jpeg', '.jpg', '.png', '.tiff')

    for category in categories:
        category_path = os.path.join(data_dir, category)
        plt.figure(figsize=(10, 5))
        
        for i, file in enumerate(os.listdir(category_path)):
            file_path = os.path.join(category_path, file)
            if file_path.lower().endswith(valid_extensions):
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  
                if img is not None:
                    plt.hist(img.ravel(), bins=256, range=(0, 256), alpha=0.5, label=file if i == 0 else None)
                    
            if i == 200:  
                break

        plt.title(f'Pixel Intensity Histogram for {category}')
        plt.xlabel('Pixel Intensity (0-255)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

