# -*- coding: utf-8 -*-
"""
ROSE: Robust Optical Segmentation and Evaluation
Automatically generated and cleaned script for plant image segmentation,
feature extraction, and classification using ML and DL models.
"""

import os
import cv2
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dense
from skimage.feature import hog, local_binary_pattern, greycomatrix, greycoprops

# --- Step 1: Dataset Download ---
def download_dataset():
    folder_id = "your_folder_id_here"  # Replace with actual if used
    subprocess.run(["gdown", "--id", folder_id, "--folder"])
    print("Dataset downloaded.")

# --- Step 2: Data Cleaning ---
def clean_data(dataset_path):
    for root, _, files in os.walk(dataset_path):
        for file in files:
            path = os.path.join(root, file)
            try:
                if cv2.imread(path) is None:
                    os.remove(path)
            except:
                os.remove(path)

# --- Step 3: Load and Resize ---
def load_and_resize_images(path, size=(256, 256)):
    imgs, labels = [], []
    for cls in os.listdir(path):
        cls_path = os.path.join(path, cls)
        if os.path.isdir(cls_path):
            for file in os.listdir(cls_path):
                if file.lower().endswith(('jpg', 'jpeg', 'png')):
                    img = cv2.imread(os.path.join(cls_path, file))
                    if img is not None:
                        imgs.append(cv2.resize(img, size))
                        labels.append(cls)
    return imgs, labels

# --- Step 4: Feature Extraction ---
def extract_hog(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return hog(gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), transform_sqrt=True)

def extract_hist(img, space='rgb'):
    if space == 'hsv': img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif space == 'lab': img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([img], [0,1,2], None, [8,8,8], [0,256]*3)
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp, bins=256, range=(0, 255))
    return hist

def extract_glcm(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = greycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    return np.array([
        greycoprops(glcm, p)[0,0] for p in ['contrast','dissimilarity','homogeneity','energy','correlation']
    ])

def extract_dl_features(images, model_name='vgg16'):
    base = {
        'vgg16': (VGG16, preprocess_vgg),
        'resnet50': (ResNet50, preprocess_resnet),
        'inceptionv3': (InceptionV3, preprocess_inception)
    }[model_name]
    base_model = base[0](weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))
    X = np.array([base[1](cv2.resize(img, (224,224))) for img in images])
    return model.predict(X, batch_size=32)

# --- Step 5: CNN Build ---
def build_cnn(input_shape, num_classes):
    model = tf.keras.Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape), MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'), MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu'), MaxPooling2D((2,2)),
        Flatten(), Dense(128, activation='relu'), Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Main Execution ---
if __name__ == "__main__":
    path = "Groundnut Leaf Spot"
    if not os.path.exists(path):
        download_dataset()
    clean_data(path)
    images, labels = load_and_resize_images(path)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    X_train_idx, X_test_idx = train_test_split(np.arange(len(images)), test_size=0.2, random_state=42)

    features = {
        'hog': np.array([extract_hog(images[i]) for i in range(len(images))]),
        'rgb_hist': np.array([extract_hist(images[i], 'rgb') for i in range(len(images))]),
        'hsv_hist': np.array([extract_hist(images[i], 'hsv') for i in range(len(images))]),
        'lbp': np.array([extract_lbp(images[i]) for i in range(len(images))]),
        'vgg16': extract_dl_features(images, 'vgg16')
    }

    models = {
        'SVM': SVC(kernel='linear'),
        'RF': RandomForestClassifier(n_estimators=100),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'AdaBoost': AdaBoostClassifier(n_estimators=50),
        'GB': GradientBoostingClassifier(n_estimators=100),
        'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
    }

    for name, model in models.items():
        feat_key = 'hog' if name in ['SVM', 'AdaBoost'] else 'rgb_hist' if name == 'RF' else 'lbp' if name == 'KNN' else 'hsv_hist' if name == 'GB' else 'vgg16'
        X_train = features[feat_key][X_train_idx]
        X_test = features[feat_key][X_test_idx]
        model.fit(X_train, y[X_train_idx])
        y_pred = model.predict(X_test)
        print(f"\n{name} on {feat_key.upper()}:")
        print(classification_report(y[X_test_idx], y_pred, target_names=le.classes_))

    print("\nTraining CNN model...")
    cnn = build_cnn((224,224,3), len(le.classes_))
    cnn.fit(
        np.array([cv2.resize(images[i], (224,224))/255. for i in X_train_idx]),
        y[X_train_idx],
        validation_data=(
            np.array([cv2.resize(images[i], (224,224))/255. for i in X_test_idx]),
            y[X_test_idx]
        ),
        epochs=10, batch_size=32
    )
    y_pred_cnn = np.argmax(cnn.predict(np.array([cv2.resize(images[i], (224,224))/255. for i in X_test_idx])), axis=1)
    print("\nCNN Report:")
    print(classification_report(y[X_test_idx], y_pred_cnn, target_names=le.classes_))
