import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.feature_selection import chi2
import scipy.io as sio

# Load ResNet50 model as AlexNet replacement
net = ResNet50(weights='imagenet', include_top=True)
layer_name = net.layers[-3].name  # Get the third last layer's name (for feature extraction)
input_size = net.layers[0].input_shape[1:3]

# Initialize lists to store features and labels
X = []
y = []
indices = []

# Iterate through all image files in the directory
for filename in os.listdir('.'):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Read and preprocess the image
        img = image.load_img(filename, target_size=input_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Extract features using the specified layer
        model = tf.keras.Model(inputs=net.input, outputs=net.get_layer(layer_name).output)
        features = model.predict(img_array)

        # Append the features and labels to the lists
        X.append(features.flatten())
        y.append(int(filename[3:5]))  # Extract label from the filename
        indices.append(int(filename[0:2]))  # Extract index from the filename

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)
indices = np.array(indices)

# Normalize the features
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + np.finfo(float).eps)

# Feature selection using Chi-square test
chi2_scores, p_values = chi2(X, y)
idx = np.argsort(chi2_scores)[::-1][:256]  # Get indices of top 256 features

# Select top 256 features
X_selected = X[:, idx]

# Combine features and labels
son = np.hstack((X_selected, y.reshape(-1, 1)))

print("Processing complete and data saved.")
