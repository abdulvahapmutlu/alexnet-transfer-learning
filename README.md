# Alexnet Transfer Learning for Feature Extraction and Selection

 This repository contains code for extracting features from images using a pre-trained AlexNet model through transfer learning. It then performs feature selection using the Chi-square test to identify the most relevant features. This can be utilized for various machine learning tasks such as image classification and clustering.

### Features

- *Transfer Learning with AlexNet*: Utilizes the AlexNet model pre-trained on ImageNet for feature extraction.
- *Feature Extraction*: Extracts deep learning features from the penultimate layer of the AlexNet model.
- *Feature Selection*: Uses the Chi-square test to select the top 256 features from the extracted feature set.
- *Data Normalization*: Normalizes the extracted features to ensure they are on a consistent scale.

### Code Overview

1. *Loading the Model*: Loads the AlexNet model with pre-trained ImageNet weights.
2. *Preprocessing*: Reads and preprocesses images to fit the input requirements of AlexNet.
3. *Feature Extraction*: Extracts features from the penultimate layer of AlexNet.
4. *Normalization*: Normalizes the features to a common scale.
5. *Feature Selection*: Selects the top 256 features using the Chi-square test.
6. *Saving Results*: Saves the selected features and labels for further analysis.

### How to Use

1. *Clone the repository*:
   sh
   git clone https://github.com/abdulvahapmutlu/alexnet-transfer-learning.git
   cd alexnet-transfer-learning
   

2. *Prepare your image dataset*: Place your image files in the repository directory. The filenames should follow the format where labels and indices can be extracted as shown in the code.

3. *Run the script*:
   sh
   python alexnet_1.py
   

### Dependencies

- TensorFlow
- NumPy
- SciPy
- Scikit-learn
- OpenCV

Install the required packages using pip:
sh
pip install tensorflow numpy scipy scikit-learn opencv-python

