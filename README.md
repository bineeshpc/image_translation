# MRI Image Translation using CycleGAN

## Problem Statement

Misdiagnosis in the medical field is a very serious issue but it’s also uncomfortably common to occur. Imaging procedures in the medical field require an expert radiologist’s opinion since interpreting them is not a simple binary process (Normal or Abnormal). Even so, one radiologist may see something that another does not. This can lead to conflicting reports and make it difficult to effectively recommend treatment options to the patient.

One of the complicated tasks in medical imaging is to diagnose MRI (Magnetic Resonance Imaging). Sometimes to interpret the scan, the radiologist needs different variations of the imaging which can drastically enhance the accuracy of diagnosis by providing practitioners with a more comprehensive understanding.

However, having access to different imaging is difficult and expensive. With the help of deep learning, we can use style transfer to generate artificial MRI images of different contrast levels from existing MRI scans. This will help to provide a better diagnosis with the help of an additional image.

In this project, you will use CycleGAN to translate the style of one MRI image to another, which will help in a better understanding of the scanned image. Using GANs, you will create T2 weighted images from T1 weighted MRI images and vice-versa.

## Project Structure

### Requirements

Python 3.x
TensorFlow
NumPy
Matplotlib
scikit-image
imageio
glob
keras

### Installation

To install the necessary libraries, you can use the following commands:

### Usage

Data Loading: Load the MRI images from the specified dataset path.
Data Preprocessing: Normalize and resize the images to the required dimensions.
Model Building: Build the CycleGAN model using a modified U-Net architecture.
Training: Train the model for a specified number of epochs.
Evaluation: Generate and visualize the translated MRI images.

### Running the Project

To run the project, execute the image_translation.py script:

### Checkpoints

The model checkpoints will be saved in the ./Trained_Model directory. If a checkpoint exists, the latest checkpoint will be restored.

### Results

The generated images will be saved as PNG files and a GIF will be created to visualize the results.

### License

This project is licensed under the MIT License.

### Acknowledgements

This project uses the CycleGAN architecture for image translation.
Special thanks to the contributors of the libraries used in this project.
For more details, refer to the image_translation.py file.
