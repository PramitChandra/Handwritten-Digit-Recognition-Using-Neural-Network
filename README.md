# Handwritten Digit Recognition Using Neural Network

This project demonstrates a neural network-based handwritten digit recognition system using the MNIST dataset. The code provided loads the dataset, trains a neural network model, and allows users to test the model on custom handwritten digit images.

## Project Overview

The key components of this project are as follows:

- **Dataset**: The MNIST dataset, containing handwritten digit images, is loaded using TensorFlow's built-in dataset.

- **Data Preprocessing**: The input data is scaled down to ensure that pixel values are between 0 and 1, enhancing model performance.

- **Neural Network Model**: A sequential neural network model is constructed using TensorFlow's Keras API. The model architecture includes flattening the input, adding dense layers with ReLU activation functions, and a final softmax output layer for classification.

- **Model Training**: The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss. It is trained using the training data for a specified number of epochs.

- **Model Evaluation**: The trained model is evaluated using the testing dataset to calculate loss and accuracy.

- **Handwritten Digit Recognition**: Custom handwritten digit images (stored as PNG files) can be provided to the trained model, and the model predicts the recognized digit. The prediction results are displayed, and the image is shown using matplotlib.

## Usage

1. Ensure you have the necessary Python packages installed, including TensorFlow, OpenCV (cv2), NumPy, and Matplotlib.

2. Run the provided Python code to train the model and save it.

3. To use the trained model for recognizing handwritten digits from custom images, place the PNG image files of digits in the specified folder (E:\VS Folder\Python Projects\Digits_Samples).

4. Modify the code to specify the correct folder path where the custom digit images are located.

5. Run the modified code to predict the recognized digit for each custom image.

## Dependencies

- TensorFlow
- OpenCV (cv2)
- NumPy
- Matplotlib

## Sample Images

- Sample digit images can be placed in the directory specified in the code (adjust the path as needed).

## Acknowledgments

The model is trained on the MNIST dataset, which is widely used for handwritten digit recognition tasks in machine learning.

## License

This project is licensed under the [MIT License](LICENSE).
