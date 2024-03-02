
# Audio Classification with TensorFlow

As part of my placement at Arbury Logic Ltd, I have worked on a project that demonstrates a workflow for audio classification using deep learning techniques. The project primarily focuses on classifying audio clips into two sound categories - `"dog_bark"` and `"siren"`, using data from the UrbanSound8K dataset. This project is a precursor to a broader machine learning initiative within the company.

Moreover, the model developed in this project will be converted to TensorFlow Lite using both full-integer and quantization floating-point representations. This conversion will ensure that the model can be optimised for deployment on mobile and embedded devices. Our goal is to provide not only a functional classification system but also a streamlined process for integrating it into real-world applications. The project provides code for:

-	**Data Preprocessing:** Extracting Mel spectrograms from audio files and splitting data into training and testing sets.

-	**Model Building:** Creating a CNN architecture optimized for audio classification.

-	**Training and Evaluation:** Training the model with the UrbanSound8K dataset and evaluating its performance.

-	**Model Conversion:** Converting the trained model to TensorFlow Lite (TFLite) for mobile and embedded devices, both in full-integer quantization and floating-point representation.

-	**Testing TFLite Model:** Evaluating the accuracy of the converted TFLite models on the test set.



## Project Structure

The project contains the following folders:

-	`data`: Contains the dataset for the `"dog_bark"` and `"siren"` classes.

-	`models`: Stores the saved Keras and TFLite models.

-	`notebooks`: Contains Jupyter notebooks for various stages of the project.

## Notebooks

`audio_classification.ipynb`: This notebook showcases the audio classification workflow using TensorFlow. It covers data preprocessing, model building, training, and evaluation. The model architecture is a convolutional neural network (CNN) configured for audio analysis using Mel spectrogram representations of the audio data.

`dataset.ipynb`: This notebook demonstrates the creation of specialized datasets for the `"dog_bark"` and `"siren"` classes from the UrbanSound8K dataset. It filters and copies audio files into separate directories to create the required datasets.

`convert_to_TFLM.ipynb`: This notebook converts a pretrained Keras model to a TensorFlow Lite model for optimized deployment on mobile and embedded devices. It demonstrates the process of quantization to reduce model size while retaining accuracy.

`Convert_to_quantized_TFLM.ipynb`: This notebook converts the Keras model to a quantized TensorFlow Lite model for further optimization in terms of size and latency on mobile devices.

`test.ipynb`: This notebook tests the accuracy of the TensorFlow Lite model for audio classification. It loads the model, performs inference on a test set, and calculates the accuracy compared to ground truth labels.



## Usage

1.	**Data Preparation:** Ensure that the audio dataset is organized in the data folder with subfolders for each class.

2.	**Training the Model:** Execute the `audio_classification.ipynb` notebook to preprocess the data, build, train, and evaluate the model.

3.	**Model Conversion:** Use the `convert_to_TFLM.ipynb` or `Convert_to_quantized.ipynb` notebooks to convert the trained model to TensorFlow Lite format for deployment.

4.	**Testing the Model:** Finally, use the `test.ipynb` notebook to test the accuracy of the TensorFlow Lite model on a test set.

## Dependencies

The project requires the following dependencies:

-	TensorFlow
-	Librosa
-	NumPy
-	Pandas
-	scikit-learn

## Key Highlights

-	Utilizes Mel spectrograms as the audio representation for effective classification.

-	Demonstrates model training and evaluation with accuracy metrics.

-	Explores model conversion to TFLite for deployment on mobile and embedded devices.

-	Compares model sizes and accuracy between original and quantized TFLite versions.
