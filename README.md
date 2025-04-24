# Image Classification with Transfer Learning and Segmentation

This project aims to use **transfer learning** to classify images into 4 distinct categories. The approach is divided into two parts: in the first part, we create a deep learning model from scratch, and in the second part, we use a pre-trained model and adapt it to our dataset. Additionally, in the second part, an **image segmentation** process is incorporated to classify small segments of an image, allowing us to identify multiple landscapes within a single image.

## Objectives

- **Transfer Learning**: Use a pre-trained model and fine-tune it for our dataset with 4 classes.
- **Segmentation**: Divide an image into small patches and predict the class of each using the trained model.
- **Improve Results**: Achieve higher accuracy by leveraging transfer learning and image segmentation.

## Dataset

The dataset used in this project was sourced from Kaggle and contains images categorized into four classes. These classes represent different types of landscapes (e.g., desert, water, green area, clouds, etc.). The images are preprocessed and split into training and testing sets using a Python script. This ensures that the model does not have access to the testing data during the training process.

## Project Structure

1. **Creating the Model from Scratch**: Build a deep learning model from scratch and train it for image classification.
2. **Transfer Learning**: Load a pre-trained model (e.g., ResNet50 or VGG16) and adapt it to our dataset.
3. **Image Segmentation**: Divide an image into smaller patches and classify each of those patches using the trained model.
4. **Model Evaluation**: Evaluate the performance of the model using the test dataset.

## Project Steps

1. **Data Preprocessing**: Clean and organize the dataset into training and testing directories.
2. **Creating a Model from Scratch**: Train a deep learning model from scratch, without using pretraining, to get basic classification results.
3. **Transfer Learning**: Load a pre-trained model, adapt it to our dataset, and fine-tune it to improve accuracy.
4. **Image Segmentation**: For each image, divide it into smaller patches and classify each of those patches, enabling multi-class classification within a single image.
5. **Model Evaluation**: Evaluate the model's performance using the test dataset and analyze performance metrics such as accuracy and confusion matrix.

## Technologies Used

- **Python**
- **Jupyter Notebooks**
- **Keras/TensorFlow** (for deep learning)
- **OpenCV** (for image segmentation)
- **Matplotlib** (for data visualization)

## Installation

To run this project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/image-classifier-dl.git
    ```

2. Install the necessary dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook to train the model and perform image segmentation.

## How It Works

1. **Data Splitting**: The dataset is divided into training and testing sets using a Python script that ensures a proper split without data leakage.
2. **Training the Model**: A deep learning model is trained from scratch, and in the second part, a pre-trained model (e.g., ResNet50 or VGG16) is loaded and adapted to the dataset.
3. **Image Segmentation**: For each input image, it is divided into smaller patches, and the model classifies each of these patches, allowing the recognition of multiple landscapes within a single image.

## Model Evaluation

The model's performance is evaluated on a test dataset to determine how well it can classify the images. Additionally, metrics such as accuracy and the confusion matrix are used to assess the model's effectiveness.

## Conclusions

### Part 1: Creating the Model from Scratch

- Good results were achieved, as the validation accuracy was close to 90%. However, this might be due to a phenomenon known as "data leakage," where validation accuracy is sometimes higher than training accuracy. This can happen when some samples from the training or validation set are mistakenly included in the other set, causing the model to memorize rather than predict.
- To address this and improve performance, transfer learning was applied to increase accuracy in both the training and validation sets. Once the model was trained, it was tested using a test dataset, ensuring that these samples had never been seen by the model.

### Part 2: Image Segmentation

#### Analysis of Results:

1. **Water Areas**: The model can almost perfectly distinguish water areas from other regions/classes.
2. **Desert vs Green Areas**: The model struggles to clearly differentiate between desert and green areas but is quite effective at distinguishing between green areas and water, achieving near-perfect separation.
3. **Clouds**: The model does not perform well when clouds are present in the images; it often confuses them with green areas. In general, it almost never detects clouds correctly.
4. **Image Sources**: The images used for the "image segmentation" task were not taken from the same dataset used to train the model. That is, the model was trained with images captured by one satellite, but the test images came from different satellites. This affects the results because variations in resolution, zoom level, and color filters used between satellites exist.
5. **Data Augmentation**: Had we made better use of data augmentation (e.g., altering the colors of the training images), and retrained the model, it might have learned to better differentiate the classes regardless of image quality or satellite used. However, the main goal was for the model to predict images from the test dataset, which it successfully managed to do. While performance drops a bit when extending the analysis to image segmentation, this is understandable since the model was not originally designed for this task.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
