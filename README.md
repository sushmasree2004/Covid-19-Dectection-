# COVID-19 Detection Using Chest X-Rays

##  Overview

A deep learning-based project for detecting **COVID-19 from chest X-ray images** using Convolutional Neural Networks (CNNs) and Transfer Learning.

The project evaluates multiple pre-trained CNN architectures, including **VGG16, ResNet50, and InceptionV3**, and uses image augmentation techniques to improve model generalization and reduce overfitting.

##  Features

* COVID-19 detection from chest X-ray images
* CNN-based image classification
* Transfer Learning using pre-trained models
* Comparison of VGG16, ResNet50, and InceptionV3
* Image augmentation using rescaling, zoom, and shear
* Model evaluation using validation accuracy and classification metrics

##  Tech Stack

* Python
* TensorFlow
* Keras
* Scikit-learn
* CNN
* Transfer Learning
* NumPy
* Pandas
* Matplotlib
* OpenCV

##  Dataset

The project uses the **COVID-19 Radiography Dataset**, containing approximately **10,000 chest X-ray images** for training and evaluation.

The images were preprocessed and augmented before being provided to the deep learning models.

##  Models Used

The following CNN architectures were evaluated:

* **VGG16**
* **ResNet50**
* **InceptionV3**

Pre-trained models were used as feature extractors through **Transfer Learning**, followed by custom classification layers for COVID-19 detection.

##  Data Preprocessing

To improve model performance and reduce overfitting, the following image preprocessing and augmentation techniques were applied:

* Image rescaling
* Zoom augmentation
* Shear augmentation
* Image normalization

```python
ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2
)
```

##  Workflow

```text
Chest X-Ray Dataset
        ↓
Data Preprocessing
        ↓
Image Augmentation
        ↓
Transfer Learning
        ↓
VGG16 / ResNet50 / InceptionV3
        ↓
Model Training
        ↓
Model Evaluation
        ↓
COVID-19 Prediction
```

##  Results

The models were trained and evaluated on the chest X-ray dataset.

**Best Validation Accuracy: 83%**

Different CNN architectures were compared to identify an effective model for COVID-19 image classification.

##  Project Structure

```text
COVID-19-XRay-Detection/
│
├── dataset/
├── notebooks/
│   └── covid_detection.ipynb
│
├── models/
├── train.py
├── requirements.txt
└── README.md
```

##  Installation

Clone the repository:

```bash
git clone https://github.com/sushmasree2004/Covid-19-Dectection-.git
```

Navigate to the project directory:

```bash
cd COVID-19-XRay-Detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

##  Usage

Run the training script:

```bash
python train.py
```

Or open the Jupyter Notebook:

```bash
jupyter notebook
```

Run the notebook cells sequentially to preprocess the dataset, train the models, and evaluate their performance.

##  Evaluation Metrics

The models can be evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

##  Future Improvements

* Experiment with newer architectures such as EfficientNet
* Perform hyperparameter tuning
* Increase dataset size and diversity
* Implement ensemble learning
* Deploy the model as a web application
* Use Grad-CAM for model explainability



##  Author

**Sushma Sree Balijapalli**


