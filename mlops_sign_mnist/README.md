# Sign Language MNIST Classification Using Convolutional Neural Networks (CNNS)

### Project Description
The goal of our project is to use convolutional neural networks (CNNs) to accurately classify hand gestures from the Sign Language MNIST dataset. This dataset contains images of American Sign Language (ASL) letters, and our aim is to create a model that can recognize and classify these gestures with high accuracy.

### Objective 
Our primary objective is to develop a CNN model capable of classifying ASL hand gestures represented in the MNIST-like dataset. This project will involve data preprocessing, model training, evaluation, and optimization to achieve the highest possible classification accuracy.

### Framework and Integration
We utilize the PyTorch framework for constructing our deep neural network for predictions.

### Data Colleciton and Initial Dataset 
The dataset used in this project is sourced from Kaggle and is known as the Sign Language MNIST dataset. It consists of 28x28 pixel grayscale images of hand gestures corresponding to 24 ASL letters (excluding J and Z due to their dynamic nature).

Dataset details:
1. Training images: 27,455
2. Testing images: 7,172

### Model Architecture
We will employ a Convolutional Neural Network (CNN) due to its effectiveness in image recognition tasks.

Our models will be evaluated based on : Accuracy (The proportion of correctly classified images).


### Team Members
1. Anjali Sarawgi
2. Ali Najibpour Nashi
3. Annas Namouchi
4. John-Pierre Weideman



## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── mlops_sign_mnist  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
