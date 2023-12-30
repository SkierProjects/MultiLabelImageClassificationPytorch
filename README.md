# Multi-Label Image Classification using Pytorch

MultiLabelImageClassificationPytorch is a robust and flexible library designed to simplify the process of multilabel image classification with a dataset of images. This library provides a suite of scripts and modules to load various models, fine-tune hyperparameters, and even train multiple models sequentially with ease. The project boasts in-depth visualization support through TensorBoard, enabling users to compare performance across models and optimize their specific use cases efficiently.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Understanding Results](#understanding-results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up the environment to run the code, follow these steps:

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Environment Setup

1. Clone the repository to your local machine:
   ```sh
   git clone https://github.com/SkierProjects/MultiLabelImageClassificationPytorch.git
   cd MultiLabelImageClassificationPytorch
   ```

2. Create and activate the conda environment from the `environment.yml` file:
   ```sh
   conda env create -f environment.yml
   conda activate multilabelimage_model_env
   ```

## Usage

### Dataset Preparation

Place your `dataset.csv` file in the `Dataset` directory. The CSV file should have the following format:
```
filepath,classname0,classname1,...
/path/to/image1.jpg,0,1,...
/path/to/image2.jpg,1,0,...
```
Run `analyzeData.py` to get insights on the class balance in the dataset:
```sh
python Dataset/analyzeData.py
```

Run `computemean.py` to calculate the mean and standard deviation of your dataset
```sh
python src/computemean.py
```
Take the outputs for the mean and standard deviation and place them inside `src/config.py` for `dataset_normalization_mean` and `dataset_normalization_std`. These will be used to normalize the images used for training.

### Training a Model

To train a model, use the `train.py` script:
```sh
python src/train.py
```

For training multiple models, use the `train_many_models.py` script. Modify `train_many_models.json` to include the models you want to train, which overrides values in `config.py`:
```sh
python src/train_many_models.py
```

### Evaluating a Model

To evaluate the performance of a model, you can use the `test.py` script:
```sh
python src/test.py
```

### Evaluating a Model

To run a model and get results at runtime, you can use the `inference.py` script:
```sh
python src/inference.py
```

### TensorBoard

To view TensorBoard logs, run:
```sh
tensorboard --logdir=tensorboard_logs
```

## Understanding Results

The results of the model training and evaluation will be stored in the following directories:
- `logs`: Contains log files with detailed information about the training process.
- `outputs`: Contains saved models in `.pth` format.
- `tensorboard_logs`: Contains TensorBoard logs for visualizing training progress and metrics.

## Project Structure

- `Dataset/`: Contains the dataset and scripts for analyzing the dataset.
- `logs/`: Where training logs are stored.
- `outputs/`: Where trained model weights and checkpoints are saved.
- `tensorboard_logs/`: Where TensorBoard logs are output.
- `src/`: Contains all source code for the project.
- `environment.yml`: Conda environment file with all required dependencies.

## Contributing

To contribute, please submit a pull request to the repository. Your contributions will be reviewed and considered for merging.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for more details.