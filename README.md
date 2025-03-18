# Anomaly detection from satellite imagery

![Python3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![v1.0.0](https://img.shields.io/badge/version-1.0.0-blue)

This repository contains all the code, documentation and related information about "Anomaly detection from satellite imagery", the Master Thesis developed by Raúl Barba Rojas with the supervision of Jorge Díez Peláez and José Luis Espinosa Aranda within the scope of the "Master Degree in Artificial Intelligence Research" offered by UIMP and AEPIA.

This project has a two-fold objective and scientific contribution:

1. We **evaluate different state-of-the-art approaches to anomaly detection in satellite imagery**. We further analyse the results, noticing that self-supervised learning state-of-the-art anomaly detection methods usually find it harder to deal with false negatives (and/or false positives, depending on the datasets and fields of application).

2. We propose a novel method, **DualAnoDAE**, which utilises an ensemble of two AutoEncoders with different receptive fields, to further improve the anomaly detection capabilities of the model. DualAnoDAE achieves state-of-the-art results in the two datasets used for evaluation.

The results shown in the paper are fully reproducible and the project remains open-source, so that it can help the scientific community design and develop more robust and effective anomaly detection methods. While the project is open-source and uses the MIT license, please cite this work if you consider it helpful for your research.


## Reproducibility

Reproducibility is a very relevant concern in current scientific research. This section describes how you can fully reproduce the results we show in our work. The steps involve installing our `aerial_anomaly_detection` package, downloading the datasets used for training and evaluation anomaly detection methods, as well as guidelines on how to download the weights of the trained models (optional), and evaluating the models with trained weights.

### Installation

Our project can simply be installed in two different ways: (I) using GitHub, and (II) cloning and building from source code. In both cases, the usage of virtual environments is highly recommended. Please pay attention to `pyproject.toml` for a further understanding of the dependencies of this project.

#### GitHub Installation

```
pip install git+https://github.com/RaulBarbaRojas/AerialImageryAnomalyDetection.git@v1.0.0
```

#### Local Installation

```
git clone https://github.com/RaulBarbaRojas/AerialImageryAnomalyDetection.git@v1.0.0
cd AerialImageryAnomalyDetection
pip install .
```

#### Checking Installation

Last, the installation can easily be checked by opening the Python Interpreter CLI (with the virtual environment if required) and running the following:

```python
try:
    import aerial_anomaly_detection
    print('Module `aerial_anomaly_detection` successfully installed')
except ImportError:
    print('Module `aerial_anomaly_detection` cannot be found')
```

### Datasets

Our work uses two datasets in order to train and evaluate different anomaly detection approaches: LandCover.ai and HRC_WHU. Both datasets are deeply described in the following sections, together with the URLs that can be used for downloading them. Both datasets are publicly available for scientific research. If you use these datasets, please cite the work of the original authors as we have done in our paper.

> **Note**
> Although the datasets often include a previous partitioning, our work uses a random partitioning to prevent biases.
>
> Similarly, our work uses our own scripts to divide the satellite acquisition (scene) into smaller patches. For more details, please see the technical documentation `docs/aerial_anomaly_detection.html` for a further understanding of the `tiler` used.

#### Landcover.ai

The LandCover.ai dataset is a well-known dataset for segmentation problems. We use this dataset to train our anomaly detection models in one (randomly chosen) class, namely water, so that the they can be used to detect non-anomalous (water) and anomalous (non-water) patches. The dataset is publicly available for research (please, cite their work in case you are going to use it) and can be downloaded in the [official webpage](https://landcover.ai.linuxpolska.com/).

Once downloaded, the following structure must be created (please keep the exact same names used in the diagram below, or change the corresponding variables in the `project.yml` file, so that [AFML](https://github.com/AlbertoVelascoMata/afml) can correctly find the dataset):

```
📂 AerialImageryAnomalyDetection/
├── 📂 data/
│   ├── 📂 download/
|   |   ├── 📂 LandCoverAI/
|   |   |   ├── 📂 images
|   |   |   ├── 📂 masks
|   |   |   ├── 📄 split.py
|   |   |   ├── 📄 test.txt
|   |   |   ├── 📄 train.txt
|   |   |   ├── 📄 val.txt
├── 📄 README.md
```

Once the download folder with the LandCoverAI dataset is set up, the user must execute the following code for preprocessing the dataset (make sure that the project and its dependencies are installed):

```python
afml run -j "Preprocess LandCover.ai dataset"
```

This code must be run in a Python CLI (with active virtual environment if required) using the main folder of this repository as the working directory. If no errors are found, a `data/processed/LandCoverAI` folder will be created with binary files containing information regarding partitioning of satellite imagery acquisitions, the (train) tiles that will be used for training the models, among other relevant aspects.

#### HRC_WHU

HRC_WHU is a cloud segmentation dataset publicly available. It contains only two classes of pixels (cloud and non-cloud), and it was used in the same manner as the previous one to simulate an anomaly detection problem, by training the models on just one of the two classes. In this case, we selected clouds as the non-anomalous class, therefore a tile that does not contain clouds had to be detected by the anomaly detection models as an anomalous patch. The dataset is described in the [official GitHub repository](https://github.com/dr-lizhiwei/HRC_WHU), which also contains the download links.

The following structure must be met:

```
📂 AerialImageryAnomalyDetection/
├── 📂 data/
│   ├── 📂 download/
|   |   ├── 📂 HRC_WHU/
|   |   |   ├── 📂 HRC_WHU
|   |   |   ├── 📄 HRC_WHU：High-resolution cloud cover validation data.pdf
|   |   |   ├── 📄 List of training and test data.txt
├── 📄 README.md
```

Once the download folder with the HRC_WHU dataset is set up, the user must execute the following code for preprocessing the dataset (make sure that the project and its dependencies are installed):

```python
afml run -j "Preprocess HRC_WHU dataset"
```

This code must be run in a Python CLI (with active virtual environment if required) using the main folder of this repository as the working directory. If no errors are found, a `data/processed/HRC_WHU` folder will be created with binary files containing information regarding partitioning of satellite imagery acquisitions, the (train) tiles that will be used for training the models, among other relevant aspects.

### Models

This section explains how to reproduce our results using the implemented pipeline. However, a user could be interested in two different things:

1. **Training the models on its own**: if a user wants to train the implemented models, with the same or different settings, please refer to [Training models with aerial_anomaly_detection](#training-models-with-aerial_anomaly_detection).

2. **Using the trained models to verify the evaluation results**: if the user simply wants to use the models that were trained and whose evaluation results where used for writing our scientific paper, please refer to the [Using the trained weights](#using-the-trained-weights).

#### Training models with aerial_anomaly_detection

In order to train models with the implemented package and pipeline, the following commands can be run on a Python CLI (with the active virtual environment if required and using the main directory of this repository as the working directory). Note that having preprocessed LandCover.ai and HRC_WHU datasets is a requirement for the models to be trained correctly. Please refer to [Datasets](#datasets) to download and correctly set up both datasets.

```python
afml run -j "Train DCGAN components"
afml run -j "Train BiGAN components"
afml run -j "Train DualAnoDAE components"
afml run -j "Train anomaly detectors"
```

The four commands must be **run in order**, because certain models may be required to train others. For example, ZIZ and IZI architectures require a pre-trained GAN (e.g., BiGAN or DCGAN) to be trained. Similarly, our DualAnoDAE approach requires two pre-trained AutoEncoders. These "previous steps" are performed with the first three commands, whereas the last command trains all the anomaly detection models (except for BiGAN, which is trained with the second command, as it could be used for training other GAN-dependant architectures).

Please read and adjust `project.yml` so that the training process uses the desired settings. Adjustable settings include the number of epochs, the batch size, learning rate among other aspects. Please make sure to use the same tile size as the one used when preprocessing the datasets (otherwise, delete the `data/processed` folder and preprocess the datasets again with your desired tile size, making sure it is correctly replaced for all the pipeline jobs that require it).

Model weights will be stored in the `runs` folder, which will have as many subfolders as trained anomaly detection models, each of them having one subfolder per dataset used (each model has weights for each implemented dataset).

> **Note**
> Please note that training models is a costful process and can take a significant amount of time.


#### Using the trained weights

If the user simply wants to download pre-trained models to quickly verify the obtained evaluation results, they can be downloaded from the official releases of this repository. Within the release, a `runs.zip` file is attached, which must be placed according to the structure described next. Such a folder contains the weights of the models that were trained during our research and allow to obtain the exact same results that can be found in our paper.

```
📂 AerialImageryAnomalyDetection/
├── 📂 runs/
│   ├── 📂 AutoEncoder/
|   ├── 📂 BiGAN/
|   ...
├── 📄 README.md
```

### Evaluation

In order to run the automated evaluation of all anomaly detection methods on the implemented datasets, please use the following command in a Python CLI (with the active virtual environment if required and using the main directory of this repository as the working directory):

```python
afml run -j "Evaluation"
```

> [!IMPORTANT]  
> Evaluation can only succeed when trained weights are available. Please train your own models or use our trained weights as described in [Models](#models).

## Documentation

The paper linked to this repository can be found in `docs/paper.pdf`. It describes the problem, existing state-of-the-art approaches and their limitations, our proposed method DualAnoDAE, as well as a comparison of results in two different datasets: LandCover.ai and HRC_WHU. See the paper or the reproducibility section of this repository to obtain further details.

Besides, this project uses `Sphinx` to create its technical documentation. The code documentation contains all relevant information for developers to understand, and potentially extend, the functionalities of the `aerial_anomaly_detection` package.

The technical documentation is available in HTML and can be visualized by opening `docs/_build/aerial_anomaly_detection.html` using any browser.

### Regenerating documentation

#### In a Windows system

To regenerate the project's documentation in a Windows system, we can execute the following commands:

```bash
cd docs
.\make.bat html
```

#### In a Linux system

Alternatively, if the user has a Linux system, then the following commands can be used.

```bash
cd docs
make html
```

## Contributing and issues

We welcome contributions from the open-source community. If you want to extend the capabilities of `aerial_anomaly_detection`, please fork the repository, develop your improvements, and submit a pull request. Please follow similar coding guidelines than the ones used, so that the project documentation remains coherent.

Similarly, if a user has **problems when running the project, please make sure to create an issue on this repository** and we will try to solve it as soon as possible. **Questions, bugs, comments and suggestions for improvement are completely welcome** and we will try our best to address them in the shortest amount of time possible.

## Credit

This work was carried out by Raúl Barba Rojas, together with José Luis Espinosa Aranda and Jorge Díez Peláez.
