## Day 1: 02-01-2025

We will be using the Airbus Ship Detection dataset which can be found in [Kaggle](https://www.kaggle.com/competitions/airbus-ship-detection). This dataset contains two different pieces of information:

* **Training information**: on the one hand, the dataset provides the user with data that can be used to train the models to be used in the original competition, including object detection models (e.g., YOLO), segmentation models (e.g.,  UNet), etc.

* **Submission information**: similarly, the dataset contains input images whose predictions, performed with any sort of technique, can be submitted to the competition, so as to obtain an evaluation of how good the model solves the task proposed in the original competition.

Since the dataset is used for academic purposes, completely allowed and supported by their creators as shown in the [license](https://www.kaggle.com/competitions/airbus-ship-detection), information related to the submission will not be used. Conversely, the training information will be used to obtain: (I) satellite imagery of seas and oceans without ships, which will be used as the "non-anomalous" samples of data; and (II) satellite imagery of seas and oceans including images with and without ships. This split of data allows us to train anomaly detection models, evaluating their goodness on known, ground truth, satellite imagery. For this reason, the original training dataset will be divided into three different partitions: train, validation and test. The first partition will be used for training the models, the second one will be used for deciding the best hyperparameters for the models, and the latter will be used for evaluating their goodness. This decision looks optimal because the datasets employed are big, thus there is enough information for training the models and evaluating them in an honest manner.

## Day 2: 03-01-2025

Another relevant aspect to be considered is the required preprocessing in the data. Regarding the Airbus Ship Detection dataset, several preprocessing steps are required to maximise the quality of the data to be used for training models:

* **Tile generation**: the input images of the dataset have a shape of 3x768x768. Since these images have a relatively big shape, each individual image is split into a grid of 9 different sub-images of size 3x256x256. The application of this tiling strategy is two-fold: (I) on the one hand, it allows us to obtain multiple tiles (anomalous and non-anomalous) for a single individual image, effectively increasing the size of the data to be used for training and developing more effective models; on the other hand (II), using smaller tiles contributes to a reduction of the memory usage, as the computational device employed has a limited amount of GPU VRAM (6GB).

* **Data normalisation**: firstly, applying normalisation to the input data can lead to more robust and effective models [Normalization effects on deep neural networks Jiahui Yu, Konstantinos Spiliopoulos], thus Mean-Scale (or Z-score) normalisation will be applied, as it can make the resulting model more robust to datasets with outliers or unfound values in the training dataset [omogeneous Data Normalization and Deep Learning: A Case Study in Human Activity Classification].

* **Mask calculation**: furthermore, the input dataset only provides information about the borders of each ship. While such information is useful, ship masks are required, so that tiles can be split into tiles without ships (non-anomalous) and tiles with ships (anomalous).

With all this in mind, the development started by using the AFML framework to create a basic pipeline that could apply these preprocessing steps automatically. The introduction of MLOps frameworks, such as Automation Framework for Machine Learning ([AFML](https://github.com/AlbertoVelascoMata/afml)), is the quality improvement of the resulting models, even in production environments [AI-powered DevOps and MLOps frameworks: Enhancing collaboration, automation, and scalability in machine learning pipelines].

## Day 3: 04-1-2025

Mask calculation was developed during this third day. However, a quick visualization of a small sample of images of the dataset showed a quality issue that could affect the performance of anomaly detection models: images include not only sea/ocean images, but also images with land, and even images without any water at all. While the availability of this information could help develop more robust models, the unavailability of the "land mask" does not allow the creation of a balanced dataset with sea and land images. As a result, it was decided to use this dataset when studying the limitations of the anomaly detection models, whereas the "MASATI (v2)" dataset was used to replace the airbus ship detection dataset, as it provides full control on the kind of images with which we can train the anomaly detection models.

## Day 4: 06-01-2025

The MASATI (v2) dataset was downloaded and the partitioning strategy was implemented. This strategy worked as follows:

* **Train and validation partitions**: first, the train and validation partitions were obtained from the "water" sub-set of aerial images. Such sub-set of images only contains images with water, i.e. no land, and no ships. Indeed, the utilisation of this sub-set of images seems to be ideal for training anomaly detection models aimed at detecting ships through non-supervised anomaly detection methods.

* **Test partition**: on the other hand, the test partition was obtained from the combination of the "ship" and "multi" sub-sets of images, which contain sea/ocean images with just one ship (in the former case) and multiple ships (in the latter case).

The result of this partitioning is a *partitions.csv* file containing the path of the image (relative to the dataset folder), some metadata related to it (such as the width and height of the image), and the partition it belongs to.

The application of this partitioning strategy allows us to use an *overlap tiling* strategy to further increase the number of images in the train partition. An *overlap tiling* strategy is that strategy by which an image of size CxHxW is converted into N CxH'xW' images, which are overlapped, i.e. two images may have a shared part in different locations of the image. The *overlap tiling* strategy, not implemented during this development day, allows the generation of more images from an original, relatively small, sub-set of images (as is the case with the MASATI v2 dataset, which is a rather small dataset). The *overlap tiling* strategy that will be implemented is based on picking up 3x128x128 images, with steps of 64 pixels at a time, both in the X and the Y axes of the image. As a result, the tiling strategy will work as follows:

* First, an image of size 3x128x128 is obtained from the X = 0 and Y = 0 location.

* Then, a step of 64 pixels is applied to the X axis (X = 64). Another 3x128x128 image is obtained from this new location (X = 64, Y = 0). It must be noted that the image mentioned in the previous step and the one mentioned in this step share a common portion of the image.

* This process is repeated until a step is applied where X does not allow the obtention of a image of 3x128x128. In that case, the X location is reset to 0, and the Y location is increased by 64 (same step size). The full process would continue until both the X and Y locations do not allow for the generation of more images of size 3x128x128.

## Days 5 to 7: (until 10-01-2025 due to illness)

Once images were split into train/val/test subsets, two processes followed:

* First, images were tiled into smaller patches to (I) reduce memory consumption and (II) potentially improve the performance of the anomaly detection methods, as anomalies can be dilluted when images are large (dillution in loss).

* Furthermore, masks were generated for all images. Training and validation images involve a mask where all pixels are 0, i.e. no ships, whereas test images include ship pixels.

With all the preprocessing so far, only a few more steps were required for the dataset to be fully usable for training anomaly detection methods:

* **Mask tiling**: similar to the images, masks must also be tiled so that patches can be fully utilised themselves. The same tiling must be applied so that the mask of the patch carefully relates to the patch itself, without any deviation.

* **Stat calculation**: since data normalization techniques will be implemented to create more robust models, stats like training data mean (per colour channel) and scale (per colour channel too) must be computed, so that images can be normalised using only training information.

* **DataLoader integration**: on the other hand, since PyTorch is used to implement all the anomaly detection methods, a class must be implemented to extend the dataset class of PyTorch, so that the dataset is iterable and can easily be used for training these anomaly detection methods in a simple way.

## Day 8: 11-01-2025 (still ill due to flu)

The stats calculation task was developed during this day. The essence of this task was to develop the functionality that allows us to loop through all train images, calculating the mean and standard deviation of each color channel for any given dataset (only train information to prevent data leakage). With those two pieces of information, the normalization of patches is trivial, as a simple mean-scale normalization can be applied once both mean and scale are known.