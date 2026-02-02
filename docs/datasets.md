# Datasets

[Back to README](../README.md) | [Back (Installation)](./installation.md) | [Next (Training)](./training.md)

This document provides instructions for setting up the pre-processed datasets required for using the Neural Atlas Graphs (NAG) repository.
Currently we are happy to release all pre-processed data which we created for our experiments on the Waymo open dataset and the Davis dataset. To a later point in time, we are planning to provide scripts to convert further Waymo segements as well as further Davis sequences into our used formats. Create a GitHub issue if you are interested in this or have any questions.


## Downloading the Datasets

Our pre-processed datasets can be downloaded from the Sciebo cloud storage:  
https://uni-siegen.sciebo.de/s/8koHXQw6ZJsR2D2?path=%2F


Please download the `waymo` and `davis` folders from the above link and place them into the `data/datasets` folder of the repository, such that you have the following structure:

```
nag/
  nag/
    ...
  data/
    datasets/
      waymo/
        segment-xxxx/
      davis/
        [davis sequences]/
  .env
```

When using a different structure, make sure to use your actual dataset path when configuring the dataset path in the config files, and / or adjust the `DATA_PATH` environment variable accordingly.


## Environment Variables

To facilitate easy access to the datasets, you need to set the `DATA_PATH` environment variable to point to the `data/datasets` folder. This can be done by adding the following line to your `.env` file in the root of the repository:

```
DATA_PATH=data/datasets
```

Make sure to adjust the path if your datasets are located elsewhere on your system.


> **Note:** Ensure that the datasets are correctly placed and the environment variable is set before running any training or evaluation scripts to avoid path-related errors.
>

For more details on how to use the datasets with our training scripts, please refer to the [Training Instructions](./training.md).

## Custom Dataset Preparation

If you wish to prepare further Waymo segments for training or convert additional Davis sequences, we refer to the [Waymo Dataset Preparation Guide](./waymo_dataset_extraction.md) and the [Davis Dataset Preparation Guide](./davis_dataset_preparation.md) for detailed instructions on how to do so.