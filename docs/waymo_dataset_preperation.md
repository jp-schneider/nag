# Waymo Dataset Extraction Guide

[Back to README](../README.md) | [Back (Datasets)](./datasets.md)

This document provides instructions on how to extract and utilize custom, not pre-processed Waymo Open Dataset for training Neural Atlas Graphs.

Part of the extraction process relies on the official Waymo Open Dataset tools, further on the EmerNerf Waymo processing codebase, so you might want to check their [GitHub repository](https://github.com/NVlabs/EmerNeRF)
aswell. 

## Prerequisites

If you haven't already, please follow the initial setup instructions in the [Neural Atlas Graphs README](../README.md) to install necessary dependencies and set up your environment.

As the waymo open dataset provides no windows support for their processing tools, a linux or macOS environment is required. You can use WSL2 on Windows to create a compatible environment.

### P.1 Registered for the Waymo Open Dataset 

You need to register and accept the terms of use for the Waymo Open Dataset. You can do this by following the instructions on the [Waymo Open Dataset website](https://waymo.com/open/). 

### P.2 Installed gcloud executable

Further you need the google cloud sdk installed to download the Waymo dataset files. You can find installation instructions [here](https://cloud.google.com/sdk/docs/install).

### P.3 Configured gcloud with your Google account

After installing the gcloud sdk, you need to initialize it and authenticate with your Google account. You can do this by running. Make sure to use the same Google account for both the registration at waymo and within this init.:

```bash
gcloud init
```

#### P.4 Install gcloud hash tool

To download the Waymo dataset files, you also need to install the `gcloud` hash tool. You can do this by running:

```bash
gcloud components install gcloud-crc32c
```

#### P.5 Set up Additional Python Environment

Since the Waymo dataset processing relies on specific versions of certain libraries and tensorflow, it is recommended to set up a separate Python virtual environment for this task. You can use `poetry`, `venv` or `conda` to create a new environment. Here, we provide instructions using `poetry`.

Inside the `nag-waymo-processing/` submodule directory is an additional `pyproject.toml` file that specifies the required dependencies for the Waymo dataset processing. To set up the environment, navigate to this directory and run:

```bash
poetry shell
```

Make sure that `which python` points to the correct virtual environment before proceeding with the dataset extraction.
E.g. its name should be something like `nag-waymo-processing-...` depending on your python version. 

If this is correct, install the dependencies by running:

```bash
poetry install
```

#### P.6 Download SAM2 Checkpoint

If you want to create dense segmentations for new datasets, we recommend to pre-run SAM2 on the existing scarce segmentations within the waymo segments and fine tune them manually afterwards. To run sam2, you will need to download a checkpoint to the checkpoint directory using:

```bash
cd nag-waymo-processing/data/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt 
```

## 1. Creating a Data Set Path

To download and extract Waymo segments, you first need to create a folder where the extracted segments will be stored. For example, you can create a folder named `waymo/raw` and `waymo/processed` within the `data/` directory of the repository using:

```bash
mkdir -p ./data/waymo/raw   
mkdir -p ./data/waymo/processed 
```

## 2. Downloading Waymo Segments

To download individual Waymo segments, you can use the `cli.py` script provided in the `nag_waymo_processing` subpackage. This script allows you to specify scene IDs or a split file to download specific segments.
We provided our own split file located at `./nag-waymo-processing/data/nag.txt` where you can find the list of scene IDs we used for our experiments in the main paper, while `./nag-waymo-processing/data/nag_ext.txt` includes additional scene IDs used in the supplementary material. The format of the split file corresponds to the one used in the EmerNerf repository.

To download all the segments on our main paper, run the following command:

```bash
nagwp download --split-file ./nag-waymo-processing/data/nag.txt --target-dir data/waymo/raw
```

Whereby the following will also download the extended segments from our supplementary.

```bash
nagwp download --split-file ./nag-waymo-processing/data/nag_ext.txt --target-dir data/waymo/raw
```

You can also download special sequences using scene ids, matching the 0-based line numbers in the [waymo_train_list](../nag-waymo-processing/data/waymo_train_list.txt) e.g.:

```bash
nagwp download --scene-ids 115 --target_dir data/waymo/raw
```
for the segment "segment-12511696717465549299_4209_630_4229_630_with_camera_labels.tfrecord".

## 3. Extract Waymo Data

Once the '.tfrecord' files are downloaded, we provide a script to extract these into our intermediate data directory for further processing.
You can either use the scene ids as above, or segment ids, which refers to the number within the tfrecord name `segment-[segment_id]_[...]`.  You dont have to provide the full id, it must be only unique within the downloaded data.

```bash
nagwp extract --scene-ids 115 --target_dir data/waymo/processed
```

This will extract the segment with scene id 115 e.g. segment-12511696717465549299_[...] into a new folder called
segment-12511696717465549299 within the `data/waymo/processed/` directory. This is further referred to as base_path in the following commands.
It produces the following structure:

```
data/waymo/processed/segment-12511696717465549299/
├── boxes
    └── raw
        ├── 00.json
        ├── ...
        └── N.json
├── initial_masks
├── images
└── segment_info.json
```


Within the folder you will now find the extracted images, initial_masks (Segmented frames available in waymo), boxes (3D bounding boxes) 

## 4. Matching Boxes

After extracting the Waymo data, you need to match the 3D bounding boxes to the segmentation masks since they are not aligned by default. You can use the `match_boxes` command provided in the `cli.py` script for this purpose.



```bash
nagwp match_boxes --base-path data/waymo/processed/[my_segment_folder]
```


## 5. Producing Dense Segmentations

Only a part of the frames within each waymo sequence is segmented, usually around 10 %. 