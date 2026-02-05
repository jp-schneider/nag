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

Inside the `libs/nag-waymo-processing/` submodule directory is an additional `pyproject.toml` file along with cli tools which specifies the required dependencies and routines for the Waymo dataset processing. To set up the environment, navigate to this directory and run:

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

## Processing Waymo Dataset

In the following, we will go through the steps to download and extract Waymo segments, match the 3D bounding boxes to the segmentation masks, and create dense segmentations using SAM2, and briefly explain how to refine the segmentations using the Anylabeling tool.

Some of the steps will create checkpoint files ([Identifier].done), which are used to avoid redundant processing and potential overwriting of existing files. If you want to re-run a step, you can simply delete the respective checkpoint file, which will trigger the re-processing of the respective step. The checkpoint files are stored directly in the respective output directories, so you can easily identify them.

## 1. Creating a Data Set Path

To download and extract Waymo segments, you first need to create a folder where the extracted segments will be stored. For example, you can create a folder named `waymo/raw` and `waymo/processed` within the `data/` directory of the repository using:

```bash
mkdir -p ./data/waymo/raw   
mkdir -p ./data/waymo/processed 
```

## 2. Downloading Waymo Segments

To download individual Waymo segments, you can use the `cli.py` script provided in the `nag_waymo_processing` subpackage. This script allows you to specify scene IDs or a split file to download specific segments.
We provided our own split file located at `./libs/nag-waymo-processing/data/nag.txt` where you can find the list of scene IDs we used for our experiments in the main paper, while `./libs/nag-waymo-processing/data/nag_ext.txt` includes additional scene IDs used in the supplementary material. The format of the split file corresponds to the one used in the EmerNerf repository.

To download all the segments on our main paper, run the following command:

```bash
nagwp download --split-file ./libs/nag-waymo-processing/data/nag.txt --target-dir data/waymo/raw
```

Whereby the following will also download the extended segments from our supplementary.

```bash
nagwp download --split-file ./libs/nag-waymo-processing/data/nag_ext.txt --target-dir data/waymo/raw
```

You can also download special sequences using scene ids, matching the 0-based line numbers in the [waymo_train_list](../libs/nag-waymo-processing/data/waymo_train_list.txt) e.g.:

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
└── images
```


Within the folder you will now find the extracted images, initial_masks (Segmented frames available in waymo ~10 %), boxes (3D bounding boxes) encoded as json where the box is a serialized version of the [ProjectedTimedBox3D](../tools/tools/labels/projected_timed_box_3d.py) class, which includes the 3D box parameters as well as the projected 2D box parameters along all frames. 

## 4. Matching Boxes

After extracting the Waymo data, one need to match the 3D bounding boxes to the segmentation masks since they are not aligned by default. You can use the `match_boxes` command provided in the `cli.py` script for this purpose.

```bash
nagwp match_boxes --base-path data/waymo/processed/[my_segment_folder]
```

The script will match all masks in waymo with the projected label for each box, calculate the IoU and assign the best matching box to each mask. The IoU scores are stored within the `mapping_results` folder inside the base_path, per masked frame (`[frame_index].json`). Each json file is a dict, mapping each box id to its IoU score with all mask ids, as well as the mask ids.

Further it will write the matched mask id (the one with highest mean IoU) into the respective box.json file (`object_id` property) within the `boxes/raw` folder. Given that the waymo masks are only "unique" per "frame group" there might be multiple masks referring to the same box across the sequence. We are just assigning the best matching mask to each box, so there might multiple masks referring to the same box, but only one mask is assigned.
To visualize the matching results, the script also produces a visualization per frame in the `inpainted` folder, where the matched box and masks are drawn on the respective image. In the legend refers to the '[mask_id/object_id]:[box_id]' format. If mask_id is `????` then there was no mask assigned to the box - e.g. because the object was never segmented in the waymo segment. This often happens for early or late frames in the sequence, when they leave / enter the field before / after the first / last segmented frame.
These objects must taken care later on, so within `inpainted\boxes_with_missing_init_masks` you can find the same visualizations but only for the boxes which have no assigned mask showing the `[box_filename]:[box_id]`, so you can easily identify them and fill missing ids later on.

If on a masked frame an object is not masked but mapped, then the object was most likely mapped with a mask id from a different frame group, which is not mapped in the current one.

If you find errors in the matching, you can also manually correct the assigned box ids within the `boxes/raw/[frame_index].json` files, which will be used for all further processing steps.

## 5. Producing Dense Segmentations

Since the Waymo dataset only provides sparse segmentations for certain frames, and we would like to have dense segmentations for all frames, we provide a script to create dense segmentations using SAM2. Yet, those are sadly not perfect and should be manually corrected afterwards, which can be done using the X-Anylabeling tool, as described in the next section. To create dense segmentations, run the following command:

```bash
nagwp segmentation --base-path data/waymo/processed/[my_segment_folder]
```

This will create dense segmentations for all frames within the `generated_masks` folder inside the base_path. The script uses the initial masks and unmatched boxes to create dense segmentations using SAM2. Further it will create *overlapping group masks* for each frame, stored in the `ov_masks` folder. We invented this format to visualize multiple non-overlapping masks in one frame while overlapping masks are within a seperate group and file. The naming convention of these files follows `img_[frame_index]_ov_[group_id].png`, where group_id refers to the overlapping group. These overlapping groups are global accross the whole sequence, so two masks ids which collide in at least one frame will be not be the same group, while two masks which never collide in any frame will be in the same group. This preserves information to refine all masks regardless of overlap and allows for easy visualization on the same time.
Further the "overlapping group format" may spread out the indices of the created masks within a frame, to increase visibilty, and allows to directly save group indices within the mask values, by storing relevant information in the exif data of the mask images. The exif data includes the actual mask ids as well as the original dtype of the masked image.

### Futher Usage of the Created Masks

In case you bring the masks yourself, or want to use a different tool, this information might be relevant for you:

In the [video_segmentation.py](../libs/nag-waymo-processing/nag_waymo_processing/video_segmentation.py) script, we use "index_mask_and_save" to index the `generated_masks` folder in the format:
```
generated_masks/
├── [MASK_ID]
    ├── [frame_index].png
    └── ...
├── ...
└── [MASK_ID]
    ├── [frame_index].png
    └── ...
```
to the overlapping group format. Such mask can easily be loaded using our [tools.segmentation.masking.load_mask](../tools/tools/segmentation/masking.py) function, which reads the exif data and returns the actual mask with the correct dtype and values. Further you might be interested in loading the full stack of masks for a frame, which can be done using the [tools.segmentation.masking.load_channel_masks](../tools/tools/segmentation/masking.py) function, which will return a tuple of masks (channel format) and their respective mask ids in numpy arrays of shape T x H x W x C and C, where T is the number of frames, H and W are the height and width of the masks, and C is the number of total masks. Further, saving is possible using the [tools.segmentation.masking.save_mask](../tools/tools/segmentation/masking.py) function, and [tools.segmentation.masking.save_channel_masks](../tools/tools/segmentation/masking.py) function, for single and stacked channel masks, respectively.

## 6. Refining Segmentations using X-Anylabeling

To refine the created segmentations, we used and modified slightly the X-Anylabeling tool, which can be used to create and alter masks, supports video segmentation, and can be used with various kinds of segmentation models. We modified the tool to streamline the workflow for our use case, so that it can directly be used to refine the created segmentations and can import them automatically.
Make sure you have set up the conda environment already then you can use our launch skript to start the Anylabeling tool with the correct parameters: 

```bash
nagwp labeling --base-path data/waymo/processed/[my_segment_folder]
```

If you are satisfied with the created segmentations and all boxes have dense and accurate segmentation masks, you can export the generated masks using a custom function in the Anylabeling tool, which will export the masks in our overlapping format to a `masks` folder inside the base_path. 
To do so choose "Export > OV Mask Annotations" - select the `color_mapping.json` in the respective base_path as colormap. Make sure that if you have introduced new label ids, they are convered in this color_mapping.json.