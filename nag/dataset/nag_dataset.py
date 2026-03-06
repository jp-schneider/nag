import os
from matplotlib.figure import Figure
import numpy as np


from nag.config.nag_config import NAGConfig
from nag.utils import utils
import torch
from torch.utils.data import Dataset
from tools.util.format import raise_on_none
from tools.io.image import load_image_stack, index_image_folder, load_image, load_image_stack_generator
from tools.segmentation.masking import load_channel_masks, index_value_masks_folder
from typing import Any, Dict, Generator, Optional, Tuple, Union
import pandas as pd
from nag.utils import utils
from tools.util.typing import DEFAULT
import psutil
from tools.util.torch import tensorify_image
from tools.logger.logging import logger
from tools.util.torch import index_of_first, tensorify
from tools.util.sized_generator import SizedGenerator, sized_generator
from tools.util.typing import VEC_TYPE
from tools.util.numpy import index_of_first as index_of_first_np
from tools.viz.matplotlib import plot_mask, saveable
from tools.util.torch import flatten_batch_dims, unflatten_batch_dims


class NAGDataset(Dataset):
    """Dataset for NAG scenes. Loads images, masks and depth images from disk."""

    _config: NAGConfig

    _index: pd.DataFrame

    _image_shape: Tuple[int, int, int, int]
    """Shape of the image stack. (T, H, W, C)"""

    _mask_path_ov_columns: Dict[int, str]

    cache_images: bool
    """If the image stack should be cached in memory."""

    cache_masks: bool
    """If the mask stack should be cached in memory."""

    _images: np.ndarray
    """Cached image stack as numpy array. Shape is (T, H, W, C) in uint8."""

    _frame_timestamps: Optional[torch.Tensor]
    """Frame timestamps in range [0, 1]."""

    _loaded_mask_ids: Optional[torch.Tensor]
    """Loaded mask ids."""

    _oids_mask: Optional[torch.Tensor]
    """Object Mask for the oids actually beeing used."""

    allow_nan_on_load: bool
    """If nan values are allowed on load."""

    def __init__(self, config: NAGConfig, frame_timestamps: Optional[torch.Tensor], allow_nan_on_load: bool = False):
        self._config = raise_on_none(config)
        self.allow_nan_on_load = allow_nan_on_load
        self._index, self._mask_path_ov_columns = self.setup_index()
        self._image_shape = self.get_image_shape()
        self._initial_image_shape = None
        self._learning_image_shape = None
        self._frame_timestamps = frame_timestamps
        self._images = None
        self._masks = None
        self._loaded_mask_ids = None
        self._oids_mask = None
        self._box_object_ids_mask_id_mapping = None
        self._mask_ids_filter = None
        self.cache_images = False
        self.cache_masks = False

        self.set_cache()

    @property
    def oids_mask(self) -> torch.Tensor:
        """Mask for the oids actually beeing used.

        Shape is (N,). for N objects.

        Returns
        -------
        torch.Tensor
            The masked tensor.
        """
        if self._oids_mask is None:
            self._oids_mask = torch.ones(
                len(self._loaded_mask_ids), dtype=torch.bool)
            if self.config.mask_indices_filter is not None:
                self._oids_mask[self.config.mask_indices_filter] = False
            # Check if filtered masks are present
            if len(self.mask_ids_filter) > 0:
                filtered_ids = torch.tensor(list(self.mask_ids_filter)).int()
                match = index_of_first(
                    self._loaded_mask_ids[:, -1], filtered_ids)
                ignore_ids = match != -1
                self._oids_mask[match[ignore_ids]] = False
        return self._oids_mask

    @property
    def mask_ids(self) -> torch.Tensor:
        """
        Mask ids of the loaded mask ids.
        Filtered if needed.

        Corresponds to the order in load_mask.

        Returns
        -------
        torch.Tensor
            The masked ids.
        """
        return torch.tensor(self._loaded_mask_ids[self.oids_mask.numpy()][:, -1]).int()

    def mask_id_to_oid(self, mask_id: Any) -> torch.Tensor:
        """
        Converts the mask ids to objects ordered ids as they are used in the runner and model.

        Parameters
        ----------
        mask_id : VEC_TYPE
            Mask id to convert.
            Shape can be any (...).

        Returns
        -------
        torch.Tensor
            The ordered object ids of the mask ids in the same shape as the input.
        """
        mask_id = tensorify(mask_id)
        mask_id, shp = flatten_batch_dims(mask_id, -1)
        return unflatten_batch_dims(index_of_first(self.mask_ids, mask_id), shp)

    def load_box_obj_mapping(self, path: str):
        import json
        with open(path, "r") as f:
            mapping = json.load(f)
        return {int(k): int(v) for k, v in mapping.items()}

    @property
    def box_object_id_mask_id_mapping(self) -> Dict[int, int]:
        """Mapping from box object id to mask id."""
        if self._box_object_ids_mask_id_mapping is None:
            self._box_object_ids_mask_id_mapping = dict()
            if self.config.boxes_object_id_mask_mapping_path is not None and os.path.exists(self.config.boxes_object_id_mask_mapping_path):
                self._box_object_ids_mask_id_mapping = self.load_box_obj_mapping(
                    self.config.boxes_object_id_mask_mapping_path)
                logger.info(
                    f"Loaded box object id to mask id mapping from {self.config.boxes_object_id_mask_mapping_path}.")
        return self._box_object_ids_mask_id_mapping

    @property
    def mask_ids_filter(self) -> set:
        """Filter for the mask ids."""
        if self._mask_ids_filter is None:
            if self.config.mask_ids_filter_path is not None and os.path.exists(self.config.mask_ids_filter_path):
                try:
                    import json
                    ret = []
                    with open(self.config.mask_ids_filter_path, "r") as f:
                        ret = json.load(f)
                    ids = [int(x) for x in ret]
                    self._mask_ids_filter = set(ids)
                except Exception as e:
                    logger.error(
                        f"Error while reading mask ids filter file {self.config.mask_ids_filter_path}: {e}")
                    self._mask_ids_filter = set()
            else:
                self._mask_ids_filter = set()
        return self._mask_ids_filter

    @property
    def frame_timestamps(self) -> torch.Tensor:
        if self._frame_timestamps is None:
            self._frame_timestamps = torch.linspace(
                0, 1, len(self), dtype=torch.float32)
        return self._frame_timestamps

    @property
    def image_shape(self) -> Tuple[int, int, int, int]:
        """Shape of the image stack. (T, H, W, C)"""
        if self._image_shape is None:
            self._image_shape = self.get_image_shape()
        return self._image_shape

    @property
    def initial_image_shape(self) -> Tuple[int, int]:
        """Initial image shape for the dataset. Shape is (H, W)."""
        if self._initial_image_shape is None:
            from tools.io.image import compute_new_size
            image_size = self.image_shape[1:3]
            if self.config.init_max_image_size is not None:
                image_size = compute_new_size(
                    image_size, self.config.init_max_image_size)

                learn_size = self.learning_image_shape
                # Check if image_size larger than learning size, then set to learning size
                if (torch.tensor(image_size) > torch.tensor(learn_size)).any():
                    image_size = learn_size

            self._initial_image_shape = image_size
        return self._initial_image_shape

    @property
    def learning_image_shape(self) -> Tuple[int, int]:
        """Learning image shape for the dataset. Shape is (H, W)."""
        if self._learning_image_shape is None:
            from tools.io.image import compute_new_size
            image_size = torch.tensor(self.image_shape[1:3])
            new_size = image_size
            if self.config.max_image_size is not None and any(self.config.max_image_size < image_size):
                new_size = torch.tensor(compute_new_size(
                    image_size, self.config.max_image_size))
                logger.warning(
                    f"Max image size {tuple(self.config.max_image_size)} is smaller than native (GT) image size {image_size} changing to {tuple(new_size.tolist())}."
                )

            if self.config.learn_resolution_factor < 1:
                downscaled_size = (
                    tensorify(new_size) * self.config.learn_resolution_factor).round().int()
                if (downscaled_size < new_size).any():
                    logger.warning(
                        f"Learn resolution factor {self.config.learn_resolution_factor} is smaller than 1, downscaling images...")
                    new_size = downscaled_size

            # If image_size is smaller than the new_size, warn the user
            if (new_size < tensorify(image_size)).any():
                logger.warning(
                    f"Learning image size {tuple(new_size.tolist())} is smaller than native (GT) image size {image_size}, this will limit the model-image quality.")
            image_size = tuple(new_size.tolist())
            self._learning_image_shape = image_size
        return self._learning_image_shape

    def times_to_indices(self, times: torch.Tensor) -> torch.Tensor:
        """Convert frame timestamps to indices.
        
        Parameters
        ---------
        times: torch.Tensor
            Converts times to 0-based indices of seen ground truth images.

        Returns
        -------
        torch.Tensor
            Frame indices in range [0, T-1] for T beeing the total number of times.
        """
        from tools.util.torch import index_of_first
        ts = self.frame_timestamps
        return index_of_first(ts, times)

    def times_to_frame_indices(self, times: torch.Tensor) -> torch.Tensor:
        """Convert frame timestamps to frame indices (actual image indices as specified e.g. in the filename)."""
        indices = self.times_to_indices(times)
        return self.indices_to_frame_indices(indices)

    def indices_to_times(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert indices to frame timestamps.

        Parameters
        ----------
        indices: torch.Tensor
            Frame indices to convert to timestamps. Shape is (N,). In range [0, T-1].

        Returns
        -------
        torch.Tensor
            Frame timestamps in range [0, 1]. Shape is (N,).
        """
        return self.frame_timestamps[indices]

    def indices_to_frame_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert indices to frame indices (actual image indices as specified e.g. in the filename).

        Parameters
        ----------
        indices: torch.Tensor
            Frame indices to convert to timestamps. Shape is (N,). In range [0, T-1].

        Returns
        -------
        torch.Tensor
            Frame indices in range [?, ?]. Depending on Data. Shape is (N,).
        """
        return torch.tensor(self._index["index"].values)[indices]

    def frame_indices_to_indices(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """Convert frame indices (actual image indices as specified e.g. in the filename) to indices.

        Parameters
        ----------
        frame_indices: torch.Tensor
            Frame indices to convert to timestamps. Shape is (N,). From Sequence based image idx.

        Returns
        -------
        torch.Tensor
            Frame indices in range [0, T-1]. Shape is (N,).
        """
        indices = torch.tensor(self._index["index"].values)
        return index_of_first(indices, frame_indices)

    def frame_indices_to_times(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """Convert frame indices (actual image indices as specified e.g. in the filename) to frame timestamps in range [0, 1].

        Will return -1 for indices that are not found in the dataset.

        Parameters
        ----------
        frame_indices: torch.Tensor
            Frame indices to convert to timestamps. Shape is (N,). In range [0, N_Idx].
            Or Min-Index to Max-Index of the frame indices.

        Returns
        -------
        torch.Tensor
            Frame timestamps in range [0, 1]. Shape is (N,).
            -1 for indices that are not found in the dataset.
        """
        indices = torch.tensor(self._index["index"].values)
        idx = index_of_first(indices, frame_indices)
        timestamps = torch.zeros_like(frame_indices, dtype=torch.float32)
        # Check for idx is not -1
        timestamps[idx != -1] = self.frame_timestamps[idx[idx != -1]]
        # Fill with -1
        timestamps[idx == -1] = -1
        return timestamps

    def set_cache(self):
        if self._config.cache_images == DEFAULT:
            # Calculate the size of the image stack
            # 3 channels, 4 bytes per float
            image_stack_size = np.prod(self._image_shape) * 3 * 4
            # Calculate the size of the image stack in percent of the available memory
            mem = psutil.virtual_memory()
            image_stack_size_percent = image_stack_size / mem.available
            if image_stack_size_percent < 0.1:
                self.cache_images = True
            else:
                self.cache_images = False
        else:
            self.cache_images = self._config.cache_images
        if self._config.cache_masks == DEFAULT:
            # Calculate the size of the mask stack
            # 3 channels, 4 bytes per float
            num_obj = len(
                self.oids_mask) if self._loaded_mask_ids is not None else 10

            image_stack_size = np.prod(self._image_shape) * 1 * 0.125 * num_obj
            # Calculate the size of the image stack in percent of the available memory
            mem = psutil.virtual_memory()
            image_stack_size_percent = image_stack_size / mem.available
            if image_stack_size_percent < 0.05:
                self.cache_masks = True
            else:
                self.cache_masks = False
        else:
            self.cache_masks = self._config.cache_masks

    def get_image_shape(self):
        # Load first image
        image = load_image_stack(
            None, None,
            sorted_image_paths=self._index["image_path"].values[:1])
        image_shape = image.shape
        # patch T by length of the image stack
        image_shape = (len(self._index),) + image_shape[1:]
        return image_shape

    def setup_index(self) -> Tuple[pd.DataFrame, Dict[int, str]]:
        from tools.util.format import consecutive_indices_string
        image_paths = index_image_folder(
            self._config.images_path, filename_format=self._config.images_filename_pattern, return_dict=True)
        mask_paths = index_value_masks_folder(
            self._config.masks_path, filename_pattern=self._config.masks_filename_pattern, return_dict=True)
        depth_paths = index_image_folder(
            self._config.depths_path, filename_format=self._config.depths_filename_pattern, return_dict=True)
        # Check if images, masks and depth have the same length and larger than 0
        if len(image_paths) == 0:
            raise ValueError(
                f"No images found with image path {self._config.images_path} and filename pattern {self._config.images_filename_pattern}")
        if len(mask_paths) == 0:
            raise ValueError(
                f"No masks found with mask path {self._config.masks_path} and filename pattern {self._config.masks_filename_pattern}")
        if len(depth_paths) == 0:
            raise ValueError(
                f"No depth images found with depth path {self._config.depths_path} and filename pattern {self._config.depths_filename_pattern}")

        # if len(image_paths) != len(mask_paths[0]) or len(image_paths) != len(depth_paths):
        #     if len(mask_paths[0]) < len(image_paths):
        #         logger.warning(
        #             f"Number of masks ({len(mask_paths[0])}) is smaller than number of images ({len(image_paths)}), will assume that there are no objects in frames where the mask is missing.")
        #     else:
        #         raise ValueError("Length of images, masks and depth images do not match found: images: {}, masks: {}, depth: {}".format(
        #             len(image_paths), len(mask_paths[0]), len(depth_paths)))

        frame = pd.DataFrame(image_paths)
        frame["image_path"] = frame["path"]
        frame.drop(columns=["path"], inplace=True)
        frame.set_index("index", inplace=True)

        mask_columns = dict()
        # Add masks
        # Get number of overlap indices
        overlap_idx = list(mask_paths.keys())
        for ov in overlap_idx:
            column_name = "mask_path_ov_" + str(ov)
            mpd = pd.DataFrame(mask_paths[ov]).set_index("index")
            if "ov_index" in mpd.columns:
                mpd.drop(columns=["ov_index"], inplace=True)
            mpd.rename(columns={"path": column_name}, inplace=True)
            frame = frame.join(mpd, how="outer", on="index")
            mask_columns[ov] = column_name

        # Add depth
        depth_df = pd.DataFrame(depth_paths).set_index("index")
        depth_df.rename(columns={"path": "depth_path"}, inplace=True)
        frame = frame.join(depth_df, how="outer", on="index")

        # Check if all images have a depth image
        if "index" in frame.columns:
            frame = frame.set_index("index", drop=True)

        # Sort on index
        frame = frame.sort_index(ascending=True)

        # Check if index is monotonic increasing
        idx = torch.tensor(frame.index.values).int()
        vals = (idx[1:] - idx[:-1]).unique()
        if len(vals) > 1 or vals[0] != 1:
            # Warning if not monotonic
            logger.warning(
                f"Frame indices are not monotonic increasing. Found {vals.tolist()} different step sizes. This might cause an issue with the dataset, take care!")

        # If frame filter is set, filter the dataframe
        if self.config.frame_indices_filter is not None:
            fidx = torch.tensor(frame.index).int()[
                self.config.frame_indices_filter]
            frame = frame[np.isin(frame.index.values, (fidx.numpy()))]

        # Check if frame contains NA values
        NA_cols = frame.isnull().values
        if NA_cols.any():
            column_contains_empty = NA_cols.any(axis=0)
            # Allow for empty masks, but not for empty images or depth images
            na_values = dict()
            for na_cols in frame.columns.values[column_contains_empty]:
                if na_cols.startswith("mask_path_ov"):
                    continue
                else:
                    na_values[na_cols] = consecutive_indices_string(
                        frame[frame[na_cols].isnull()].index.values)
            if len(na_values) > 0 and not self.allow_nan_on_load:
                raise ValueError(
                    f"Frame contains NA values in the following columns with given indices:{os.linesep} {os.linesep.join(['- ' + k +': '+ v for k, v in na_values.items()])}")

        frame.reset_index(inplace=True)
        return frame, mask_columns

    @property
    def config(self) -> NAGConfig:
        return self._config

    @property
    def bundle(self) -> Optional[Dict[str, Any]]:
        return utils.load_bundle(self._config.bundle_path) if self._config.bundle_path is not None else None

    def get_images(self) -> np.ndarray:
        if self._images is None:
            # Full size stack
            _images = self.load_full_image_stack(False, False)
            if self.cache_images:
                self._images = _images
            return _images
        return self._images

    def get_masks(self) -> np.ndarray:
        if self._masks is None:
            # Full size stack
            _masks = self.load_mask_stack(False)
            if self.cache_masks:
                self._masks = _masks
            return _masks
        return self._masks

    def load_full_image_stack(self,
                              init_size: bool = False,
                              progress_bar: bool = DEFAULT) -> np.ndarray:
        """Load the full image stack."""
        shape = self.initial_image_shape if init_size else self.learning_image_shape
        images = load_image_stack(
            self._config.images_path,
            sorted_image_paths=self._index["image_path"].values,
            size=shape,
            progress_bar=self._config.use_progress_bar if progress_bar == DEFAULT else progress_bar, progress_factory=self._config.progress_factory
        )
        return images

    def load_image_stack(self,
                         init_size: bool = True,
                         idx: Optional[Union[int, torch.Tensor]] = None,
                         native_size: bool = False,
                         progress_bar: bool = DEFAULT) -> np.ndarray:
        """Load the image stack

        Parameters
        ---

        """
        if idx is not None:
            if isinstance(idx, torch.Tensor):
                if len(idx.shape) == 0:
                    idx = idx.unsqueeze(0)
                idx = idx.cpu().numpy()
            if isinstance(idx, int):
                idx = np.array([idx])
        if not self.cache_images or init_size or native_size:
            if native_size:
                shape = None
            else:
                shape = self.initial_image_shape if init_size else self.learning_image_shape
            paths = None
            if idx is not None:
                paths = self._index["image_path"].values[idx]
                if isinstance(paths, str):
                    paths = [paths]
            else:
                paths = self._index["image_path"].values
            images = load_image_stack(
                self._config.images_path,
                sorted_image_paths=paths,
                size=shape,
                progress_bar=self._config.use_progress_bar if progress_bar == DEFAULT else progress_bar, progress_factory=self._config.progress_factory
            )
        else:
            images = self.get_images()
            if idx is not None:
                images = images[idx]
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255
        if images.shape[-1] == 4:
            # Remove alpha channel and set black background
            images = images[..., :3] * images[..., 3:4]
        return images

    def load_mask_stack_idx(self,
                            init_size: bool = True,
                            idx: Optional[Union[int, torch.Tensor]] = None,
                            progress_bar: bool = DEFAULT,
                            native_size: bool = False,
                            ) -> np.ndarray:
        """Load the mask stack.

        Parameters
        --------
        init_size: bool, optional
            If True, load the mask stack with the initial image shape. If False, load the mask stack with the learning image shape. Default is True.

        idx: Optional[Union[int, torch.Tensor]], optional
            Indices of the frames to load. If None, load all frames. Default is None.

        progress_bar: bool, optional
            If True, show a progress bar while loading the mask stack. Default is the setting within the config.

        native_size: bool, optional
            If True, load the mask stack with the native image shape. Default is False.

        Returns
        ------- 
        np.ndarray
            The loaded mask stack. Shape is (T, H, W, N) where T is the number of frames, H and W are the height and width of the masks and N is the number of objects.

        """
        if idx is not None:
            if isinstance(idx, torch.Tensor):
                if len(idx.shape) == 0:
                    idx = idx.unsqueeze(0)
                idx = idx.cpu().numpy()
            if isinstance(idx, int):
                idx = np.array([idx])
        if not self.cache_masks or init_size or native_size:
            shape = self.initial_image_shape if init_size else self.learning_image_shape
            if native_size:
                shape = None
            pths = dict()
            for ov, col in self._mask_path_ov_columns.items():
                if idx is not None:
                    item = self._index[col].values[idx]
                    if isinstance(item, str):
                        items = [item]
                    else:
                        items = item
                    pths[ov] = items
                else:
                    pths[ov] = self._index[col].values

            masks, ovidx = load_channel_masks(self._config.masks_path, overlapping_mask_paths=pths, size=shape,
                                              progress_bar=self._config.use_progress_bar if progress_bar == DEFAULT else progress_bar,
                                              progress_factory=self._config.progress_factory)
            if len(masks.shape) == 3:
                masks = masks[np.newaxis, ...]
            native_ovidx = ovidx.copy()
            mapping = self.box_object_id_mask_id_mapping
            inv_mapping = {v: k for k, v in mapping.items()}
            mapped_back_masked_ids = np.array(
                [inv_mapping.get(x, x) for x in ovidx[:, -1]])

            new_odis = np.zeros_like(ovidx)
            new_odis[:] = ovidx[:]
            new_odis[:, -1] = mapped_back_masked_ids
            ovidx = new_odis

            # Check that the order of masks matches the current order
            if self._loaded_mask_ids is not None:
                # If masks are missing as there are not available for the index, fill the dim
                if len(self._loaded_mask_ids[:, -1]) > len(ovidx[:, -1]):
                    missing = np.array(
                        list(set(self._loaded_mask_ids[:, -1]) - set(ovidx[:, -1])))
                    zmsk = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], len(
                        missing)), dtype=masks.dtype)
                    missing_oids = missing[:, None]
                    missing_oids = np.concatenate(
                        (np.zeros((len(missing), 1), dtype=np.int32), missing_oids), axis=1)
                    masks = np.concatenate((masks, zmsk), axis=-1)
                    ovidx = np.concatenate((ovidx, missing_oids), axis=0)
                # Argsort that ovidx is in the same order as the loaded mask ids
                order = index_of_first_np(
                    ovidx[:, -1], self._loaded_mask_ids[:, -1])
                # Check that length matches and all order is larger than -1
                if not (len(order) == len(ovidx) and (order >= 0).all()):
                    # Check if they are the same if set
                    if set(ovidx[:, -1]) == set(self._loaded_mask_ids[:, -1]):
                        logger.warning(
                            "Duplicates found in the loaded mask ids, this might cause issues.")
                    else:
                        raise ValueError(
                            "Order of masks does not match the loaded mask ids.")
                masks = masks[..., order]
                ovidx = ovidx[order]
            else:
                if (native_ovidx != ovidx).any():
                    logger.info(f"Loaded {len(ovidx)} masks with the following ids: {ovidx[:, -1]} \
                                which where eventually remapped based on a mapping:\n" + "\n".join([str(native_ovidx[i, -1]) + " -> " + str(ovidx[i, -1]) for i in range(len(ovidx))]))
                else:
                    logger.info(
                        f"Loaded {len(ovidx)} masks with the following ids: {ovidx[:, -1]}")
                self._loaded_mask_ids = ovidx
                self.check_plane_names()
        else:
            masks = self.get_masks()
            if idx is not None:
                masks = masks[idx]

        if masks.dtype != np.bool_:
            masks = masks.astype(np.bool_)
        return masks

    @sized_generator()
    def load_image_stack_generator(self,
                                   idx: Optional[Union[int,
                                                       torch.Tensor]] = None,
                                   size: Optional[VEC_TYPE] = DEFAULT,
                                   native_size: bool = False,
                                   progress_bar: bool = False
                                   ) -> Generator[np.ndarray, None, None]:
        """Load the image stack

        Parameters
        -------

        """
        if idx is not None:
            if isinstance(idx, torch.Tensor):
                if len(idx.shape) == 0:
                    idx = idx.unsqueeze(0)
                idx = idx.cpu().numpy()
            if isinstance(idx, int):
                idx = np.array([idx])
        if native_size:
            size = None
        else:
            if size == DEFAULT:
                size = self.learning_image_shape
        paths = None
        if idx is not None:
            paths = self._index["image_path"].values[idx]
        else:
            paths = self._index["image_path"].values
        image_gen = load_image_stack_generator(
            self._config.images_path,
            sorted_image_paths=paths,
            size=size,
            progress_bar=self._config.use_progress_bar if progress_bar == DEFAULT else progress_bar, progress_factory=self._config.progress_factory
        )
        yield len(image_gen)

        for images in image_gen:
            if images.dtype == np.uint8:
                images = images.astype(np.float32) / 255
            if images.shape[-1] == 4:
                # Remove alpha channel and set black background
                images = images[..., :3] * images[..., 3:4]
            yield images

    def check_plane_names(self):
        if self._config.plane_names is not None:
            order = []
            invalid = []
            if isinstance(self._config.plane_names, dict):
                for k, v in self._config.plane_names.items():
                    try:
                        if isinstance(k, str):
                            if k.isnumeric():
                                k = int(k)
                            else:
                                raise ValueError("Key is not numeric.")
                        idx = np.argwhere(self._loaded_mask_ids[:, -1] == k)
                        if len(idx) == 0:
                            raise KeyError(
                                f"Key not found in _loaded_mask_ids.")
                        order.append((idx.squeeze(), k, v))
                    except Exception as err:
                        invalid.append((k, err))
            # If invalid keys are found, create warning
            if len(invalid) > 0:
                msg = "Invalid plane names:" + \
                    ", ".join(
                        [f"{type(v).__name__}: {k} {str(v)}" for k, v in invalid])
                logger.warning(
                    "Invalid plane names are found, these are beeing substituded." + os.linesep + msg)
            # Check which ids are missing, insert these at the end of the list
            covered = set([v[1] for v in order])
            for i, num in enumerate(self._loaded_mask_ids[:, -1]):
                num = int(num)
                if num not in covered:
                    order.append((i, num, f"{i} ({num})"))
            # Sort the order by index
            order = sorted(order, key=lambda x: x[0])
            # Create new list
            _names = np.array([v[2] for v in order])
            # Filter by the objects actually beeing used
            _names = _names[self.oids_mask]
            self._config.plane_names = _names.tolist()

    def load_mask_stack(self, init_size: bool = True) -> np.ndarray:
        """Load the mask stack from the config."""
        pths = dict()
        shape = self.initial_image_shape if init_size else self.learning_image_shape
        for ov, col in self._mask_path_ov_columns.items():
            pths[ov] = self._index[col].values
        masks, ovidx = load_channel_masks(
            self._config.masks_path, overlapping_mask_paths=pths, size=shape)

        mapping = self.box_object_id_mask_id_mapping
        inv_mapping = {v: k for k, v in mapping.items()}
        mapped_back_masked_ids = np.array(
            [inv_mapping.get(x, x) for x in ovidx[:, -1]])

        new_odis = np.zeros_like(ovidx)
        new_odis[:] = ovidx[:]
        new_odis[:, -1] = mapped_back_masked_ids
        ovidx = new_odis

        if self._loaded_mask_ids is not None:
            # Argsort that ovidx is in the same order as the loaded mask ids
            order = index_of_first_np(
                self._loaded_mask_ids[:, -1], ovidx[:, -1])
            # Check that length matches and all order is larger than -1
            if not (len(order) == len(ovidx) and (order >= 0).all()):
                raise ValueError(
                    "Order of masks does not match the loaded mask ids.")
            masks = masks[..., order]
            ovidx = ovidx[order]
        else:
            self._loaded_mask_ids = ovidx

        self.check_plane_names()
        if len(masks.shape) == 3:
            masks = np.expand_dims(masks, axis=0)
        return masks

    def load_depth_stack(self, init_size: bool = True) -> np.ndarray:
        """Load the depth stack from the config."""
        shape = self.initial_image_shape if init_size else self.learning_image_shape
        depth = load_image_stack(
            self._config.depths_path, sorted_image_paths=self._index[
                "depth_path"].values, size=shape,
            progress_bar=self._config.use_progress_bar, progress_factory=self._config.progress_factory
        )
        if len(depth.shape) == 3:
            depth = np.expand_dims(depth, axis=-1)
        return depth

    def load_image(self,
                   idx: Union[int, torch.Tensor],
                   init_size: bool = False,
                   native_size: bool = False) -> torch.Tensor:
        img = self.load_image_stack(
            init_size=init_size, native_size=native_size, idx=idx, progress_bar=False)
        return tensorify_image(img)

    def load_mask(self,
                  idx: Union[int, torch.Tensor] = None,
                  init_size: bool = False,
                  native_size: bool = False) -> torch.Tensor:
        """Return the mask for the given index.

        Parameters
        ----------
        idx : Union[int, torch.Tensor]
            Index or indices of the mask to load.
            Shape is (N,) or scalar.
        init_size : bool, optional
            If its the init size or the learning size, by default False
        native_size : bool, optional
            If the native size should be loaded, by default False

        Returns
        -------
        torch.Tensor
            Channel mask tensor of shape (T, O, H, W)
            Returns only the used Object masks in the correct order.
        """
        msk = self.load_mask_stack_idx(
            init_size=init_size, idx=idx, progress_bar=self.config.use_progress_bar, native_size=native_size)
        msk = tensorify_image(msk)
        if msk.shape[0] > 0:
            return msk[:, self.oids_mask]
        return msk

    def load_mask_checked(self,
                          idx: Union[int, torch.Tensor] = None,
                          init_size: bool = False,
                          native_size: bool = False) -> torch.Tensor:
        masks = self.load_mask(idx, init_size, native_size=native_size)
        ids = self.mask_ids
        # Check if masks are present
        obj_size = masks.sum(dim=(0, 2, 3))
        zero_size = obj_size == 0

        idx = torch.arange(len(self.oids_mask))
        existing = idx[self.oids_mask]
        empty = existing[zero_size]

        existing_filter = torch.arange(len(existing))
        non_empty = existing_filter[~zero_size]

        if len(empty) > 0:
            logger.warning(
                f"Found empty masks for the following object ids: {self.mask_ids[zero_size].tolist()} ignoring these.")

        self._oids_mask[empty] = False
        msk = masks[:, non_empty]

        return msk

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> torch.Tensor:
        return self.load_image(idx)

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        # state.pop("_images", None)
        # state.pop("images", None)
        return state

    def __setstate__(self, state: object) -> None:
        self.__dict__.update(state)
        # self._images = None
        # return None

    # region Semantic correspondence

    def load_semantic_labels(self) -> Dict[int, str]:
        import json
        data_dir = self.config.data_path
        path = os.path.join(data_dir, "semantic_labels.json")
        if not os.path.exists(path):
            logger.warning(
                f"Semantic labels file {path} not found, no labels available.")
        try:
            with open(path, "r") as f:
                return {int(k): v for k, v in json.load(f).items()}
        except Exception as e:
            logger.warning(
                f"Error while loading semantic labels from {path}: {e}")
        return dict()

    def load_semantic_correspondences(self) -> Dict[int, int]:
        import json
        data_dir = self.config.data_path
        path = os.path.join(data_dir, "semantic_correspondence.json")
        if not os.path.exists(path):
            logger.warning(
                f"Semantic correspondence file {path} not found, no labels available.")
        try:
            with open(path, "r") as f:
                return {int(k): int(v) for k, v in json.load(f).items()}
        except Exception as e:
            logger.warning(
                f"Error while loading semantic labels from {path}: {e}")

    def get_semantic_mapping(self, only_used: bool = True) -> Dict[int, Optional[str]]:
        labels = self.load_semantic_labels()
        correspondences = self.load_semantic_correspondences()
        mapping = dict()
        if correspondences is None:
            return dict()
        if only_used:
            used_ids = self.mask_ids.tolist()
        else:
            used_ids = None
        corresp = {k: correspondences.get(
            k, None) for k in used_ids} if used_ids is not None else correspondences

        for oid, class_ in corresp.items():
            if class_ in labels:
                mapping[oid] = labels[class_]
            else:
                mapping[oid] = None
        return mapping

    @saveable()
    def plot_gt_item(self,
                     idx: int,
                     init_size: bool = True,
                     mask_ids: Optional[torch.Tensor] = None,
                     use_labels: bool = True,
                     **kwargs) -> Figure:
        """
        Plot the ground truth item.
        Plots the gt image with the inpainted masks and semantic labels if available.

        Parameters
        ----------
        idx : int
            Index of the item to plot.

        init_size : bool, optional
            If the image should be loaded in the initial size or the learning size, by default True.

        Returns
        -------
        Figure
            Figure with the image and masks.
        """
        mask_ids = tensorify(
            mask_ids) if mask_ids is not None else self.mask_ids
        all_mask_ids = self.mask_ids
        masks = self.load_mask(idx, init_size=init_size)[0]

        sel_idx = index_of_first(all_mask_ids, mask_ids)
        masks = masks[sel_idx]

        images = self.load_image(idx, init_size=init_size)[0]
        corresp = self.get_semantic_mapping(only_used=True)
        # remove none values
        corresp = {k: v for k, v in corresp.items() if v is not None}
        if use_labels:
            labels = [str(k) + ": " + corresp.get(k, "Undefined")
                      for k in mask_ids.tolist()]
        else:
            labels = None
        return plot_mask(images, masks, labels=labels, **kwargs)
