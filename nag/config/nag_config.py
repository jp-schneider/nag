from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional, Type, Union
from pathlib import Path
from tools.util.typing import DEFAULT, _DEFAULT
import torch
from tools.util.path_tools import process_path

from nag.config.encoding_config import EncodingConfig
from nag.config.network_config import NetworkConfig
from nag.config.intrinsic_camera_config import IntrinsicCameraConfig
from nag.config.pinhole_camera_config import PinholeCameraConfig
from nag.model.phase import Phase
from tools.mixin.env_config_mixin import EnvConfigMixin
from tools.config.experiment_output_config import ExperimentOutputConfig
import os
from datetime import datetime

def default_phases() -> List[Phase]:
    return [
                        Phase(
                            background_color_fadeout=DEFAULT,
                            is_color_alpha_learnable=False,
                            is_flow_learnable=False,
                            is_grad_scaler_enabled=False,
                            is_position_gradient_rescaling_enabled=True,
                            is_position_learnable=True,
                            is_view_dependence_learnable=False,
                            is_lr_scheduler_active=False,
                            length=5,
                            name="position_only",
                            plane_normal_rescaling_hook='nag.model.hooks.alpha_chain_scaling_hook',
                            plane_normal_rescaling_hook_kwargs=dict(
                                alpha_chain_scaling=True,
                                total_hit_scaling=True
                            ),
                            plane_position_rescaling_hook='nag.model.hooks.alpha_chain_scaling_hook',
                            plane_position_rescaling_hook_kwargs=dict(
                                alpha_chain_scaling=True,
                                total_hit_scaling=True
                            ),
                        ),
                        Phase(
                            background_color_fadeout=DEFAULT,
                            is_color_alpha_learnable=True,
                            is_flow_learnable=False,
                            is_grad_scaler_enabled=False,
                            is_position_gradient_rescaling_enabled=True,
                            is_position_learnable=True,
                            is_view_dependence_learnable=False,
                            is_lr_scheduler_active=False,
                            length=15,
                            name="position_color",
                            plane_normal_rescaling_hook='nag.model.hooks.alpha_chain_scaling_hook',
                            plane_normal_rescaling_hook_kwargs=dict(
                                alpha_chain_scaling=False,
                                total_hit_scaling=True
                            ),
                            plane_position_rescaling_hook='nag.model.hooks.alpha_chain_scaling_hook',
                            plane_position_rescaling_hook_kwargs=dict(
                                alpha_chain_scaling=False,
                                total_hit_scaling=True
                            ),
                        ),
                        Phase(
                            background_color_fadeout=DEFAULT,
                            is_color_alpha_learnable=True,
                            is_flow_learnable=True,
                            is_grad_scaler_enabled=False,
                            is_position_gradient_rescaling_enabled=True,
                            is_position_learnable=True,
                            is_view_dependence_learnable=True,
                            is_lr_scheduler_active=True,
                            length=-1,
                            name="position_color_flow_view",
                            plane_normal_rescaling_hook='nag.model.hooks.alpha_chain_scaling_hook',
                            plane_normal_rescaling_hook_kwargs=dict(
                                alpha_chain_scaling=False,
                                total_hit_scaling=True
                            ),
                            plane_position_rescaling_hook='nag.model.hooks.alpha_chain_scaling_hook',
                            plane_position_rescaling_hook_kwargs=dict(
                                alpha_chain_scaling=False,
                                total_hit_scaling=True
                            ),
                        ),
                    ]

def default_loss_kwargs() -> Dict[str, Any]:
    return dict(
        mask_loss_weight=0.005,
    )

def default_plane_kwargs(size: str = "large") -> Dict[str, Any]:
    return dict(
            encoding_view_dependence_config=size,
            network_view_dependence_config=size,
        )

def default_plane_init_strategy_kwargs() -> Dict[str, Any]:
    return dict(
            alpha_mask_resolution=DEFAULT,
            color_mask_resolution=DEFAULT,
            orientation_smoothing=False,
            plane_position_strategy='nag.strategy.border_condition_plane_position_strategy.BorderConditionPlanePositionStrategy',
            temporal_alpha_consistency=False,
            temporal_color_consistency=False,
            translation_smoothing=False
        )

def default_sampler_kwargs(ray_ratio: float = 1.) -> Dict[str, Any]:
    return dict(
                num_batches=int(round(140 * (1 / ray_ratio))),
                num_rays=int(round(100000 * ray_ratio)),
                num_timestamps=20
            )

def default_background_plane_kwargs(size: str = "large") -> Dict[str, Any]:
    return dict(
        color_mask_resolution=DEFAULT,
        color_mask_smoothing=False,
        coarse_to_fine_color=False,
        encoding_image_config=size,
        encoding_flow_config=size,
        network_image_config=size,
        network_flow_config=size,
        encoding_view_dependence_config=size,
        network_view_dependence_config=size,
    )

@dataclass
class NAGConfig(ExperimentOutputConfig, EnvConfigMixin):
    """Configuration for the NAGModel."""

    # region Paths and stuff

    data_path: Union[str, os.PathLike, Path] = field(default=None)
    """Base path for the data. Must be set. Placeholder evaluation is supported."""

    images_path: Union[str, os.PathLike, Path] = field(default="{data_path}/images")
    """Path to the folder containing images. Placeholder evaluation is supported."""

    masks_path: Union[str, os.PathLike, Path] = field(default="{data_path}/masks")
    """Path to the folder containing masks. Placeholder evaluation is supported."""

    depths_path: Union[str, os.PathLike, Path] = field(default="{data_path}/depth")
    """Path to the folder containing depth images. Placeholder evaluation is supported."""

    boxes_path: Optional[Union[str, os.PathLike, Path]] = field(default="{data_path}/boxes")
    """Path to the folder containing boxes. Placeholder evaluation is supported."""

    boxes_object_id_mask_mapping_path: Optional[Union[str, os.PathLike, Path]] = field(default=None)
    """Path to the object id mask mapping.
    Should be a json file with the following structure:
    {
        "object_id": "mask_id"
        ...
    }
    Placeholder evaluation is supported."""

    images_filename_pattern: str = field(default=r"(?P<index>[0-9]+)\.((png)|(jpg)|(jpeg))")
    """Pattern for the image filenames. Used for finding and sorting the images in the image folder."""

    masks_filename_pattern: str = field(default=r"img_(?P<index>[0-9]+)_ov_(?P<ov_index>[0-9]+).png")
    """Pattern for the mask filenames. Used for finding and sorting the masks in the mask folder."""

    depths_filename_pattern: str = field(default=r"(?P<index>[0-9]+).tiff")
    """Pattern for the depth filenames. Used for finding and sorting the depth images in the depth folder."""

    boxes_filename_pattern: str = field(default=r"(?P<index>[0-9]+).json")
    """Pattern for the box filenames. Used for finding and sorting the boxes in the box folder."""

    pinhole_camera_config: Optional[Union[str, PinholeCameraConfig]] = field(default=None)
    """Configuration for the pinhole camera. Only used if no bundle is provided. If None, the camera will be set to a default value."""

    use_distance_quantiles: bool = field(default=False)
    """If the distance quantiles in the pinhole camera should be used. If True, the depth values will be upscaled."""

    normalize_camera: bool = field(default=False)
    """If the camera should be normalized. If True, the camera will be normalized to point z forward on the first frame."""

    synthetic_camera_config: Optional[Union[dict, IntrinsicCameraConfig]] = field(default=None)
    """Configuration for the synthetic camera. Only used if no bundle is provided. If None, the camera will be set to a default value."""

    # endregion

    # region Data

    max_image_size: Optional[int] = field(default=2040)
    """Maximum image size during training. If set, images will be resized so the max of its spatial demension will be limited to this size. By default 2040."""

    cache_images: Union[_DEFAULT, bool] = field(default=DEFAULT)
    """If the image stack should be cached in memory. If DEFAULT, the value will be calculated so, that the image stack should consume max 10 percent of the available memory."""

    cache_masks: Union[_DEFAULT, bool] = field(default=DEFAULT)
    """If the mask stack should be cached in memory. If DEFAULT, the value will be calculated so, that the mask stack should consume max 10 percent of the available memory."""

    # endregion

    # region Model

    model_type: Union[str, Type] = field(default="nag.model.nag_functional_model.NAGFunctionalModel")
    """Type of the model. Can be a type or a string which is a valid path to a class."""

    mask_indices_filter: Optional[Union[int, List[int]]] = field(default=None)
    """Filter for the mask indices. If set, only the masks with the given index will be used. If None, all masks will be used."""

    mask_ids_filter_path: Optional[Union[str, os.PathLike, Path]] = field(default="{data_path}/mask_indices_filter.json")
    """Ignore mask indices. Should be a json file with the following structure: [mask_id1, mask_id2, ...] Specifies the mask ids to be ignored."""

    frame_indices_filter: Optional[Union[int,
                                         List[int], slice]] = field(default=None)
    """If set, filters the used frames for the training. Can be anything with which a tensor can be indexed. None will use all frames. Default None."""

    relative_plane_margin: float = field(default=0.8)
    """Relative margin for the planes which will extend the ray hittable area of each plane in addition to the masks predicted one."""

    camera_translation_offset_weight: float = field(default=0.5)
    """Weight for the translation offset of the camera. Is the factor for defining the importance of the learned translation offset."""

    camera_rotation_offset_weight: float = field(default=0.5)
    """Weight for the rotation offset of the camera. Is the factor for defining the importance of the learned rotation offset."""

    object_rigid_control_points: Optional[int] = field(default=None)
    """The number of rigid control points for the all objects - camera + planes - discrete points in time where the planes can learn their position for. None means the number of frames // 2."""

    object_rigid_control_points_ratio: float = field(default=1.)
    """Ratio of the control points for the rigid object w.r.t the scene length. Only used if control points are not specified directly. Default 0.5"""

    object_position_spline_approximation: bool = field(default=True)
    """If the object position should be approximated by a spline which has the same control points as the learned one. Default True."""

    plane_names: Optional[List[str]] = field(default=None)
    """Names of the planes should be in the same order as the masks."""

    plane_flow_control_points: Optional[int] = field(default=None)
    """The number of flow control points for the planes - discrete points where the planes can learn their position for. None means using ratio."""

    plane_flow_control_points_ratio: Optional[int] = field(default=0.1)
    """Ratio of the control points for the flow w.r.t the scene length. Default 0.1"""

    plane_view_dependent_control_point_ratio: float = field(default=0.1)
    """Ratio of the control points for the view dependency w.r.t the scene length. Only active if a spline modelled view dependency is used. Default 0.25"""

    plane_translation_offset_weight: float = field(default=0.5)
    """Weight for the translation offset of the planes. Is the factor for defining the importance of the learned translation offset."""

    plane_rotation_offset_weight: float = field(default=0.5)
    """Weight for the rotation offset of the planes. Is the factor for defining the importance of the learned rotation offset."""

    plane_flow_weight: float = field(default=0.1)
    """Weight for the flow of the planes. Is the factor for defining the importance of the learned flow."""

    plane_color_weight: float = field(default=0.1)
    """Weight for the color of the planes. Is the factor for defining the importance of the learned color w.r.t the planes base color."""

    plane_alpha_weight: float = field(default=0.1)
    """Weight for the alpha of the planes. Is the factor for defining the importance of the learned alpha w.r.t the planes base alpha."""

    plane_init_strategy: Union[str, Type] = field(default="nag.strategy.tilted_plane_initialization_strategy.TiltedPlaneInitializationStrategy")
    """Initialization strategy for the planes. Can be a type or a string which is a valid path to a class."""

    plane_init_strategy_kwargs: Optional[Dict[str, Any]] = field(default_factory=default_plane_init_strategy_kwargs)
    """Keyword arguments for the plane initialization strategy constructor."""

    use_dataset_color_reprojection: bool = field(default=True)
    """If True, uses the reprojection method based on full resultion datasamples loaded from disk."""

    plane_rgb_rescaling: bool = field(default=True)
    """Rescaling of RGB initial prediction of networks. Default True."""

    plane_alpha_rescaling: bool = field(default=True)
    """Rescaling of alpha initial prediction of networks. Default True."""

    plane_flow_rescaling: bool = field(default=True)
    """Rescaling of flow initial prediction of networks. Default True."""

    plane_align_corners: bool = field(default=True)
    """If the gridsample of the initial color should be using align_corners. Default True."""

    plane_kwargs: Dict[str, Any] = field(default_factory=default_plane_kwargs)
    """Keyword arguments for the plane constructor."""

    plane_position_gradient_rescaling: bool = field(default=True)
    """If the position gradient of the planes should be rescaled."""

    plane_position_rescaling_hook: Union[str, Type, callable] = field(
        default="nag.model.hooks.alpha_chain_scaling_hook")

    plane_position_rescaling_hook_kwargs: Dict[str, Any] = field(
        default_factory=dict)

    plane_normal_rescaling_hook: Union[str, Type, callable] = field(
        default="nag.model.hooks.alpha_chain_scaling_hook")
    plane_normal_rescaling_hook_kwargs: Dict[str, Any] = field(
        default_factory=dict)

    is_plane_translation_learnable: bool = field(default=True)
    """If the translation of the planes should be learnable."""

    is_plane_rotation_learnable: bool = field(default=True)
    """If the rotation of the planes should be learnable."""

    is_camera_translation_learnable: bool = field(default=True)
    """If the translation of the camera should be learnable."""

    is_camera_rotation_learnable: bool = field(default=True)
    """If the rotation of the camera should be learnable."""

    has_background_plane: bool = field(default=True)
    """If a background plane should be added to the scene. Has a fixed color and its position is fixed to the camera."""

    has_camera_aberration_plane: bool = field(default=False)
    """If a lens aberration plane should be added to the scene. Has a learned color and its position is fixed to the camera."""

    background_plane_type: Union[str, Type, _DEFAULT] = field(default="nag.model.view_dependent_background_image_plane_scene_node_3d.ViewDependentBackgroundImagePlaneSceneNode3D")
    """Type of the background plane. Can be a type or a string which is a valid path to a class.
    If DEFAULT, calculates wether the masks cover the whole image () and sets the background plane to a fixed color if so,
    if not it will be the BackgroundImagePlaneSceneNode3D."""

    background_plane_kwargs: Dict[str, Any] = field(default_factory=default_background_plane_kwargs)
    """Keyword arguments for the background plane constructor."""

    mask_background_coverage_threshold: float = field(default=0.9)
    """Threshold for the mask coverage of the image. If the masks cover more than this threshold, the background plane will be a fixed color if background_plane_type is default."""

    background_camera_distance: float = field(default=1.5)
    """Distance of the background plane to the camera."""

    background_relative_scale_margin: float = field(default=0.1)
    """Relative margin for the background plane which will extend the ray hittable area of the background plane in addition to the camera image."""

    is_background_color_learnable: bool = field(default=True)
    """If the background color should be learnable."""

    learn_resolution_factor: float = field(default=1)
    """A factor to adjust the learn resolution of the model. Values < 1 will result in learning from a subsampled camera.
    While > 1 will superresolute the object. Factor will be applied to the resolution of the camera.
    """

    init_max_image_size: Optional[int] = field(default=1080)
    """Maximum size of the images and masks for the initialization.
    Will be used to resize the images for the initialization. If none, resizes everything to the image size."""

    plane_encoding_image_config: Union[str, EncodingConfig] = field(default="large")
    """Encoding config for the image encoding of the planes. Can be a string interpreted as present name or an EncodingConfig object."""

    plane_encoding_alpha_config: Union[str, EncodingConfig] = field(default="large")
    """Encoding config for the alpha encoding of the planes. Can be a string interpreted as present name or an EncodingConfig object."""

    plane_encoding_flow_config: Union[str, EncodingConfig] = field(default="large")
    """Encoding config for the flow encoding of the planes. Can be a string interpreted as present name or an EncodingConfig object."""

    plane_network_image_config: Union[str, NetworkConfig] = field(default="large")
    """Network config for the image network of the planes. Can be a string interpreted as present name or an NetworkConfig object."""

    plane_network_alpha_config: Union[str, NetworkConfig] = field(default="large")
    """Network config for the alpha network of the planes. Can be a string interpreted as present name or an NetworkConfig object."""

    plane_network_flow_config: Union[str, NetworkConfig] = field(default="large")
    """Network config for the flow network of the planes. Can be a string interpreted as present name or an NetworkConfig object."""

    plane_position_sampling: bool = field(default=False)
    """If the plane position or orientation should be sampled within the forward with the sampling fnc. Default False"""

    plane_position_sampling_hook: Optional[Union[str, Type, callable]] = field(
        default="nag.model.hooks.sample_positions_hook")
    """Function to sample the plane position or orientation. Default None"""

    plane_position_sampling_hook_kwargs: Dict[str, Any] = field(
        default_factory=dict)
    """Keyword arguments for the plane position sampling hook. Will induced in a partial function call."""

    plane_coarse_to_fine_color: bool = field(default=False)
    """If the color of the planes should be learned in a coarse to fine manner. Default True."""

    plane_coarse_to_fine_alpha: bool = field(default=False)
    """If the alpha of the planes should be learned in a coarse to fine manner. Default True."""

    plane_type: Union[str, Type] = field(default="nag.model.view_dependent_image_plane_scene_node_3d.ViewDependentImagePlaneSceneNode3D")
    """Type of the plane. Can be a type or a string which is a valid path to a class."""

    sampler_type: Union[str, Type] = field(default="nag.sampling.random_timed_uv_grid_sampler.RandomTimedUVGridSampler")
    """Type of the ray sampler. Ray Sampler controls how the rays are sampled during training."""

    sampler_kwargs: Dict[str, Any] = field(default_factory=default_sampler_kwargs)
    """Arguments for creating the sampler."""

    loss_type: Union[str, Type] = field(default="nag.loss.l1_mask_loss.L1MaskLoss")
    """Type of the loss. Can be a type or a string which is a valid path to a class."""

    loss_kwargs: Dict[str, Any] = field(default_factory=default_loss_kwargs)
    """Keyword arguments for the loss constructor."""

    checkpoint_path: Optional[Union[str, os.PathLike, Path]] = field(
        default=None)
    """Path to the checkpoint file. If None, no checkpoint will be loaded. Placeholder evaluation is supported."""

    checkpoint_tracker_path: Optional[Union[str,
                                            os.PathLike, Path]] = field(default=None)
    """Path to the checkpoint tracker directory. If None, no checkpoint will be loaded. Placeholder evaluation is supported."""

    save_after_phase_change: bool = field(default=True)
    """If the model should be saved after a phase change. Default True."""

    # endregion

    # region Training

    detect_anomaly: bool = field(default=False)
    """If anomaly detection should be enabled, just for debugging. Default is False."""

    log_gradient: bool = field(default=False)
    """If the gradient should be logged. Default is True."""

    deterministic: Union[bool, str] = field(default=False)
    """If the training should be deterministic. If True, use deterministic algorithms and raise if not possible. Set to ''warn' to use deterministic algorithms whenever possible and warn if not."""

    jit_enabled: bool = field(default=True)
    """If just in time compilation should be enabled. Will speed up the training significantly. Default is True."""

    max_epochs: int = field(default=80)
    """
    Number of training epochs.
    """

    coarse_to_fine_start_epoch: int = field(default=0)
    """Start epoch for the coarse to fine training. From the given epoch on, coarse to fine optimisation will unlock more parts of the encoding. Default 0."""

    coarse_to_fine_end_epoch: Optional[int] = field(default=65)
    """End epoch for the coarse to fine training. From the given epoch on, all masking will lose its effect. If None, max_epochs will be the end epoch."""

    epoch_data_ratio: float = field(default=1.0)
    """The ratio of the number of epochs to the number of data points. Describes how "often" (statistically) a datapoint is seen per epoch. Will be used to calculate the number of batches per epoch."""

    training_phases: Optional[List[Union[dict, Phase]]] = field(default_factory=default_phases)
    """Training phases for the model."""

    num_batches: Optional[int] = field(default=None)
    """The number of batches per epoch. As per epoch every pixel is sampled, increasing the number of batches will decrease the number of samples / rays per batch."""

    max_total_batch_size: Optional[int] = field(default=2.0e7)
    """The maximum total batch size for training. As batch operations are related to the number of object, time and XY-resolution, this will be a product of these values."""

    max_total_batch_size_inference: Optional[int] = field(default=4.0e7)
    """The maximum total batch size for inference. As batch operations are related to the number of object, time and XY-resolution, this will be a product of these values."""

    num_objects: Optional[int] = field(default=None)
    """Number of objects to use. Will be filled automatically when loading masks. If filled before, the number of objects will be limited to this value."""

    dtype: Union[str, torch.dtype] = field(default=torch.float32)
    """Precision of the model."""

    tinycudann_network_dtype: Union[str, torch.dtype] = field(default=torch.float32)
    """Precision of the network for the tiny-cuda-nn. Should corresond to the precision tinycuda was build with. Highly recommend to keep it at float32 and have a 32 bit build of tinycuda."""

    lr: float = field(default=1e-3)
    """
    Learning rate.
    """

    num_workers: int = field(default=1)
    """
    Number of dataloader workers. Used for PyTorch dataloader.
    """

    accelerator: Optional[str] = field(default="gpu")
    """
    Hardware accelerator. Used for PyTorch Lightning Trainer.
    """

    accelerator_devices: Union[str, int] = field(default=1)
    """
    Hardware accelerator devices. Used for PyTorch Lightning Trainer.
    """

    # endregion

    # region In Training Plotting and monitoring

    fast: bool = field(default=False)
    """If the training should be fast. If True, all logging and saving of intermediates will be skipped except for the first and last epoch."""

    debug: bool = field(default=False)
    """If the debug mode should be enabled. If True, will set the fast_dev_run flag of the Trainer to True."""

    in_training_checkpoint_interval: Optional[int] = field(default=None)
    """Describes when it will save a checkpoint. current_epoch modulo in_training_checkpoint_interval == 0"""

    in_training_save_geometry_interval: int = field(default=1)
    """Describes when it will save the geometry of nodes in the scene. current_epoch modulo in_training_save_geometry_interval == 0"""

    in_training_plot_geometry_interval: int = field(default=5)
    """Describes when it will generate plots to show the geometry of the scene. current_epoch modulo in_training_plot_geometry_interval == 0"""

    in_training_plot_geometry_object_indices: Optional[List[Union[List[int], int]]] = field(
        default=None)
    """Describes which objects should be plotted. If None, all objects will be plotted. If a list of lists, each list will be plotted in a separate plot."""

    in_training_plot_geometry_camera: bool = field(default=True)
    """If the camera should be plotted in the geometry plot."""

    in_training_epoch_plot_interval: int = field(default=5)
    """Describes when it will generate plots to show training progress. current_epoch modulo in_training_epoch_plot_interval == 0"""

    in_training_epoch_plot_log: bool = field(default=True)
    """If the in training plots should be logged to tensorboard."""

    in_training_epoch_plot_log_l1: bool = field(default=True)
    """If the L1 loss training plots should be logged to tensorboard."""

    in_training_plot_resolution_factor: float = field(default=1 / 4)
    """Resolution factor for the in training plots. The image will be scaled by this factor."""

    final_plot_resolution_factor: float = field(default=1)
    """Resolution factor for the final plots. The image will be scaled by this factor."""

    final_plot_without_view_dependency: bool = field(default=True)
    """If there should be additional plots without view dependency (if view dependency is activated). Default True."""

    final_diff_plot: bool = field(default=True)
    """If the difference between the final model and the GT images should be plotted and saved. Default True."""

    final_modalities_save: bool = field(default=True)
    """If after training, the per object color alpha and flow modalities should be saved. Default True"""

    final_per_object_save: bool = field(default=True)
    """If after training, each object should be rendered individuallyand saved. Default True"""

    initial_plot: bool = field(default=True)
    """If the the model init should be plotted default True."""

    initial_diff_plot: bool = field(default=True)
    """If the difference between the model init and the GT images should be plotted and saved. Default True."""

    initial_plot_resolution_factor: float = field(default=1 / 4)
    """Resolution factor for the initial plots and video. The image will be scaled by this factor."""

    initial_plot_video: bool = field(default=False)
    """If the initial plot should be a video."""

    in_training_plot_on_first_epoch: bool = field(default=True)
    """If the in training plots should be generated on the first / 0. epoch."""

    in_training_plot_times: List[float] = field(
        default_factory=lambda: [0.0, 0.5, 1.0])
    """Times for the in training plots. Will be used to generate the in training plots at these times."""

    in_training_save_plots_and_images: bool = field(default=True)
    """If the in training plots and images should be saved to the output folder."""

    log_every_n_steps: int = field(default=1)
    """Log every n steps for the lightning logger ? trainer."""

    final_evaluation_metrics: List[Union[str, Type]] = field(
        default_factory=lambda:
        ["tools.metric.torch.psnr.PSNR",
         "nag.loss.ssim.SSIM",
         ("tools.metric.torch.lpips.LPIPS", dict(net_type="alex", normalize=True))])
    """Metrics to evaluate the final model with. Can be a type or a string which is a valid path to a class."""

    final_mask_evaluation_metrics: List[Union[str, Type]] = field(
        default_factory=lambda:
        ["tools.metric.torch.psnr.PSNR",
         "nag.loss.ssim.SSIM", ])
    """Metrics to evaluate on a per-object basis with a mask."""

    save_video: bool = field(default=True)
    """If all outputs in the end should be used to create a video."""

    lr_scheduler: bool = field(default=True)
    """If a learning rate scheduler should be used."""

    lr_scheduler_cooldown: int = field(default=4)
    """Cooldown for the learning rate scheduler in epochs."""

    reevaluate_output_path: bool = field(default=False)
    """If the output path should be reevaluated on config preperation."""

    skip_object_on_missing_box: bool = field(default=False)
    """If an object should be skipped if no box is available."""

    project: str = field(default="nag")
    """Project name for logging in wandb."""

    # endregion

    # region Deprecation Flags for compatibility, do not change them unless you know what you are doing!

    deprecated_flow: bool = field(default=False)
    """Legacy flow handeling. Do not change!."""

    bundle_path: Optional[Union[str, os.PathLike, Path]] = field(default=None)
    """Path to frame_bundle.npz. Placeholder evaluation is supported. Legacy field do not change."""

    _old_scene_children_order: bool = field(default=False)
    """Old world scene children order. This is compatibility flag to load older version do not change."""

    # endregion

    def prepare(self, embed_files: bool = True, create_output_path: bool = True) -> None:
        from tools.logger.logging import logger
        from tools.metric.torch.reducible import Metric
        from tools.util.format import parse_type
        super().prepare(create_output_path=create_output_path,
                        reevaluate_output_path=self.reevaluate_output_path)

        # Interpolate paths
        self.data_path = process_path(
            self.data_path, need_exist=True, interpolate=True, interpolate_object=self, variable_name="data_path")

        self.bundle_path = process_path(self.bundle_path, need_exist=False, allow_none=True,
                                        interpolate=True, interpolate_object=self, variable_name="bundle_path")
        self.images_path = process_path(self.images_path, need_exist=True,
                                        interpolate=True, interpolate_object=self, variable_name="images_path")
        self.masks_path = process_path(
            self.masks_path, need_exist=True, interpolate=True, interpolate_object=self, variable_name="masks_path")
        self.depths_path = process_path(self.depths_path, need_exist=True,
                                        interpolate=True, interpolate_object=self, variable_name="depths_path")
        self.boxes_path = process_path(self.boxes_path, need_exist=False, allow_none=True,
                                       interpolate=True, interpolate_object=self, variable_name="boxes_path")

        self.mask_ids_filter_path = process_path(self.mask_ids_filter_path, need_exist=False, allow_none=True,
                                                 interpolate=True, interpolate_object=self, variable_name="mask_ids_filter_path")

        if self.mask_ids_filter_path is not None:
            if not os.path.exists(self.mask_ids_filter_path):
                logger.warning(
                    "Mask ids filter path does not exist. Will be ignored.")
                self.mask_ids_filter_path = None

        self.boxes_object_id_mask_mapping_path = process_path(self.boxes_object_id_mask_mapping_path, need_exist=False, allow_none=True,
                                                              interpolate=True, interpolate_object=self, variable_name="boxes_object_id_mask_mapping_path")

        if isinstance(self.dtype, str):
            self.dtype = torch.dtype(self.dtype)

        self.loss_type = parse_type(
            self.loss_type, Metric, variable_name="loss_type")

        if self.synthetic_camera_config is not None:
            if isinstance(self.synthetic_camera_config, dict):
                self.synthetic_camera_config = IntrinsicCameraConfig.from_object_dict(
                    self.synthetic_camera_config, force_cls=True)
        if self.pinhole_camera_config is not None:
            if isinstance(self.pinhole_camera_config, str):
                self.pinhole_camera_config = process_path(
                    self.pinhole_camera_config, need_exist=True, interpolate=True, interpolate_object=self, variable_name="pinhole_camera_config")
                if embed_files:
                    self.pinhole_camera_config = PinholeCameraConfig.load_from_file(
                        self.pinhole_camera_config)

            elif not isinstance(self.pinhole_camera_config, PinholeCameraConfig):
                logger.warning(
                    "Pinhole camera config is not a PinholeCameraConfig object. Will be ignored.")
                self.pinhole_camera_config = None

        if self.final_plot_resolution_factor is not None and self.learn_resolution_factor is not None:
            if self.final_plot_resolution_factor > self.learn_resolution_factor:
                logger.warning(
                    "Final plot resolution factor is greater than the learn resolution factor. Resetting it to: {}".format(self.learn_resolution_factor))
        if self.in_training_plot_resolution_factor is not None and self.learn_resolution_factor is not None:
            if self.in_training_plot_resolution_factor > self.learn_resolution_factor:
                logger.warning(
                    "In training plot resolution factor is greater than the learn resolution factor. Resetting it to: {}".format(self.learn_resolution_factor))
        if self.initial_plot_resolution_factor is not None and self.learn_resolution_factor is not None:
            if self.initial_plot_resolution_factor > self.learn_resolution_factor:
                logger.warning(
                    "Initial plot resolution factor is greater than the learn resolution factor. Resetting it to: {}".format(self.learn_resolution_factor))

        if self.checkpoint_path is not None:
            self.checkpoint_path = process_path(self.checkpoint_path, need_exist=True,
                                                interpolate=True, interpolate_object=self, variable_name="checkpoint_path")
            if self.checkpoint_tracker_path is None:
                logger.warning(
                    "Checkpoint tracker path is None. This may lead to problems in logging having the wrong state.")
            else:
                self.checkpoint_tracker_path = process_path(self.checkpoint_tracker_path, need_exist=True,
                                                            interpolate=True, interpolate_object=self, variable_name="checkpoint_tracker_path")

        if self.experiment_datetime.astimezone() < datetime.fromisoformat("2025-04-29T00:00:00+01:00"):
            self._old_scene_children_order = True

    def allow_cli_config_overwrite(self, field, config_value, cli_value):
        if field.name == "name":
            if isinstance(cli_value, str) and cli_value.lower() in ["", "default", "example"]:
                return False
        return True

    @classmethod
    def argparser_ignore_fields(cls) -> List[str]:
        return [
            "is_waymo",
            "deprecated_flow",
            "_old_scene_children_order",
            "new_homography",
            "bundle_path",
        ]
