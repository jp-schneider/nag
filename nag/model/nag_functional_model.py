
from functools import partial
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Tuple, List, Union, Generator

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from nag.config.nag_config import NAGConfig
import torch
from nag.model.background_plane_scene_node_3d import BackgroundPlaneSceneNode3D
from nag.model.learned_camera_scene_node_3d import LearnedCameraSceneNode3D
from nag.model.nag_model import NAGModel, compute_object_rgba, get_object_intersection_points, undo_intersection_ordering
from tools.logger.logging import logger
from nag.model.discrete_plane_scene_node_3d import default_plane_scale, default_plane_scale_offset, plane_coordinates_to_local, batched_local_to_plane_coordinates, batched_plane_coordinates_to_local
from nag.model.timed_camera_scene_node_3d import get_global_rays, local_to_image_coordinates
from nag.model.timed_discrete_scene_node_3d import NON_EQUAL_TIME_STEPS_WARNED, get_translation, get_orientation, global_to_local, global_to_local_mat, local_to_global
from nag.model.learned_offset_scene_node_3d import default_offset_times
from tools.util.torch import tensorify
from nag.model.learned_offset_scene_node_3d import LearnedOffsetSceneNode3D, get_combined_translation_orientation

from nag.model.timed_plane_scene_node_3d import TimedPlaneSceneNode3D
from nag.sampling.regular_uv_grid_sampler import RegularUVGridSampler
from nag.transforms.transforms_timed_3d import _get_interpolate_index_and_distance
from tools.transforms.geometric.transforms3d import unitquat_to_rotmat
from tools.model.module_scene_node_3d import ModuleSceneNode3D
from nag.model.timed_discrete_scene_node_3d import TimedDiscreteSceneNode3D, compose_translation_orientation
from tools.viz.matplotlib import saveable
from tools.util.numpy import numpyify
from torch.utils.hooks import RemovableHandle
from tools.util.progress_factory import ProgressFactory

from tools.context.temporary_device import TemporaryDevice
from tools.context.temporary_training import TemporaryTraining
from tools.util.typing import VEC_TYPE, NUMERICAL_TYPE, DEFAULT, _DEFAULT
from tools.util.torch import batched_generator_exec, flatten_batch_dims, unflatten_batch_dims
from nag.model.learned_aberration_plane_scene_node_3d import LearnedAberrationPlaneSceneNode3D
from tools.util.sized_generator import SizedGenerator
from tools.transforms.geometric.quaternion import quat_subtraction
from tools.util.torch import TensorUtil


@torch.jit.script
def get_translation_orientation(
    t: torch.Tensor,
    translations: torch.Tensor,
    translation_offset: torch.Tensor,
    orientations: torch.Tensor,
    rotation_offset: torch.Tensor,
    translation_offset_weight: torch.Tensor,
    rotation_offset_weight: torch.Tensor,
    times: torch.Tensor,
    offset_times: torch.Tensor,
    equidistant_times: bool,
    equidistant_offset_times: bool,
    right_idx: torch.Tensor,
    rel_frac: torch.Tensor,
    right_idx_offset: torch.Tensor,
    rel_frac_offset: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    O = translations.shape[0] - 1
    T = len(t)
    N = O + 1  # Objects + Camera Without Background

    # When right_idx and rel_frac are provided, these are usually valid for the whole batch / number of objects so we need to repeat them
    ob_idx = torch.arange(N, device=translations.device, dtype=torch.int32)
    right_idx_o = right_idx.repeat(N, 1, 1)
    right_idx_o[:, :, 0] = ob_idx.unsqueeze(-1)
    right_idx_o = right_idx_o.reshape(N * T, 2)
    rel_frac_o = rel_frac.repeat(N, 1)
    right_idx_offset_o = right_idx_offset.repeat(N, 1, 1)
    right_idx_offset_o[:, :, 0] = ob_idx.unsqueeze(-1)
    right_idx_offset_o = right_idx_offset_o.reshape(N * T, 2)
    rel_frac_offset_o = rel_frac_offset.repeat(N, 1)

    translations, orientations = get_combined_translation_orientation(
        translation=translations,
        translation_offset=translation_offset,
        orientation=orientations,
        rotation_offset=rotation_offset,
        translation_offset_weight=translation_offset_weight.unsqueeze(
            -1).unsqueeze(-1).repeat(1, T, 3),
        rotation_offset_weight=rotation_offset_weight.unsqueeze(
            -1).unsqueeze(-1).repeat(1, len(offset_times), 3),
        times=times,
        offset_times=offset_times,
        steps=t,
        interpolation="cubic",
        equidistant_times=equidistant_times,
        equidistant_offset_times=equidistant_offset_times,
        right_idx=right_idx_o,
        rel_frac=rel_frac_o,
        right_idx_offset=right_idx_offset_o,
        rel_frac_offset=rel_frac_offset_o
    )
    return translations, orientations


@torch.jit.script
def get_global_positions(
        t: torch.Tensor,
        translations: torch.Tensor,
        orientations: torch.Tensor,
        times: torch.Tensor,
        equidistant_times: bool,
        camera_idx: int,
        background_idx: Optional[torch.Tensor],
        has_background: bool,
        right_idx: torch.Tensor,
        rel_frac: torch.Tensor,
        background_orientation: torch.Tensor,
        background_translation: torch.Tensor,
        background_attached_to_camera: bool = True,
        has_camera_aberration: bool = False,
        camera_aberration_orientation: torch.Tensor = None,
        camera_aberration_translation: torch.Tensor = None,
) -> torch.Tensor:
    T = len(t)
    CIDX = camera_idx

    global_positions = compose_translation_orientation(
        translations, orientations)
    global_position_camera = global_positions[CIDX]

    # If has background plane, add the background to the translations and orientations
    if has_background:

        if background_idx is None:
            raise ValueError(
                "Background index must be provided if has_background is True.")
        elif camera_idx == background_idx:
            # Add background +=1
            background_idx = background_idx + 1

        # Resample the background to match steps
        background_translation = get_translation(
            background_translation,
            times=times,
            steps=t,
            equidistant_times=equidistant_times,
            interpolation="cubic",
            right_idx=right_idx.reshape(T, 2),
            rel_frac=rel_frac
        )

        background_orientation = get_orientation(
            background_orientation,
            times=times,
            steps=t,
            equidistant_times=equidistant_times,
            interpolation="cubic",
            right_idx=right_idx.reshape(T, 2),
            rel_frac=rel_frac
        )
        if background_attached_to_camera:
            with torch.no_grad():
                back = torch.eye(4, dtype=translations.dtype,
                                 device=translations.device).unsqueeze(0).repeat(T, 1, 1)
                back[:, :3, 3] = background_translation
                back[:, :3, :3] = unitquat_to_rotmat(background_orientation)
                global_background = torch.bmm(global_position_camera, back)
        else:
            global_background = compose_translation_orientation(
                background_translation, background_orientation)
        before_pos = global_positions[:background_idx.squeeze(0)]
        after_pos = global_positions[background_idx.squeeze(0):]
        global_positions = torch.cat(
            [before_pos, global_background.unsqueeze(0), after_pos], dim=0)

    if has_camera_aberration:
        raise NotImplementedError
        camera_aberration_translation = get_translation(
            camera_aberration_translation,
            times=times,
            steps=t,
            equidistant_times=equidistant_times,
            interpolation="cubic",
            right_idx=right_idx.reshape(T, 2),
            rel_frac=rel_frac
        )

        camera_aberration_orientation = get_orientation(
            camera_aberration_orientation,
            times=times,
            steps=t,
            equidistant_times=equidistant_times,
            interpolation="cubic",
            right_idx=right_idx.reshape(T, 2),
            rel_frac=rel_frac
        )

        with torch.no_grad():
            abr = torch.eye(4, dtype=translations.dtype,
                            device=translations.device).unsqueeze(0).repeat(T, 1, 1)
            abr[:, :3, 3] = camera_aberration_translation
            abr[:, :3, :3] = unitquat_to_rotmat(camera_aberration_orientation)
            global_camera_aberration = torch.bmm(global_position_camera, abr)

        global_positions = torch.cat(
            [global_positions, global_camera_aberration.unsqueeze(0)], dim=0)

    return global_positions


def simple_zorder(
    object_nodes: TimedDiscreteSceneNode3D,
    camera: LearnedCameraSceneNode3D,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Easyst way to get the z order of the objects in the scene.
    Based on the centeroid of the objects in camera space.

    Just considers their node position. Not accurate but fast.
    Interlacing of planes is not considered.

    Parameters
    ----------
    object_nodes : TimedDiscreteSceneNode3D
        The object nodes to get the z order for.

    camera : LearnedCameraSceneNode3D
        The camera to get the z order for.

    t : torch.Tensor
        The time steps to get the z order for.
        Shape (...)

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        1. The argsorted z order of the objects in the scene.
            The indices of object_nodes in ascending order, such that when object_nodes is ordered a correct overlay is achieved.
            Shape (..., N) where N is the number of object nodes.
            Same batch size as t.
        2. The number of objects that are not in front of the camera, and need to be ignored. These are placed last in the z order and need to be cropped.
            Shape (..., 1).
        Example:
        [object_nodes[o] for o in z_order][:-last_num] if last_num > 0 will give the correct overlay.

    """
    t, shp = flatten_batch_dims(t, -1)
    # Get the global position of all nodes, and get them into camera space. Order them by z
    glob_positions = torch.stack([x.get_global_position(
        t=t) for x in object_nodes], dim=0)  # (N, T, 4, 4)
    cam_loc = camera.global_to_local(
        glob_positions[..., :3, 3], t=t, v_include_time=True)[..., :3]  # (N, T, 3)
    not_infront = cam_loc[..., 2] < 0
    cam_loc[..., 2] = torch.where(not_infront, 10000, cam_loc[..., 2])
    # Determine the z order
    z_order = torch.argsort(cam_loc[..., 2], dim=0, descending=False)  # (N, T)
    ignore_last_num = not_infront.sum(0, keepdim=True)
    # Flip dims
    z_order = z_order.permute(1, 0)  # (T, N)
    ignore_last_num = ignore_last_num.permute(1, 0)  # (T, 1)
    # (..., N)
    return unflatten_batch_dims(z_order, shp), unflatten_batch_dims(ignore_last_num, shp)


@torch.jit.script
def plane_hits(
    uv: torch.Tensor,
    inverse_intrinsics: torch.Tensor,
    lens_distortion: torch.Tensor,
    camera_idx: int,
    focal_length: torch.Tensor,
    global_positions: torch.Tensor,
    local_plane_scale: torch.Tensor,
    local_plane_scale_offset: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Calculate the intersection points of rays with objects / planes in the scene.

    Parameters
    ----------
    uv : torch.Tensor
        UV coordinates of the rays (in Camera space). Shape (B, T, 2)
        B is the batch size and T is the number of time steps.
    inverse_intrinsics : torch.Tensor
        Inverse intrinsics of the camera. Shape (T, 3, 3)
    lens_distortion : torch.Tensor
        Lens distortion of the camera. Shape (6, )
    camera_idx : int
        Index of the camera in the global positions.
    focal_length : torch.Tensor
        Focal length of the camera. Shape (1, )
    global_positions : torch.Tensor
        Global position matricies of all planes and the camera. Shape (OC, T, 4, 4)
        OC is the number of objects + camera.
    local_plane_scale : torch.Tensor
        Local scale of the planes. Shape (O, 2) where O is the number of planar objects.
    local_plane_scale_offset : torch.Tensor
        Local scale offset of the planes. Shape (O, 2) where O is the number of planar objects.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
        1. Global ray origins. Shape (B, T, 3)
        2. Global ray directions. Shape (B, T, 3)
        3. Global positions matricies of the planes (excluding camera). Shape (O, T, 4, 4)
        4. Selected mask for the planes, corresponds to the selction mask of input planar global positions. Shape (OC,)
        5. Intersection points of the rays with the infinite planes in the scene. Shape (O, B, T, 3)
        6. Boolean tensor indicating if the rays intersect with the plane within its bounds. Shape (O, B, T)
        7. Plane positions per ray as used to compute the intersection. Shape (O, B, T, 3) May be None if return_plane_positions is False.
        8. Plane normals per ray as used to compute the intersection. Shape (O, B, T, 3) May be None if return_plane_positions is False.
    """
    CIDX = camera_idx
    OB_mask = torch.ones_like(global_positions[:, 0, 0, 0], dtype=torch.bool)
    # Mask out the camera
    OB_mask[CIDX] = False
    global_plane_positions = global_positions[OB_mask]
    global_position_camera = global_positions[CIDX]

    global_ray_origins, global_ray_directions = get_global_rays(uv=uv,
                                                                inverse_intrinsics=inverse_intrinsics,
                                                                lens_distortion=lens_distortion,
                                                                global_position=global_position_camera,
                                                                focal_length=focal_length,
                                                                uv_includes_time=False)

    intersection_points, is_inside, plane_p, plane_n = get_object_intersection_points(
        global_plane_positions, local_plane_scale, local_plane_scale_offset, global_ray_origins, global_ray_directions, return_plane_positions=True)

    return global_ray_origins, global_ray_directions, global_positions, OB_mask, intersection_points, is_inside, plane_p, plane_n


class NAGFunctionalModel(NAGModel):

    _translations: torch.Tensor
    """Fixed translations of all the objects and the camera.
    These translations are absolute assuming no hierarchy.
    Shape (N, T, 3), where N is the number of objects + camera (Excluding Background) and camera is always the last index.
    """
    _orientation: torch.Tensor
    """Fixed orientations as (N, t, 4) (Excluding Background). (x, y, z, w) for each time step t. Order for the translations apply also here."""

    _node_indices: torch.Tensor
    """The node / object indices for correspondences in the model. Includes all Nodes. Shape (OBA + 1, ) All objects"""

    _camera_index_in_node_indices: torch.Tensor
    """The camera index in the node indices. Shape (1, )"""

    _background_index_in_node_indices: Optional[torch.Tensor]
    """The background index in the node indices. Shape (1, )"""

    _times: torch.Tensor
    """Timestamps for which the base translations and orientations are recorded. Will be in range [0, 1]. Shape (t,)"""

    _equidistant_times: bool
    """Flag whether the times are equidistant."""

    _offset_translation: torch.nn.Parameter
    """Learnable translation offsets for each object and the camera. Shape (N, TC, 3)"""

    _offset_translation_learned_mask: torch.Tensor
    """Boolean buffer describing which objects have a offset learned translation (N, )"""

    _offset_rotation_vector: torch.nn.Parameter
    """Learnable rotation offsets for each object and the camera. Shape (N, TC, 3)"""

    _offset_rotation_vector_learned_mask: torch.Tensor
    """Boolean buffer describing which objects have a offset learned rotation (N, )"""

    _offset_times: torch.Tensor
    """Timestamps for which the offset translations and rotations are recorded. Will be in range [0, 1]. Shape (TC,)"""

    _offset_translation_weight: torch.Tensor
    """Learnable weight for the translation offsets. Shape (N,)"""

    _offset_rotation_weight: torch.Tensor
    """Learnable weight for the rotation offsets. Shape (N,)"""

    _times: torch.Tensor
    """The times for which the scene is evaluated. Shape (T,)"""

    camera_idx: int
    """Index of the camera in the translations and orientations. Usually the last index in translations."""

    background_idx: Optional[int]
    """Index of the background in plane_scale and plane_scale_offset. Usually the last index.
    """
    _plane_scale: torch.Tensor
    """The scale of all the planes. Shape (O [+1], 2)  Contains Background if present."""

    _plane_scale_offset: torch.Tensor
    """The scale offset of all the planes. Shape (O [+1], 2) Contains Background if present."""

    _background_translation: torch.Tensor
    """Fixed translations of the background. Shape (T, 3)"""

    _background_orientation: torch.Tensor
    """Fixed orientations of the background. Shape (T, 4)"""

    _is_background_attached_to_camera: torch.Tensor
    """Boolean Flag tensor whether the background is attached to the camera."""

    num_objects: int
    """Number of objects in the scene O. Excluding the camera, the background and camera aberration plane."""

    _interpolated_steps: Optional[torch.Tensor]
    """The interpolated steps for which _right_idx and _rel_frac ... are computed. Shape (S, )"""

    _right_idx: Optional[torch.Tensor]
    """The right index for the interpolation. Shape (S, 2) where the first is collapsed batch index (0) and the second is the object index."""

    _rel_frac: Optional[torch.Tensor]
    """The relative fraction for the interpolation. Shape (1, S)"""

    _right_idx_offset: Optional[torch.Tensor]
    """The right index for the interpolation for offset stuff. Shape (S, 2) where the first is collapsed batch index (0) and the second is the object index."""

    _rel_frac_offset: Optional[torch.Tensor]
    """The relative fraction for the interpolation for offset stuff. Shape (1, S)"""

    _plane_normal_gradient_hook_fnc: Optional[Callable[[
        torch.Tensor], torch.Tensor]]
    """Hook function for the plane normal gradient. Will be applied during forward. Shape of Incomming Gradient is (N, B, T, 3)."""

    _plane_normal_gradient_hook_kwargs: Optional[Dict[str, Any]]
    """Keyword arguments for the plane normal gradient hook function."""

    _plane_position_gradient_hook_fnc: Optional[Callable[[
        torch.Tensor], torch.Tensor]]
    """Hook function for the plane position gradient. Will be applied during forward. Shape of Incomming Gradient is (N, B, T, 3)."""

    _plane_position_gradient_hook_kwargs: Optional[Dict[str, Any]]
    """Keyword arguments for the plane position gradient hook function."""

    __plane_normal_gradient_hook: Optional[RemovableHandle]
    """Handle for the plane normal gradient hook."""

    __plane_position_gradient_hook: Optional[RemovableHandle]
    """Handle for the plane normal gradient hook."""

    _position_gradient_rescaling: bool
    """Flag whether the gradient rescaling is enabled."""

    _sample_position_fnc: Optional[Callable[[
        torch.Tensor, torch.Tensor, int, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]
    """Function to sample the position of the objects used for stochastic relaxation.
    Will be called after the positions are computed.
    Should return the new translations and orientations.
    Gets the translations, orientations, camera index and sin epoch as input.
    """

    def __init__(self,
                 num_objects: int,
                 times: torch.Tensor,
                 config: NAGConfig,
                 world: Optional[ModuleSceneNode3D] = None,
                 resolution: Optional[Tuple[int, int]] = None,
                 **kwargs):
        """Constructor.

        Parameters
        ----------
        num_objects : int
            Number of objects in the scene.
            Excluding the camera but including the background and abberation plane (if any).
        times : torch.Tensor
            The times for which the scene is evaluated. Shape (T,)
        config : NAGConfig
            NAG configuration.
        """
        super().__init__(config, world=world, resolution=resolution, **kwargs)
        self._equidistant_times = True
        self.init_parameters(num_objects, times)
        self._sample_position_fnc = None
        self._store_object_alphas = None
        self._store_object_flows = None
        self._store_view_dependent_tv = None
        self._store_view_dependent_lap = None
        self._object_idx = None
        self.init_hooks(config)

    def set_position_gradient_rescaling(self, enable: bool, phase: Optional[Any] = None):
        from tools.util.format import parse_type, DEFAULT
        self._position_gradient_rescaling = enable
        if enable:
            from nag.model.phase import Phase
            if phase is not None and isinstance(phase, Phase):
                plane_position_gradient_hook_fnc = parse_type(
                    phase.plane_position_rescaling_hook,
                    instance_type=Callable,
                    handle_not_a_class="ignore"
                ) if phase.plane_position_rescaling_hook != DEFAULT else DEFAULT
                plane_position_gradient_hook_kwargs = dict(
                    phase.plane_position_rescaling_hook_kwargs) if phase.plane_position_rescaling_hook_kwargs is not None else dict()
                plane_normal_gradient_hook_fnc = parse_type(
                    phase.plane_normal_rescaling_hook,
                    instance_type=Callable,
                    handle_not_a_class="ignore"
                ) if phase.plane_normal_rescaling_hook != DEFAULT else DEFAULT
                plane_normal_gradient_hook_kwargs = dict(
                    phase.plane_normal_rescaling_hook_kwargs) if phase.plane_normal_rescaling_hook_kwargs is not None else dict()

            else:
                config = self.config
                plane_position_gradient_hook_kwargs = dict(
                    config.plane_position_rescaling_hook_kwargs) if config.plane_position_rescaling_hook_kwargs is not None else dict()
                plane_position_gradient_hook_fnc = parse_type(
                    config.plane_position_rescaling_hook,
                    instance_type=Callable,
                    handle_not_a_class="ignore"
                )
                plane_normal_gradient_hook_kwargs = dict(
                    config.plane_normal_rescaling_hook_kwargs) if config.plane_normal_rescaling_hook_kwargs is not None else dict()
                plane_normal_gradient_hook_fnc = parse_type(
                    config.plane_normal_rescaling_hook,
                    instance_type=Callable,
                    handle_not_a_class="ignore"
                )

            if plane_position_gradient_hook_kwargs != DEFAULT:
                self._plane_position_gradient_hook_kwargs = plane_position_gradient_hook_kwargs
            if plane_normal_gradient_hook_kwargs != DEFAULT:
                self._plane_normal_gradient_hook_kwargs = plane_normal_gradient_hook_kwargs
            if plane_position_gradient_hook_fnc != DEFAULT:
                self._plane_position_gradient_hook_fnc = plane_position_gradient_hook_fnc
            if plane_normal_gradient_hook_fnc != DEFAULT:
                self._plane_normal_gradient_hook_fnc = plane_normal_gradient_hook_fnc
            self.__plane_normal_gradient_hook = None
            self.__plane_position_gradient_hook = None
        else:
            if self.__plane_normal_gradient_hook is not None:
                self.__plane_normal_gradient_hook.remove()
                self.__plane_normal_gradient_hook = None
            if self.__plane_position_gradient_hook is not None:
                self.__plane_position_gradient_hook.remove()
                self.__plane_position_gradient_hook = None

            self._plane_normal_gradient_hook_kwargs = None
            self._plane_position_gradient_hook_kwargs = None
            self._plane_normal_gradient_hook_fnc = None
            self._plane_position_gradient_hook_fnc = None

    @property
    def store_object_alphas(self) -> bool:
        if self._store_object_alphas is None:
            from nag.loss.mask_loss_mixin import MaskLossMixin
            self._store_object_alphas = isinstance(self.loss, MaskLossMixin)
        return self._store_object_alphas

    @property
    def store_object_flows(self) -> bool:
        if self._store_object_flows is None:
            self._store_object_flows = False  # Debugging option
        return self._store_object_flows

    @property
    def store_object_view_dependent_tv(self) -> bool:
        """Triggers storing of the view dependent control points for each object in training forward."""
        if self._store_view_dependent_tv is None:
            self._store_view_dependent_tv = False  # Debugging option
        return self._store_view_dependent_tv

    @property
    def store_object_view_dependent_lap(self) -> bool:
        """Triggers storing of the view dependent control points for each object in training forward."""
        if self._store_view_dependent_lap is None:
            self._store_view_dependent_lap = False  # Debugging option
        return self._store_view_dependent_lap

    def init_hooks(self, config: NAGConfig):
        """Initialize hooks for the model.

        Parameters
        ----------
        config : NAGConfig
            Configuration for the model.
        """
        from tools.util.format import parse_type
        self._position_gradient_rescaling = None
        self._plane_normal_gradient_hook_fnc = None
        self._plane_normal_gradient_hook_kwargs = None
        self._plane_position_gradient_hook_fnc = None
        self._plane_position_gradient_hook_kwargs = None
        self.__plane_normal_gradient_hook = None
        self.__plane_position_gradient_hook = None
        self.set_position_gradient_rescaling(
            config.plane_position_gradient_rescaling)

        if config.plane_position_sampling:
            fnc = parse_type(
                config.plane_position_sampling_hook,
                instance_type=Callable,
                handle_not_a_class="ignore"
            )
            if config.plane_position_sampling_hook_kwargs is not None and len(config.plane_position_sampling_hook_kwargs) > 0:
                fnc = partial(
                    fnc, **config.plane_position_sampling_hook_kwargs)
            self._sample_position_fnc = fnc
        else:
            self._sample_position_fnc = None

    def setup_loss(self, resolution: Optional[Tuple[int, int]]) -> None:
        from tools.metric.torch.reducible import Metric
        from tools.metric.torch.module_mixin import ModuleMixin
        from tools.util.format import parse_type
        from nag.loss.l1_mask_loss import L1MaskLoss

        """Setup the loss function for the NAG model."""
        loss_type = self.config.loss_type
        loss_args = self.config.loss_kwargs if self.config.loss_kwargs is not None else {}
        if issubclass(loss_type, ModuleMixin):
            loss_args["module"] = self
        loss = loss_type(**loss_args)
        return loss

    def _get_interpolate_index(self, t: Optional[torch.Tensor] = None):
        if t is None:
            t = self._times

        right_idx: torch.Tensor
        rel_frac: torch.Tensor
        right_idx_offset: torch.Tensor
        rel_frac_offset: torch.Tensor

        if self._interpolated_steps is not None and (t.shape == self._interpolated_steps.shape and (t == self._interpolated_steps).all()):
            right_idx = self._right_idx
            rel_frac = self._rel_frac
            right_idx_offset = self._right_idx_offset
            rel_frac_offset = self._rel_frac_offset
        else:
            # Compute the right index and relative fraction
            right_idx, rel_frac = _get_interpolate_index_and_distance(
                self._times.unsqueeze(0), t.unsqueeze(0), self._equidistant_times)
            right_idx_offset, rel_frac_offset = _get_interpolate_index_and_distance(
                self._offset_times.unsqueeze(0), t.unsqueeze(0), True)
            self._interpolated_steps = t
            self._right_idx = right_idx
            self._rel_frac = rel_frac
            self._right_idx_offset = right_idx_offset
            self._rel_frac_offset = rel_frac_offset

        # Repeat for the batch size
        right_idx = right_idx.unsqueeze(0).int()
        right_idx_offset = right_idx_offset.unsqueeze(0).int()
        return right_idx, rel_frac, right_idx_offset, rel_frac_offset

    @property
    def node_indices(self) -> torch.Tensor:
        return self._node_indices

    def forward(self,
                uv: torch.Tensor,
                t: torch.Tensor,
                batch_idx: Optional[int] = None,
                context: Optional[Dict[str, Any]] = None,
                ):
        if context is not None and self.training:
            context["store_object_alpha"] = self.store_object_alphas
            context["store_object_flow"] = self.store_object_flows

        # Compute global rays
        objects = self.objects
        has_background = self.config.has_background_plane
        has_camera_aberration = self.config.has_camera_aberration_plane
        B = uv.shape[0]
        O = self.num_objects
        T = len(t)
        OB = O + int(has_background)  # Objects + Camera Without Background
        # Objects + Camera With Background and Camera Aberration
        OBA = OB + int(has_camera_aberration)

        CIDX = self.camera_idx

        local_plane_scale = self._plane_scale
        local_plane_scale_offset = self._plane_scale_offset
        right_idx, rel_frac, right_idx_offset, rel_frac_offset = self._get_interpolate_index(
            t)

        translations, orientations = get_translation_orientation(
            t=t,
            translations=self._translations,
            translation_offset=self.get_offset_translation(),
            orientations=self._orientation,
            rotation_offset=self.get_offset_rotation_vector(),
            translation_offset_weight=self._offset_translation_weight,
            rotation_offset_weight=self._offset_rotation_weight,
            times=self._times,
            offset_times=self._offset_times,
            equidistant_times=self._equidistant_times,
            equidistant_offset_times=True,
            right_idx=right_idx,
            rel_frac=rel_frac,
            right_idx_offset=right_idx_offset,
            rel_frac_offset=rel_frac_offset
        )
        if not torch.isfinite(translations).all() or not torch.isfinite(orientations).all():
            logger.error("Translations or orientations are not finite.")

        if self._sample_position_fnc is not None and self.training:
            translations, orientations = self._sample_position_fnc(
                translations, orientations, CIDX, self.sin_epoch)

        global_positions = get_global_positions(
            t=t,
            translations=translations,
            orientations=orientations,
            times=self._times,
            equidistant_times=self._equidistant_times,
            camera_idx=CIDX,
            background_idx=tensorify(
                self._background_index_in_node_indices) if self._background_index_in_node_indices is not None else None,
            has_background=has_background,
            right_idx=right_idx,
            rel_frac=rel_frac,
            background_orientation=self._background_orientation,
            background_translation=self._background_translation,
            background_attached_to_camera=self._is_background_attached_to_camera,
            has_camera_aberration=self.config.has_camera_aberration_plane,
            camera_aberration_orientation=self._camera_aberration_orientation,
            camera_aberration_translation=self._camera_aberration_translation
        )

        global_ray_origins, global_ray_directions, _, OB_mask, intersection_points, is_inside, plane_p, plane_n = plane_hits(
            uv=uv,
            inverse_intrinsics=self.camera.get_inverse_intrinsics(t),
            lens_distortion=self.camera.get_lens_distortion(),
            camera_idx=CIDX,
            focal_length=self.focal_length,
            global_positions=global_positions,
            local_plane_scale=local_plane_scale,
            local_plane_scale_offset=local_plane_scale_offset,
        )

        global_plane_positions = global_positions[OB_mask]

        colors = torch.zeros(
            (OBA, B, T, 3), device=uv.device, dtype=uv.dtype)
        alphas = torch.zeros(
            (OBA, B, T, 1), device=uv.device, dtype=uv.dtype)
        for i, obj in enumerate(objects):
            # Trace each ray through the object if it is inside
            obj: torch.nn.Module
            # obj_inter = intersection_points[i]
            obj_is_inter = is_inside[i]
            any_per_time = obj_is_inter.any(dim=-1)

            colors[i, any_per_time], alphas[i, any_per_time] = obj(intersection_points[i, any_per_time],
                                                                   is_inside=obj_is_inter[any_per_time],
                                                                   ray_origins=global_ray_origins[any_per_time],
                                                                   ray_directions=global_ray_directions[any_per_time],
                                                                   t=t,
                                                                   sin_epoch=self.sin_epoch,
                                                                   global_position=global_plane_positions[i],
                                                                   plane_scale=local_plane_scale[i],
                                                                   plane_scale_offset=local_plane_scale_offset[i],
                                                                   next_sin_epoch=self.next_sin_epoch,
                                                                   batch_idx=batch_idx,
                                                                   max_batch_idx=self.num_batches,
                                                                   context=context
                                                                   )

        if not self._position_gradient_rescaling or not torch.is_grad_enabled():
            object_colors, object_alphas = compute_object_rgba(colors=colors, alphas=alphas, t=t,
                                                               global_ray_origins=global_ray_origins, global_ray_directions=global_ray_directions,
                                                               is_inside=is_inside, intersection_points=intersection_points)

            ray_color = (object_alphas * object_colors).sum(dim=0)
        else:
            object_colors, object_alphas, order, alpha_chain, sorted_alphas = compute_object_rgba(colors=colors, alphas=alphas, t=t,
                                                                                                  global_ray_origins=global_ray_origins, global_ray_directions=global_ray_directions,
                                                                                                  is_inside=is_inside, intersection_points=intersection_points, get_object_alpha_chain=True)

            _, _, unsorted_alpha_chain, unsorted_alphas = undo_intersection_ordering(
                order=order, sorted_alpha_chain=alpha_chain, sorted_alpha=sorted_alphas)

            ray_color = (object_alphas * object_colors).sum(dim=0)

            if self._plane_normal_gradient_hook_fnc is not None:
                plane_n.requires_grad_(True)
                self.__plane_normal_gradient_hook = plane_n.register_hook(
                    partial(self._plane_normal_gradient_hook_fnc,
                            alpha_chain=unsorted_alpha_chain,
                            alpha=unsorted_alphas,
                            is_inside=is_inside,
                            **self._plane_normal_gradient_hook_kwargs))
            if self._plane_position_gradient_hook_fnc is not None:
                plane_p.requires_grad_(True)
                self.__plane_position_gradient_hook = plane_p.register_hook(partial(
                    self._plane_position_gradient_hook_fnc, alpha_chain=unsorted_alpha_chain,
                    alpha=unsorted_alphas,
                    is_inside=is_inside,
                    **self._plane_position_gradient_hook_kwargs))

        if context is not None:
            if self._store_object_alphas:
                valid_ids = self.node_indices[OB_mask.cpu()]

                # Merge alphas along channel dimension
                value = context.get("object_alpha", None)
                if value is not None:
                    object_alpha_stack = torch.zeros(
                        (B, T, O), device=uv.device, dtype=uv.dtype)
                    for i, v in enumerate(value.keys()):
                        iidx = torch.argwhere(valid_ids == v).item()
                        object_alpha_stack[is_inside[iidx].any(
                            dim=-1), :, i] = value[v][..., 0]
                    covered_ids = torch.tensor(
                        list(value.keys()), device=uv.device, dtype=torch.int32)
                    context["object_alpha_index"] = covered_ids
                    context["object_alpha"] = object_alpha_stack[...,
                                                                 :len(covered_ids)]
                else:
                    context["object_alpha_index"] = torch.tensor(
                        [], device=uv.device, dtype=torch.int32)
                    context["object_alpha"] = torch.zeros(
                        (B, T, 0), device=uv.device, dtype=uv.dtype)
            if self.store_object_flows:
                value = context.get("object_flow", None)
                object_flow_stack = torch.stack(
                    [value[k] for k in sorted(value.keys())], dim=-1)  # Shape (B, T, 2, O)
                context["object_flow"] = object_flow_stack
        return ray_color

    def training_step(self, train_batch: Any, batch_idx: torch.Tensor) -> torch.Tensor:
        # torch.autograd.set_detect_anomaly(True)
        uv, true_rgb, t, weight_t, data = train_batch
        # Squeeze the Dataloader batch dim
        uv = uv.squeeze(0)
        true_rgb = true_rgb.squeeze(0)
        t = t.squeeze(0)
        weight_t = weight_t.squeeze(0)

        pred_rgb = self(uv, t, batch_idx, context=data)

        metrics = dict()

        data["uv"] = uv
        data["t"] = t

        loss = self.loss(pred_rgb, true_rgb,
                         time_weight=weight_t,
                         context=data, metrics=metrics)

        # plot = True
        # if plot:
        #     self.trainer.train_dataloader.dataset.plot_output(uv, true_rgb, t=t, t_weight=weight_t, data=data, outputs=pred_rgb, save=True, open=True)

        metrics["loss/sum"] = loss.detach().cpu()
        metrics["sin_epoch"] = self.sin_epoch.detach().cpu()
        metrics["epoch"] = self.current_epoch

        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_after_backward(self) -> None:
        if self._position_gradient_rescaling:
            if self.__plane_normal_gradient_hook is not None:
                hk = self.__plane_normal_gradient_hook
                self.__plane_normal_gradient_hook = None
                hk.remove()
            if self.__plane_position_gradient_hook is not None:
                hk = self.__plane_position_gradient_hook
                self.__plane_position_gradient_hook = None
                hk.remove()

    def get_grad_scaler(self) -> Optional[Any]:
        if not self.config.use_amp:
            return None
        try:
            return self._trainer.strategy._lightning_optimizers[0]._strategy.precision_plugin.scaler
        except Exception as e:
            return None

    def get_object_translation_orientations(self, t: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        right_idx, rel_frac, right_idx_offset, rel_frac_offset = self._get_interpolate_index(
            t)
        translations, orientations = get_translation_orientation(
            t=t,
            translations=self._translations,
            translation_offset=self.get_offset_translation(),
            orientations=self._orientation,
            rotation_offset=self.get_offset_rotation_vector(),
            translation_offset_weight=self._offset_translation_weight,
            rotation_offset_weight=self._offset_rotation_weight,
            times=self._times,
            offset_times=self._offset_times,
            equidistant_times=self._equidistant_times,
            equidistant_offset_times=True,
            right_idx=right_idx,
            rel_frac=rel_frac,
            right_idx_offset=right_idx_offset,
            rel_frac_offset=rel_frac_offset
        )
        return translations, orientations

    def get_oid_in_to_mask(self, oid: int) -> torch.Tensor:
        indices = self.get_node_indices()
        is_obj = torch.ones_like(indices, dtype=torch.bool)
        is_learned_obj = torch.ones_like(indices, dtype=torch.bool)
        is_obj[self.camera_idx] = False
        obj_indices = indices[is_obj]
        back_idx = obj_indices[self.background_idx]
        is_learned_obj[indices == back_idx] = False
        not_background_nodes = indices[is_learned_obj]
        look_mask = not_background_nodes == oid
        return look_mask

    def get_oid_to_positions_mask(self, oid: int) -> torch.Tensor:
        base_mask = torch.ones(self._node_indices.shape[0], dtype=torch.bool)
        # Disable background
        if self._background_index_in_node_indices is not None:
            base_mask[self._background_index_in_node_indices] = False
        ids = self._node_indices[base_mask]
        return ids == oid

    def get_oid_to_plane_mask(self, oid: int) -> torch.Tensor:
        base_mask = torch.ones(self._node_indices.shape[0], dtype=torch.bool)
        # Disable Camera
        base_mask[self._camera_index_in_node_indices] = False
        ids = self._node_indices[base_mask]
        return ids == oid

    def get_object_translation_orientations_oid(self, t: torch.Tensor, oid: int) -> Tuple[torch.Tensor, torch.Tensor]:
        right_idx, rel_frac, right_idx_offset, rel_frac_offset = self._get_interpolate_index(
            t)

        indices = self.get_node_indices()
        is_obj = torch.ones_like(indices, dtype=torch.bool)
        is_learned_obj = torch.ones_like(indices, dtype=torch.bool)

        is_obj[self.camera_idx] = False
        obj_indices = indices[is_obj]

        back_idx = obj_indices[self.background_idx]
        is_learned_obj[indices == back_idx] = False

        not_background_nodes = indices[is_learned_obj]
        look_mask = not_background_nodes == oid
        translations, orientations = get_translation_orientation(
            t=t,
            translations=self._translations,
            translation_offset=self.get_offset_translation(),
            orientations=self._orientation,
            rotation_offset=self.get_offset_rotation_vector(),
            translation_offset_weight=self._offset_translation_weight,
            rotation_offset_weight=self._offset_rotation_weight,
            times=self._times,
            offset_times=self._offset_times,
            equidistant_times=self._equidistant_times,
            equidistant_offset_times=True,
            right_idx=right_idx,
            rel_frac=rel_frac,
            right_idx_offset=right_idx_offset,
            rel_frac_offset=rel_frac_offset
        )
        return translations[look_mask], orientations[look_mask]

    def set_object_translation_orientations(self, translations: torch.Tensor, orientations: torch.Tensor, t: torch.Tensor, oid: int):
        if not torch.allclose(t, self._times):
            raise ValueError(
                "Times are not equal to the original times. Can't set translation/orientation.")
        look_mask = self.get_oid_in_to_mask(oid)
        cur_val_t, cur_val_o = self.get_object_translation_orientations_oid(
            t, oid)

        delta_t, delta_o = cur_val_t - \
            translations, quat_subtraction(cur_val_o, orientations)

        new_translations = self._translations[look_mask] - delta_t
        new_quat = quat_subtraction(self._orientation[look_mask], delta_o)

        self._translations[look_mask] = new_translations
        self._orientation[look_mask] = new_quat

    def retime_object(self, oid: int, t_shift: int = 0):
        """
        Shift the object translation and orientation timely by t_shift timestamps.

        Parameters
        ----------
        oid : int
            Oid to shift.

        t_shift : int, optional
            Timeshift, positive values shift into the future, negative into the past, by default 0
        """
        def shift_vector(vector: torch.Tensor, roll: int) -> torch.Tensor:
            """Shift the vector by roll amount."""
            vector = tensorify(vector, dtype=torch.float32).clone()
            vector = torch.roll(vector, roll, dims=1)
            if roll > 0:
                vector[:, :roll] = vector[:, roll]
            elif roll < 0:
                vector[:, roll:] = vector[:, roll-1]
            return vector

        org_trans, org_orient = self.get_object_translation_orientations_oid(
            self._times, oid)
        new_trans = shift_vector(org_trans, -1 * t_shift)
        new_orient = shift_vector(org_orient, -1 * t_shift)
        self.set_object_translation_orientations(
            new_trans, org_orient, self._times, oid)

    def shift_object(self, oid: int, translation: torch.Tensor, orientation: Optional[torch.Tensor] = None):
        """
        Shift the objects translation and orientation by the given translation and orientation.

        Parameters
        ----------
        oid : int
            Oid to shift.

        translation : torch.Tensor
            Translation to shift by. Shape (3,) or (T, 3).

        orientation : Optional[torch.Tensor], optional
            Orientation to shift by. Shape (4,) or (T, 4), by default None
        """
        from tools.transforms.geometric.quaternion import quat_product

        org_trans, org_orient = self.get_object_translation_orientations_oid(
            self._times, oid)
        translation = tensorify(
            translation, dtype=org_trans.dtype, device=org_trans.device)
        new_trans = org_trans + translation
        if orientation is not None:
            orientation = tensorify(
                orientation, dtype=org_orient.dtype, device=org_orient.device)
            orientation = orientation.expand_as(org_orient)
            new_orient = quat_product(org_orient, orientation)
        else:
            new_orient = org_orient
        self.set_object_translation_orientations(
            new_trans, org_orient, self._times, oid)

    def duplicate_object(self, oid: int) -> int:
        with torch.no_grad():
            import copy
            import re
            from nag.strategy.plane_initialization_strategy import PlaneInitializationStrategy
            from tools.util.torch import on_load_checkpoint
            obj = next((x for x in self.objects if x.get_index() == oid), None)
            if obj is None:
                raise ValueError(
                    f"Object with oid {oid} not found in the scene.")
            look_mask = self.get_oid_in_to_mask(oid)
            state = TensorUtil.apply_deep(
                obj.state_dict(), lambda x: x.detach().cpu().clone())

            translation_copy = self._translations[look_mask].clone()
            orientation_copy = self._orientation[look_mask].clone()
            offset_translation_copy = self._offset_translation[look_mask].clone(
            )
            offset_rotation_vector_copy = self._offset_rotation_vector[look_mask].clone(
            )
            offset_rotation_vector_learned_mask_copy = self._offset_rotation_vector_learned_mask[look_mask].clone(
            )
            offset_translation_weight_copy = self._offset_translation_weight[look_mask].clone(
            )
            offset_rotation_weight_copy = self._offset_rotation_weight[look_mask].clone(
            )
            plane_scale_copy = self._plane_scale[look_mask].clone()
            plane_scale_offset_copy = self._plane_scale_offset[look_mask].clone(
            )

            max_idx = self._node_indices.max().item()

            mask = self.enlarge_model(1)
            new_idx = mask.argwhere().squeeze(0)

            copy_name_pattern = "_copy\_(?P<dub>\d+)$"
            name = obj.get_name()
            matches = [re.search(name + copy_name_pattern, node.get_name())
                       for node in self.objects]
            max_copy_idx = -1
            for match in matches:
                if match:
                    copy_idx = int(match.group("dub"))
                    if copy_idx > max_copy_idx:
                        max_copy_idx = copy_idx
            if max_copy_idx == -1:
                new_name = name + "_copy_1"
            else:
                new_name = name + f"_copy_{max_copy_idx + 1}"
            strategy = PlaneInitializationStrategy()
            init_args = dict(
                object_index=max_idx + 1,
                mask_index=-1,
                images=None,
                masks=None,
                depths=None,
                times=self._times,
                camera=self.camera,
                nag_model=self,
                # dataset=dataset,
                config=self.config,
                name=new_name,
                proxy_init=True,
                dataset=None
            )
            args = strategy(**init_args)
            args = self.patch_plane_args(object_idx=new_idx, args=args)

            _type = type(obj)
            new_obj = _type(proxy_init=True, **args)
            self._objects.append(new_obj)
            if hasattr(obj, "box") and obj.box is not None:
                from nag.model.timed_box_scene_node_3d import TimedBoxSceneNode3D
                box = TimedBoxSceneNode3D(name=obj.box.get_name(
                ), size=obj.box.size.detach().clone(),  position=torch.eye(4).unsqueeze(0))
                new_obj.box = box

            on_load_checkpoint(new_obj, state, allow_loading_unmatching_parameter_sizes=True,
                               load_missing_parameters_as_defaults=True)
            new_obj.load_state_dict(state)
            new_look_mask = self.get_oid_in_to_mask(new_idx).squeeze(0)

            self._translations[new_look_mask] = translation_copy
            self._orientation[new_look_mask] = orientation_copy
            self._offset_translation[new_look_mask] = offset_translation_copy
            self._offset_rotation_vector[new_look_mask] = offset_rotation_vector_copy
            self._offset_rotation_vector_learned_mask[new_look_mask] = offset_rotation_vector_learned_mask_copy
            self._offset_translation_weight[new_look_mask] = offset_translation_weight_copy
            self._offset_rotation_weight[new_look_mask] = offset_rotation_weight_copy
            self._plane_scale[new_look_mask] = plane_scale_copy
            self._plane_scale_offset[new_look_mask] = plane_scale_offset_copy
            self.world.add_scene_children(new_obj)
            return new_idx

    def _assemble_object_rigid_data(self,
                                    t: torch.Tensor,
                                    enabled_objects: torch.Tensor,
                                    enabled_background: bool = True,
                                    enabled_aberration: bool = False,
                                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Combines object data into tensors.
        1. the rigid data e.g. global position matricies for the enabled objects, background as well as camera.  
        2. the local plane scale for the enabled objects and background
        3. the local plane scale offset for the enabled objects and background
        4. the camera index in the returned
        
        
        Parameters
        ----------
        t : torch.Tensor
            The time steps to compute the rigid data for. Shape (T,)
        enabled_objects : torch.Tensor
            Which objects are enabled. Shape (N,), N number of total objects
        enabled_background : bool, optional
            If background plane shall be included, by default True
        enabled_aberration : bool, optional
            If aberration plane shall be included, by default False

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
            1. the rigid data e.g. global position matricies for the enabled objects, background as well as camera. Shape (SOBC, T, 4, 4)
            2. the local plane scale for the enabled objects and background. Shape (SOB, 2)
            3. the local plane scale offset for the enabled objects and background. Shape (SOB, 2)
            4. the camera index in the returned 

            Where T is the number of time steps, SOBC is the number of selected objects + optional background + camera, SOB is the number of selected objects + optional background.
        """
        device = enabled_objects.device
        # enabled_objects_cam = torch.cat(
        #     [enabled_objects, torch.tensor([True], dtype=torch.bool, device=device)])

        only_obj_masks = torch.ones(
            self._node_indices.shape[0], dtype=torch.bool, device=device)
        only_cam_mask = ~only_obj_masks.clone()
        no_bg_mask = only_obj_masks.clone()
        only_bg_mask = ~only_obj_masks.clone()
        no_cam_mask = only_obj_masks.clone()

        only_cam_mask[self._camera_index_in_node_indices] = True
        only_obj_masks[self._camera_index_in_node_indices] = False
        only_obj_masks[self._background_index_in_node_indices] = False

        if self.config.has_background_plane:
            only_bg_mask[self._background_index_in_node_indices] = True
            no_bg_mask[self._background_index_in_node_indices] = False

        no_bg_mask[only_obj_masks] = enabled_objects

        no_cam_mask[self._camera_index_in_node_indices] = False
        if self.config.has_background_plane:
            no_cam_mask[self._background_index_in_node_indices] = enabled_background
        no_cam_mask[only_obj_masks] = enabled_objects

        # enabled_objects_back = enabled_objects.clone()
        # if self.config.has_background_plane:
        #     enabled_objects_back = torch.cat([enabled_objects_back, torch.tensor(
        #         [enabled_background], dtype=torch.bool, device=device)])
        # if self.config.has_camera_aberration_plane:
        #     enabled_objects_back = torch.cat([enabled_objects_back, torch.tensor(
        #         [enabled_aberration], dtype=torch.bool, device=device)])

        O = enabled_objects.sum()
        T = len(t)
        # Objects
        OB = O + int(self.config.has_background_plane and enabled_background)
        OBA = OB + \
            int(self.config.has_camera_aberration_plane and enabled_aberration)

        local_plane_scale = self._plane_scale[no_cam_mask[~only_cam_mask]]
        local_plane_scale_offset = self._plane_scale_offset[no_cam_mask[~only_cam_mask]]
        right_idx, rel_frac, right_idx_offset, rel_frac_offset = self._get_interpolate_index(
            t)

        translations, orientations = get_translation_orientation(
            t=t,
            translations=self._translations[no_bg_mask[~only_bg_mask]],
            translation_offset=self.get_offset_translation()[
                no_bg_mask[~only_bg_mask]],
            orientations=self._orientation[no_bg_mask[~only_bg_mask]],
            rotation_offset=self.get_offset_rotation_vector()[
                no_bg_mask[~only_bg_mask]],
            translation_offset_weight=self._offset_translation_weight[no_bg_mask[~only_bg_mask]],
            rotation_offset_weight=self._offset_rotation_weight[no_bg_mask[~only_bg_mask]],
            times=self._times,
            offset_times=self._offset_times,
            equidistant_times=self._equidistant_times,
            equidistant_offset_times=True,
            right_idx=right_idx,
            rel_frac=rel_frac,
            right_idx_offset=right_idx_offset,
            rel_frac_offset=rel_frac_offset
        )
        CIDX = (self._node_indices[no_bg_mask] == self._node_indices[self._camera_index_in_node_indices]).argwhere(
        ).squeeze(0).item()

        if enabled_background:
            back_idx = (self._node_indices[no_cam_mask] == self._node_indices[self._background_index_in_node_indices]).argwhere(
            ).squeeze(0)
        else:
            back_idx = None

        global_positions = get_global_positions(
            t=t,
            translations=translations,
            orientations=orientations,
            times=self._times,
            equidistant_times=self._equidistant_times,
            camera_idx=CIDX,
            background_idx=tensorify(
                back_idx) if back_idx is not None else None,
            has_background=self.config.has_background_plane and enabled_background,
            right_idx=right_idx,
            rel_frac=rel_frac,
            background_orientation=self._background_orientation,
            background_translation=self._background_translation,
            background_attached_to_camera=self._is_background_attached_to_camera,
            has_camera_aberration=self.config.has_camera_aberration_plane and enabled_aberration,
            camera_aberration_orientation=self._camera_aberration_orientation,
            camera_aberration_translation=self._camera_aberration_translation
        )
        return global_positions, local_plane_scale, local_plane_scale_offset, CIDX

    def set_grad_scaler_enabled(self, enable: bool):
        scaler = self.get_grad_scaler()
        if scaler is not None:
            scaler._enabled = enable

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        if not self.config.log_gradient:
            return
        vals = dict()
        tag = "grad/"

        if self.config.use_amp:
            scaler = self.get_grad_scaler()
            if scaler is not None and scaler._enabled:
                vals[tag + "scale"] = scaler._scale

        if self.config.log_gradient:
            for pname, param in self.named_parameters():
                name = pname.replace("_world._scene_children.",
                                     "").replace(".params", "")
                if param.grad is not None:
                    vals[tag + "norm/" + name] = param.grad.norm(2).item()
        self.log_dict(vals, on_step=True, on_epoch=False)

    def object_specific_forward(self,
                                uv: torch.Tensor,
                                t: torch.Tensor,
                                enabled_objects: torch.Tensor,
                                enabled_background: bool = True,
                                enabled_aberration: bool = True,
                                color_composing: bool = True,
                                batch_idx: Optional[int] = None,
                                context: Optional[Dict[str, Any]] = None,
                                ):
        N = len(self.objects)
        if self.config.has_background_plane:
            N -= 1
        if self.config.has_camera_aberration_plane:
            N -= 1
        object_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        all_object_indices = self._node_indices.to(self.device)
        all_mask = torch.ones(len(all_object_indices),
                              dtype=torch.bool, device=self.device)
        all_mask[self._camera_index_in_node_indices] = False
        if self.config.has_background_plane:
            all_mask[self._background_index_in_node_indices] = False
        filtered_indices = all_object_indices[all_mask]
        needed_indices = filtered_indices[enabled_objects]
        # Compute global rays
        objects = [x for i, x in enumerate(self.objects) if (x.get_index() in needed_indices) or (
            enabled_background and isinstance(x, BackgroundPlaneSceneNode3D) or (enabled_aberration and isinstance(x, LearnedAberrationPlaneSceneNode3D)))]

        B = uv.shape[0]
        O = enabled_objects.sum()
        T = len(t)
        # Objects
        OB = O + int(self.config.has_background_plane and enabled_background)
        OBA = OB + \
            int(self.config.has_camera_aberration_plane and enabled_aberration)

        global_positions, local_plane_scale, local_plane_scale_offset, camera_idx = self._assemble_object_rigid_data(
            t, enabled_objects, enabled_background, enabled_aberration=enabled_aberration)

        global_ray_origins, global_ray_directions, global_positions, OB_mask, intersection_points, is_inside, _, _ = plane_hits(
            uv=uv,
            global_positions=global_positions,
            inverse_intrinsics=self.camera.get_inverse_intrinsics(t),
            lens_distortion=self.camera.get_lens_distortion(),
            camera_idx=camera_idx,
            focal_length=self.focal_length,
            local_plane_scale=local_plane_scale,
            local_plane_scale_offset=local_plane_scale_offset,
        )

        global_plane_positions = global_positions[OB_mask]

        colors = torch.zeros(
            (OBA, B, T, 3), device=uv.device, dtype=uv.dtype)
        alphas = torch.zeros(
            (OBA, B, T, 1), device=uv.device, dtype=uv.dtype)

        local_intersection_points = global_to_local(global_plane_positions, intersection_points.permute(1, 0, 2, 3).reshape(
            B, OBA * T, -1), v_include_time=True).reshape(B, OBA, T, -1).permute(1, 0, 2, 3)[..., :2]  # Shape (OBA, B, T, 2)
        plane_intersection_points = batched_local_to_plane_coordinates(local_intersection_points.reshape(
            OBA, B * T, -1).permute(1, 0, 2), local_plane_scale, local_plane_scale_offset).permute(1, 0, 2).reshape(OBA, B, T, -1)

        if context is not None:
            # Add to debug context
            if "intersection_points" not in context:
                context["intersection_points"] = []
            context["intersection_points"].append(
                plane_intersection_points)  # Shape (O, B, T, 2)
            if "is_inside" not in context:
                context["is_inside"] = []
            context["is_inside"].append(is_inside)  # Shape (O, B, T)

        # Convert Ray_directions to local
        ray_global_target = global_ray_origins + global_ray_directions

        for i, obj in enumerate(objects):
            # Trace each ray through the object if it is inside
            obj: torch.nn.Module
            ray_local_target = global_to_local(
                global_plane_positions[i], ray_global_target, v_include_time=True)[..., :3]
            ray_local_origin = global_to_local(
                global_plane_positions[i], global_ray_origins, v_include_time=True)[..., :3]
            ray_local_directions = ray_local_target - ray_local_origin

            colors[i], alphas[i] = obj(plane_intersection_points[i],
                                       is_inside=is_inside[i],
                                       ray_origins=ray_local_origin,
                                       ray_directions=ray_local_directions,
                                       t=t,
                                       uv_in_plane_space=True,
                                       sin_epoch=self.sin_epoch,
                                       global_position=global_plane_positions[i],
                                       plane_scale=local_plane_scale[i],
                                       plane_scale_offset=local_plane_scale_offset[i],
                                       next_sin_epoch=self.next_sin_epoch,
                                       batch_idx=batch_idx,
                                       max_batch_idx=self.num_batches,
                                       context=context
                                       )

        if color_composing:
            object_colors, object_assembled_alphas = compute_object_rgba(colors=colors, alphas=alphas, t=t,
                                                                         global_ray_origins=global_ray_origins, global_ray_directions=global_ray_directions,
                                                                         is_inside=is_inside, intersection_points=intersection_points)
            ray_color = (object_assembled_alphas * object_colors).sum(dim=0)
            return ray_color, object_assembled_alphas.sum(dim=0)
        else:
            # TODO This will not support actual composing of multiple objects right now. As alpha chain is not used.
            object_colors, object_assembled_alphas, order, alpha_chain, sorted_alphas = compute_object_rgba(colors=colors, alphas=alphas, t=t,
                                                                                                            global_ray_origins=global_ray_origins, global_ray_directions=global_ray_directions,
                                                                                                            is_inside=is_inside, intersection_points=intersection_points, get_object_alpha_chain=True)

            original_sorted_colors, _, unsorted_alpha_chain, unsorted_alphas = undo_intersection_ordering(
                order=order, sorted_color=object_colors, sorted_alpha_chain=alpha_chain, sorted_alpha=sorted_alphas)
            return original_sorted_colors, unsorted_alphas, unsorted_alpha_chain

    def object_modality_forward(self,
                                uv: torch.Tensor,
                                t: torch.Tensor,
                                enabled_objects: torch.Tensor,
                                enabled_background: bool = True,
                                color: bool = True,
                                alpha: bool = True,
                                flow: bool = True,
                                out_of_bounds_termination: bool = True,
                                lens_distortion_correction: bool = False,
                                batch_idx: Optional[int] = None,
                                context: Optional[Dict[str, Any]] = None,
                                ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        enabled_objects_cam = torch.cat(
            [enabled_objects, torch.tensor([True], dtype=torch.bool, device=uv.device)])

        enabled_objects_back = enabled_objects.clone()
        if self.config.has_background_plane:
            enabled_objects_back = torch.cat([enabled_objects_back, torch.tensor(
                [enabled_background], dtype=torch.bool, device=uv.device)])

        # Compute global rays

        objects = [x for i, x in enumerate(self.objects) if (i in enabled_objects.argwhere()) or (
            enabled_background and isinstance(x, BackgroundPlaneSceneNode3D))]

        B = uv.shape[0]
        O = enabled_objects.sum()
        T = len(t)
        # Objects
        OB = O + int(self.config.has_background_plane and enabled_background)
        if self.camera_idx != self._translations.shape[0] - 1:
            raise ValueError(
                "Camera index is not the last index in translations.")
        global_positions, local_plane_scale, local_plane_scale_offset, CIDX = self._assemble_object_rigid_data(
            t, enabled_objects, enabled_background)

        global_ray_origins, global_ray_directions, _, OB_mask, intersection_points, is_inside, _, _ = plane_hits(
            uv=uv,
            global_positions=global_positions,
            inverse_intrinsics=self.camera.get_inverse_intrinsics(t),
            lens_distortion=self.camera.get_lens_distortion(),
            camera_idx=CIDX,
            focal_length=self.focal_length,
            local_plane_scale=local_plane_scale,
            local_plane_scale_offset=local_plane_scale_offset,
        )

        global_plane_positions = global_positions[OB_mask]

        colors, alphas, flows = None, None, None

        if color:
            colors = torch.zeros(
                (OB, B, T, 3), device=uv.device, dtype=uv.dtype)
        if alpha:
            alphas = torch.zeros(
                (OB, B, T, 1), device=uv.device, dtype=uv.dtype)
        if flow:
            flows = torch.zeros(
                (OB, B, T, 2), device=uv.device, dtype=uv.dtype)

        local_intersection_points = global_to_local(global_plane_positions, intersection_points.permute(1, 0, 2, 3).reshape(
            B, OB * T, -1), v_include_time=True).reshape(B, OB, T, -1).permute(1, 0, 2, 3)[..., :2]  # Shape (OB, B, T, 2)
        plane_intersection_points = batched_local_to_plane_coordinates(local_intersection_points.reshape(
            OB, B * T, -1).permute(1, 0, 2), local_plane_scale, local_plane_scale_offset).permute(1, 0, 2).reshape(OB, B, T, -1)

        ray_global_target = global_ray_origins + global_ray_directions

        for i, obj in enumerate(objects):
            # Trace each ray through the object if it is inside
            obj: torch.nn.Module
            ray_local_target = global_to_local(
                global_plane_positions[i], ray_global_target, v_include_time=True)[..., :3]
            ray_local_origin = global_to_local(
                global_plane_positions[i], global_ray_origins, v_include_time=True)[..., :3]
            ray_local_directions = ray_local_target - ray_local_origin
            c, a, f = obj.forward_modality(plane_intersection_points[i],
                                           is_inside=is_inside[i],
                                           t=t,
                                           ray_origins=ray_local_origin,
                                           ray_directions=ray_local_directions,
                                           sin_epoch=self.sin_epoch,
                                           next_sin_epoch=self.next_sin_epoch,
                                           batch_idx=batch_idx,
                                           max_batch_idx=self.num_batches,
                                           query_color=color,
                                           query_alpha=alpha,
                                           context=context
                                           )
            if color:
                colors[i] = c
            if alpha:
                alphas[i] = a
            if flow:
                flows[i] = f
        if color and out_of_bounds_termination:
            colors[~is_inside] = 0.
        if alpha and out_of_bounds_termination:
            alphas[~is_inside] = 0.

        if flow:
            if out_of_bounds_termination:
                flows[~is_inside] = 0.

            # As flow is in the object space, we need to transform it to the camera space
            flow_end = plane_intersection_points + flows
            flow_end = batched_plane_coordinates_to_local(flow_end.reshape(OB, B * T, -1).permute(
                1, 0, 2), local_plane_scale, local_plane_scale_offset).permute(1, 0, 2).reshape(OB, B, T, -1)
            flow_end = local_to_global(global_plane_positions, flow_end.permute(1, 0, 2, 3).reshape(
                B, OB * T, -1), v_include_time=True).reshape(B, OB, T, -1).permute(1, 0, 2, 3)[..., :3]  # Shape (OB, B, T, 3)
            flow_end = flow_end.reshape(OB * B, T, 3)
            flow_end = global_to_local(
                global_positions[CIDX], flow_end, v_include_time=True)  # Shape (OB * B, T, 3)
            flow_end = local_to_image_coordinates(flow_end.permute(1, 0, 2), self.camera.get_intrinsics(
                t), self.camera.focal_length, self.camera.get_lens_distortion() if lens_distortion_correction else None, v_includes_time=True)  # Shape (T, OB * B, 2)
            flow_end = flow_end.permute(
                1, 0, 2).reshape(OB, B, T, 2)
            flow_end = flow_end - \
                uv.unsqueeze(0).unsqueeze(2).expand(OB, -1, T, -1)
            flow_end[~is_inside] = 0.
            flows = flow_end
        return colors, alphas, flows

    def get_node_indices(self) -> torch.Tensor:
        """Get the indices of the nodes in the model.

        Returns
        -------
        torch.Tensor
            The indices of the nodes in the model.
        """
        # OBJS = self.num_objects + 1  # Objects + Camera
        # if self.config.has_background_plane:
        #     OBJS += 1
        # if self.config.has_camera_aberration_plane:
        #     OBJS += 1
        # not_bg = [x.get_index() for x in self.objects if not isinstance(
        #     x, BackgroundPlaneSceneNode3D) and not isinstance(x, LearnedAberrationPlaneSceneNode3D)]
        # bg = [x.get_index() for x in self.objects if isinstance(
        #     x, BackgroundPlaneSceneNode3D)]
        # abr = [x.get_index() for x in self.objects if isinstance(
        #     x, LearnedAberrationPlaneSceneNode3D)]

        # cam = self.camera.get_index()

        # # t = torch.tensor(not_bg + [cam] + bg + abr, dtype=torch.int32)
        # order = torch.zeros(OBJS, dtype=torch.int32)
        # order.fill_(-1)
        # order[self.camera_idx] = cam

        # od = torch.ones_like(order, dtype=torch.bool)
        # od[order == cam] = False
        # aw = torch.argwhere((order != cam)).squeeze()
        # m = (aw == self.background_idx + 1)  # Disables BG index
        # od[aw[m]] = False

        # order[od] = torch.tensor(not_bg, dtype=order.dtype)

        # if len(abr) > 0:
        #     order[-1] = abr[0]

        # if len(bg) > 0:
        #     order[order == -1] = bg[0]
        # return order
        return self._node_indices

    def get_global_positions(self, t: torch.Tensor) -> torch.Tensor:
        """Get the global positions of all the objects and the camera at the given times.

        Parameters
        ----------
        t : torch.Tensor
            The times for which the global positions are computed. Shape (T,)

        Returns
        -------
        torch.Tensor
            A global position matrix of shape (N, T, 4, 4) where N is the number of objects + camera.
            Camera is at camera_idx, if has_background_plane is True, the background is at the last index.
        """
        right_idx, rel_frac, right_idx_offset, rel_frac_offset = self._get_interpolate_index(
            t)

        translations, orientations = get_translation_orientation(
            t=t,
            translations=self._translations,
            translation_offset=self.get_offset_translation(),
            orientations=self._orientation,
            rotation_offset=self.get_offset_rotation_vector(),
            translation_offset_weight=self._offset_translation_weight,
            rotation_offset_weight=self._offset_rotation_weight,
            times=self._times,
            offset_times=self._offset_times,
            equidistant_times=self._equidistant_times,
            equidistant_offset_times=True,
            right_idx=right_idx,
            rel_frac=rel_frac,
            right_idx_offset=right_idx_offset,
            rel_frac_offset=rel_frac_offset
        )

        global_positions = get_global_positions(
            t=t,
            translations=translations,
            orientations=orientations,
            times=self._times,
            equidistant_times=self._equidistant_times,
            camera_idx=self.camera_idx,
            background_idx=tensorify(
                self._background_index_in_node_indices) if self._background_index_in_node_indices is not None else None,
            has_background=self.config.has_background_plane,
            background_orientation=self._background_orientation,
            background_translation=self._background_translation,
            background_attached_to_camera=self._is_background_attached_to_camera,
            has_camera_aberration=self.config.has_camera_aberration_plane,
            camera_aberration_orientation=self._camera_aberration_orientation,
            camera_aberration_translation=self._camera_aberration_translation,
            right_idx=right_idx,
            rel_frac=rel_frac)
        return global_positions

    def enlarge_model(self, num_objects: int = 1) -> torch.Tensor:
        O = self.num_objects
        N = self.num_objects + 1  # OBJ + camera
        OB = O + (1 if self.config.has_background_plane else 0)
        OBA = OB + (1 if self.config.has_camera_aberration_plane else 0)
        OBAC = OBA + 1  # All objects + camera
        NN = N + num_objects
        NO = O + num_objects
        NOB = OB + num_objects
        NOBA = OBA + num_objects
        NOBAC = OBAC + num_objects
        T = self._times.shape[0]
        TC2 = self._offset_translation.shape[1]
        device = self._translations.device
        dtype = self._translations.dtype

        added_mask = torch.zeros(NOBAC, dtype=torch.bool)
        added_mask[-num_objects:] = True

        _new_translations = torch.zeros(NN, T, 3, device=device, dtype=dtype)
        _new_translations[:N] = self._translations.detach()
        _new_orientation = torch.zeros(NN, T, 4, device=device, dtype=dtype)
        _new_orientation[:N] = self._orientation.detach()
        _new_offset_translation = torch.zeros(
            NN, TC2, 3, device=device, dtype=dtype)
        _new_offset_translation[:N] = self._offset_translation.detach()
        _new_offset_rotation_vector = torch.zeros(
            NN, TC2, 3, device=device, dtype=dtype)
        _new_offset_rotation_vector[:N] = self._offset_rotation_vector.detach()
        _new_offset_translation_learned_mask = torch.ones(
            NN, dtype=torch.bool, device=device)
        _new_offset_translation_learned_mask[:N] = self._offset_translation_learned_mask.detach(
        )
        _new_offset_rotation_vector_learned_mask = torch.ones(
            NN, dtype=torch.bool, device=device)
        _new_offset_rotation_vector_learned_mask[:N] = self._offset_rotation_vector_learned_mask.detach(
        )
        _new_offset_translation_weight = torch.zeros(
            NN, device=device, dtype=dtype)
        _new_offset_translation_weight[:N] = self._offset_translation_weight.detach(
        )
        _new_offset_rotation_weight = torch.zeros(
            NN, device=device, dtype=dtype)
        _new_offset_rotation_weight[:N] = self._offset_rotation_weight.detach()
        _new_plane_scale = torch.zeros(NOBA, 2, device=device, dtype=dtype)
        _new_plane_scale[:OBA] = self._plane_scale.detach()
        _new_plane_scale_offset = torch.zeros(
            NOBA, 2, device=device, dtype=dtype)
        _new_plane_scale_offset[:OBA] = self._plane_scale_offset.detach()
        _indices = torch.full((NOBAC,), fill_value=-1, dtype=torch.int32)
        _indices[:OBAC] = self._node_indices.detach()

        self._translations.data = _new_translations
        self._orientation.data = _new_orientation
        self._offset_translation.data = _new_offset_translation
        self._offset_rotation_vector.data = _new_offset_rotation_vector
        self._offset_translation_learned_mask.data = _new_offset_translation_learned_mask
        self._offset_rotation_vector_learned_mask.data = _new_offset_rotation_vector_learned_mask
        self._offset_translation_weight.data = _new_offset_translation_weight
        self._offset_rotation_weight.data = _new_offset_rotation_weight
        self._plane_scale.data = _new_plane_scale
        self._plane_scale_offset.data = _new_plane_scale_offset
        self._node_indices.data = _indices
        self.num_objects += num_objects
        return added_mask

    def init_parameters(self, num_objects: int, times: torch.Tensor):
        N = num_objects + 1  # Number of objects + camera, objects may include background
        if self.config.has_background_plane:
            N -= 1
        if self.config.has_camera_aberration_plane:
            N -= 1

        self.num_objects = N - 1  # Subtract camera
        O = self.num_objects
        OB = O + (1 if self.config.has_background_plane else 0)
        OBA = OB + (1 if self.config.has_camera_aberration_plane else 0)

        T = len(times)

        self.camera_idx = N - 1
        self._camera_index_in_node_indices = N - 1

        TC = self.config.object_rigid_control_points if self.config.object_rigid_control_points is not None else int(round(len(
            times) * self.config.object_rigid_control_points_ratio))
        if len(times) == 1:
            TC = 1

        TC2 = TC + 2  # Control points + 2 for the endpoints

        indices = torch.full((OBA + 1,), fill_value=-1, dtype=torch.int32)
        self.register_buffer("_node_indices", indices)
        self.register_buffer("_translations", torch.zeros(N, T, 3))
        self.register_buffer("_orientation", torch.zeros(N, T, 4))
        self._orientation[..., 3] = 1.  # Set w to 1 => Valid quaternion

        self._offset_translation = torch.nn.Parameter(torch.zeros(N, TC2, 3))
        self._offset_rotation_vector = torch.nn.Parameter(
            torch.zeros(N, TC2, 3))

        # Mask for learned offsets, used to disable learning for some objects
        self.register_buffer("_offset_translation_learned_mask",
                             torch.ones(N, dtype=torch.bool))
        self.register_buffer(
            "_offset_rotation_vector_learned_mask", torch.ones(N, dtype=torch.bool))

        owp = tensorify(self.config.plane_rotation_offset_weight,
                        dtype=self.config.dtype)
        if owp.numel() == 1:
            owp = owp.expand(N)
        if owp.numel() == N - 1:
            owp = torch.cat(
                [owp, torch.tensor([0.], dtype=self.config.dtype)], dim=-1)
        owp[self.camera_idx] = self.config.camera_rotation_offset_weight
        # View to real tensor so loading works
        self.register_buffer("_offset_rotation_weight", owp.clone())

        twp = tensorify(
            self.config.plane_translation_offset_weight, dtype=self.config.dtype)
        if twp.numel() == 1:
            twp = twp.expand(N)
        if twp.numel() == N - 1:
            twp = torch.cat(
                [twp, torch.tensor([0.], dtype=self.config.dtype)], dim=-1)
        twp[self.camera_idx] = self.config.camera_translation_offset_weight
        # View to real tensor so loading works
        self.register_buffer("_offset_translation_weight", twp.clone())

        ps = default_plane_scale(self.config.dtype).repeat(OBA, 1)
        self.register_buffer("_plane_scale", ps)
        pso = default_plane_scale_offset(self.config.dtype).repeat(OBA, 1)
        self.register_buffer("_plane_scale_offset", pso)

        # Positions for the background
        if self.config.has_background_plane:
            self.register_buffer("_background_translation", torch.zeros(T, 3))
            self.register_buffer("_background_orientation", torch.zeros(T, 4))
            self.register_buffer(
                "_is_background_attached_to_camera", torch.tensor(True))
            self._background_orientation[..., 3] = 1.
            self.background_idx = OB - 1
            # Equiv to one behind cam.
            self._background_index_in_node_indices = OB
        else:
            self.background_idx = None
            self._background_index_in_node_indices = None
            self._background_orientation = None
            self._background_translation = None
            self._is_background_attached_to_camera = None

        # Positions for the camera aberration plane (if enabled)
        if self.config.has_camera_aberration_plane:
            self.register_buffer(
                "_camera_aberration_translation", torch.zeros(T, 3))
            self.register_buffer(
                "_camera_aberration_orientation", torch.zeros(T, 4))
            self._camera_aberration_orientation[..., 3] = 1.
            self.camera_aberration_idx = OBA - 1
        else:
            self.camera_aberration_idx = None
            self._camera_aberration_orientation = None
            self._camera_aberration_translation = None

        self.register_times(times)
        self.set_offset_times(TC)

        self.register_buffer("_interpolated_steps", torch.empty(0))
        self.register_buffer("_right_idx", torch.empty(0))
        self.register_buffer("_rel_frac", torch.empty(0))
        self.register_buffer("_right_idx_offset", torch.empty(0))
        self.register_buffer("_rel_frac_offset", torch.empty(0))

    def register_times(self, value: torch.Tensor):
        if len(value.shape) != 1:
            raise ValueError("Times must be a 1D tensor.")
        self.register_buffer("_times", value)

        # Check if times are equally spaced
        eq = self._equidistant_times
        delta_t = torch.diff(self._times, dim=-1)
        self._equidistant_times = torch.allclose(
            delta_t, self._times[..., 1] - self._times[..., 0], atol=1e-4) if len(self._times) > 1 else True
        # Warn if times are not equally spaced
        if eq != self._equidistant_times and not self._equidistant_times:
            global NON_EQUAL_TIME_STEPS_WARNED
            if not NON_EQUAL_TIME_STEPS_WARNED:
                long = delta_t.shape[0] > 10
                tc = delta_t[:10] if long else delta_t
                logger.warning(
                    f"Times are not equally spaced. This can slow down interpolation.\nΔt: [{', '.join(['{:.4f}'.format(t.item()) for t in tc]) + (', ...' if long else '')}]")
                NON_EQUAL_TIME_STEPS_WARNED = True

    def set_offset_times(self, num_control_points: int):
        ot = default_offset_times(num_control_points, dtype=self.config.dtype)
        self.register_buffer("_offset_times", ot)

    def enable_position_learning(self, value: bool):
        """Enable or disable the position learning in the objects."""
        self._offset_translation.requires_grad_(value)
        self._offset_rotation_vector.requires_grad_(value)

    def get_offset_translation(self) -> torch.Tensor:
        if self._offset_translation_learned_mask.all():
            return self._offset_translation
        else:
            trans = torch.zeros(self._offset_translation_learned_mask.shape + self._offset_translation.shape[1:],
                                dtype=self._offset_translation.dtype,
                                device=self.device)
            trans[self._offset_translation_learned_mask] = self._offset_translation[self._offset_translation_learned_mask]
            return trans

    def get_offset_rotation_vector(self) -> torch.Tensor:
        if self._offset_rotation_vector_learned_mask.all():
            return self._offset_rotation_vector
        else:
            rot = torch.zeros(self._offset_rotation_vector_learned_mask.shape + self._offset_rotation_vector.shape[1:],
                              dtype=self._offset_rotation_vector.dtype,
                              device=self.device)
            rot[self._offset_rotation_vector_learned_mask] = self._offset_rotation_vector[self._offset_rotation_vector_learned_mask]
            return rot

    # Generation Functions

    def _generate_modality_outputs(self,
                                   t: torch.Tensor,
                                   device: torch.device,
                                   config: NAGConfig,
                                   resolution: torch.Tensor,
                                   object_mask: torch.Tensor,
                                   enabled_background: bool = True,
                                   query_color: bool = True,
                                   query_alpha: bool = True,
                                   query_flow: bool = True,
                                   lens_distortion_correction: bool = False,
                                   progress_bar: bool = True,
                                   progress_factory: Optional[ProgressFactory] = None,
                                   ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad(), TemporaryTraining(self, False):
            OT = len(t)
            org_t = t
            t, flow_added_times = self._get_t_with_flow_reference_times(t)
            flow_added_times_cpu = flow_added_times.cpu()

            uv_max = self.camera._image_resolution.flip(-1).detach().cpu()
            gs = RegularUVGridSampler(
                resolution=resolution, uv_max=uv_max, inter_pixel_noise_fnc=None,
                t=t, config=config,
                max_total_batch_size=config.max_total_batch_size_inference
            )

            color, subsample, subsample_offsets = gs.get_proto_image_tensor(
                t=t)
            alpha, flow = None, None

            OB = object_mask.sum() + int(enabled_background)
            T, C, H, W = color.shape
            if query_color:
                color = color.unsqueeze(0).repeat(OB, 1, 1, 1, 1)[
                    :, ~flow_added_times_cpu]
            else:
                color = None
            if query_flow:
                flow = torch.zeros((OB, OT, 2, H, W), dtype=t.dtype)

            if query_alpha:
                alpha = torch.zeros((OB, OT, 1, H, W), dtype=t.dtype)

            bar = None
            if progress_bar:
                bar = progress_factory.bar(total=len(
                    gs), desc="Generating Modality outputs", tag="nag_model_generate_outputs", delay=2, is_reusable=True)

            for i in range(len(gs)):
                uv, t = gs[i]
                uv = uv.to(device=device)
                t = t.to(device=device)
                SH, SW = gs.get_batch_shape(i)
                _color, _alpha, _flow = self.object_modality_forward(uv, t, object_mask, enabled_background, color=query_color, alpha=query_alpha,
                                                                     flow=query_flow, out_of_bounds_termination=True, lens_distortion_correction=lens_distortion_correction)
                # Outputs are OB, B, T, C
                if query_color:
                    color[:, :, :,
                          subsample_offsets[i, 1]::subsample[1],
                          subsample_offsets[i, 0]::subsample[0]] = _color.permute(0, 2, 3, 1).reshape(OB, T, 3, SH, SW)[:, ~flow_added_times].cpu()
                if query_alpha:
                    alpha[:, :, :,
                          subsample_offsets[i, 1]::subsample[1],
                          subsample_offsets[i, 0]::subsample[0]] = _alpha.permute(0, 2, 3, 1).reshape(OB, T, 1, SH, SW)[:, ~flow_added_times].cpu()
                if query_flow:
                    flow[:, :, :,
                         subsample_offsets[i, 1]::subsample[1],
                         subsample_offsets[i, 0]::subsample[0]] = _flow.permute(0, 2, 3, 1).reshape(OB, T, 2, SH, SW)[:, ~flow_added_times].cpu()
                if progress_bar:
                    bar.update()
            return color, alpha, flow

    def generate_modality_outputs(self,
                                  config: NAGConfig,
                                  t: Optional[Union[VEC_TYPE,
                                                    NUMERICAL_TYPE]] = None,
                                  resolution: Optional[VEC_TYPE] = None,
                                  objects: Optional[List[TimedPlaneSceneNode3D]] = None,
                                  query_color: bool = True,
                                  query_alpha: bool = True,
                                  query_flow: bool = True,
                                  lens_distortion_correction: bool = False,
                                  progress_bar: bool = True,
                                  progress_factory: Optional[ProgressFactory] = DEFAULT,
                                  device: Optional[torch.device] = None) -> Generator[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]], None, None]:
        # Create a UV Grid sampler
        if objects is None:
            objects = list(self.objects)

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        if progress_bar:
            if progress_factory is None:
                progress_factory = ProgressFactory()
            if progress_factory == DEFAULT:
                progress_factory = config.progress_factory

        resolution = tensorify(resolution).detach().cpu(
        ) if resolution is not None else None
        if resolution is None:
            resolution = self.camera._image_resolution.flip(-1).detach().cpu()

        t = tensorify(t, dtype=self.camera._translation.dtype,
                      device=self.camera._translation.device) if t is not None else self.camera._times
        if len(t.shape) == 0:
            t = t.unsqueeze(0)

        object_mask = None
        enabled_background = False

        object_mask = torch.zeros(
            (len(self.objects) - 1), dtype=torch.bool, device=device)
        for i, obj in enumerate(objects):
            if isinstance(obj, BackgroundPlaneSceneNode3D):
                enabled_background = True
                background_idx = torch.tensor(next((i for i, x in enumerate(
                    self.objects) if isinstance(x, BackgroundPlaneSceneNode3D))), device=device)
                continue
            idx = self.objects.index(obj)
            object_mask[idx] = True
        if not object_mask.any() and not enabled_background:
            raise ValueError(
                "No objects specified to generate outputs for.")

        with (TemporaryDevice(self, device) as dm, TemporaryTraining(self, False), torch.no_grad()):
            # Iterate over the grid sampler
            return batched_generator_exec(batched_params=["t"], default_batch_size=1)(self._generate_modality_outputs)(t=t,
                                                                                                                       device=dm.device,
                                                                                                                       config=config,
                                                                                                                       resolution=resolution,
                                                                                                                       object_mask=object_mask,
                                                                                                                       enabled_background=enabled_background,
                                                                                                                       lens_distortion_correction=lens_distortion_correction,
                                                                                                                       query_color=query_color,
                                                                                                                       query_alpha=query_alpha,
                                                                                                                       query_flow=query_flow,
                                                                                                                       progress_bar=progress_bar, progress_factory=progress_factory)

    @batched_generator_exec(batched_params=["t"], default_batch_size=1)
    def query(
        self,
        t: torch.Tensor,
        config: Optional[NAGConfig] = None,
        resolution: Optional[VEC_TYPE] = None,
        **kwargs
    ) -> SizedGenerator[torch.Tensor, None, None]:
        """
        Generates images of the encoded scene at respective timestamps.
        Yields a sized generator of images, where each image has shape [T, C, H, W].    

        Parameters
        ----------
        
        t: torch.Tensor
            Time steps to evaluate the scene at. Shape (T, ).
        
        config: NAGConfig
            Configuration for the NAG model.
            Can be used to override certain settings during generation.
            By default, uses self.config.

        resolution: Optional[VEC_TYPE]
            Resolution of the generated images. Shape (2, ).

        progress_bar: bool
            If a progress bar should be shown during generation.

        progress_factory: Optional[ProgressFactory]
            Factory to create progress bars. If None, uses the default factory.
        
        device: Optional[torch.device]
            Device to generate the images on. If None, uses the current device of the model.

        Returns
        -------
        torch.Tensor
            Generated images. Shape [T, C, H, W]
        """
        if config is None:
            config = self.config
        if resolution is None:
            resolution = self.camera._image_resolution.flip(-1).detach().cpu()
        return self.generate_outputs(
            config=config,
            t=t,
            resolution=resolution,
            **kwargs
        )

    # endregion

    # region Patching

    def _patch_general_args(self, object_idx: int, args: Dict[str, Any], is_camera: bool = False) -> Dict[str, Any]:
        from tools.util.torch import tensorify
        object_name = args.get("name", f"Object {object_idx}")

        # Set index within self
        # Set to the next free index in _node_indices
        if is_camera:
            set_idx = self.camera_idx
            self._node_indices[set_idx] = args.get("index")
        else:
            set_idx = (self._node_indices == -
                       1).argwhere().squeeze(0)[0].item()
            self._node_indices[set_idx] = tensorify(args.get(
                "index", -1), dtype=self._node_indices.dtype, device=self.device)
            # Ommiting background if needed.
            set_idx = self.get_oid_to_positions_mask(
                self._node_indices[set_idx]).argwhere().squeeze(0).item()

        if "position" in args:
            raise ValueError(
                "Position can not be set when translations and orientations are just references.")
        if "translation" in args:
            # Set the data
            self._translations.data[set_idx] = args.pop(
                "translation").clone().detach()
        # Patch the args
        args["_translation"] = self._translations[set_idx]
        if "orientation" in args:
            # Set the data
            self._orientation.data[set_idx] = args.pop(
                "orientation").clone().detach()
            # Patch the args
        args["_orientation"] = self._orientation[set_idx]
        if "times" in args:
            # Set the data
            t = args.pop("times").clone().detach()
            if t.shape != self._times.shape or not torch.allclose(t, self._times):
                # Warn if the times are changed
                logger.warning(
                    f"Changing the times of the {object_name} from Shape {t.shape} to {self._times.shape} in patching.")
        args["_times"] = self._times
        return args

    def _patch_object_args(self, object_idx: int, args: Dict[str, Any], is_camera: bool = False) -> Dict[str, Any]:
        proxy_init = args.get("proxy_init", False)
        object_name = args.get("name", f"Object {object_idx}")
        args = self._patch_general_args(object_idx, args, is_camera=is_camera)

        real_object_idx = object_idx
        if is_camera:
            object_idx = self.camera_idx
        else:
            object_idx = self.get_oid_to_positions_mask(
                object_idx).argwhere().squeeze(0).item()

        if "translation_offset_weight" in args:
            # Set the data
            d = tensorify(args.pop("translation_offset_weight")
                          ).clone().detach()
            new_weight = self._offset_translation_weight[object_idx]
            if d != new_weight:
                # Warn if the weight is changed
                logger.warning(
                    f"Changing the translation offset weight of {object_name} from {new_weight} to {d} in patching.")
        args["_translation_offset_weight"] = self._offset_translation_weight[object_idx]
        if "rotation_offset_weight" in args:
            # Set the data
            d = tensorify(args.pop("rotation_offset_weight")).clone().detach()
            new_weight = self._offset_rotation_weight[object_idx]
            if d != new_weight:
                # Warn if the weight is changed
                logger.warning(
                    f"Changing the rotation offset weight of {object_name} from {new_weight} to {d} in patching.")
        args["_rotation_offset_weight"] = self._offset_rotation_weight[object_idx]
        args["_offset_times"] = self._offset_times
        args["_offset_translation"] = self._offset_translation[object_idx]
        self._offset_translation_learned_mask[object_idx] = args.pop(
            "learnable_translation", True)
        args["learnable_translation"] = self._offset_translation_learned_mask[object_idx]
        args["_offset_rotation_vector"] = self._offset_rotation_vector[object_idx]
        self._offset_rotation_vector_learned_mask[object_idx] = args.pop(
            "learnable_rotation", True)
        args["learnable_rotation"] = self._offset_rotation_vector_learned_mask[object_idx]
        return args

    def patch_plane_args(self, object_idx: int, args: Dict[str, Any]) -> Dict[str, Any]:
        # Log info that the plane is patched
        args = self._patch_object_args(object_idx, args)
        real_object_idx = object_idx
        object_idx = self.get_oid_to_plane_mask(
            object_idx).argwhere().squeeze(0).item()
        if "plane_scale" in args:
            # Set the data
            self._plane_scale.data[object_idx] = args.pop(
                "plane_scale").clone().detach()
        args["_plane_scale"] = self._plane_scale[object_idx]
        if "plane_scale_offset" in args:
            # Set the data
            self._plane_scale_offset.data[object_idx] = args.pop(
                "plane_scale_offset").clone().detach()
        args["_plane_scale_offset"] = self._plane_scale_offset[object_idx]
        return args

    def patch_camera_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # Log info that the camera is patched
        args = self._patch_object_args(args.get("index"), args, is_camera=True)
        return args

    # def patch_object_args(self, object_idx: int, args: Dict[str, Any]) -> Dict[str, Any]:
    #     # Log info that the object is patched
    #     oidx = self.object_idx[object_idx]
    #     args = self._patch_object_args(oidx, args)
    #     return args

    def patch_background_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # Log info that the background is patched
        object_name = args.get("name", f"Background")
        bidx = self._node_indices[self._background_index_in_node_indices]
        idx = args.get("index", -1)
        if bidx != -1:
            raise ValueError(
                f"Background plane is already registered with index {bidx}.")
        self._node_indices[self._background_index_in_node_indices] = tensorify(
            idx, dtype=self._node_indices.dtype, device=self.device)

        if "is_camera_attached" in args:
            # Set the data
            self._is_background_attached_to_camera.data = tensorify(args.get(
                "is_camera_attached", True), dtype=torch.bool, device=self.device)

        if "position" in args:
            raise ValueError(
                "Position can not be set when translations and orientations are just references.")
        if "translation" in args:
            # Set the data
            t = args.pop("translation").clone().detach()
            if t.shape[0] == 1:
                # Repeat along time axis
                t = t.repeat(self._times.shape[0], 1)
            self._background_translation.data = t
            # Patch the args
        args["_translation"] = self._background_translation
        if "orientation" in args:
            # Set the data
            t = args.pop("orientation").clone().detach()
            if t.shape[0] == 1:
                # Repeat along time axis
                t = t.repeat(self._times.shape[0], 1)
            self._background_orientation.data = t
            # Patch the args
        args["_orientation"] = self._background_orientation
        if "times" in args:
            # Set the data
            t = args.pop("times").clone().detach()
            if t.shape != self._times.shape or not torch.allclose(t, self._times):
                # Warn if the times are changed
                logger.warning(
                    f"Changing the times of the {object_name} from Shape {t.shape} to {self._times.shape} in patching.")
        args["_times"] = self._times

        object_idx = self.get_oid_to_plane_mask(
            self._node_indices[self._background_index_in_node_indices]).argwhere().squeeze(0).item()

        if "plane_scale" in args:
            # Set the data
            self._plane_scale.data[object_idx] = args.pop(
                "plane_scale").clone().detach()
        args["_plane_scale"] = self._plane_scale[object_idx]
        if "plane_scale_offset" in args:
            # Set the data
            self._plane_scale_offset.data[object_idx] = args.pop(
                "plane_scale_offset").clone().detach()
        args["_plane_scale_offset"] = self._plane_scale_offset[object_idx]
        return args

    def patch_aberration_plane_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # Log info that the aberration plane is patched
        object_idx = self.camera_aberration_idx
        object_name = args.get("name", f"Aberration Plane")

        if "position" in args:
            raise ValueError(
                "Position can not be set when translations and orientations are just references.")
        if "translation" in args:
            # Set the data
            t = args.pop("translation").clone().detach()
            if t.shape[0] == 1:
                # Repeat along time axis
                t = t.repeat(self._times.shape[0], 1)
            self._camera_aberration_translation.data = t
            # Patch the args
        args["_translation"] = self._camera_aberration_translation
        if "orientation" in args:
            # Set the data
            t = args.pop("orientation").clone().detach()
            if t.shape[0] == 1:
                # Repeat along time axis
                t = t.repeat(self._times.shape[0], 1)
            self._camera_aberration_orientation.data = t
            # Patch the args
        args["_orientation"] = self._camera_aberration_orientation
        if "times" in args:
            # Set the data
            t = args.pop("times").clone().detach()
            if t.shape != self._times.shape or not torch.allclose(t, self._times):
                # Warn if the times are changed
                logger.warning(
                    f"Changing the times of the {object_name} from Shape {t.shape} to {self._times.shape} in patching.")
        args["_times"] = self._times

        if "plane_scale" in args:
            # Set the data
            self._plane_scale.data[object_idx] = args.pop(
                "plane_scale").clone().detach()
        args["_plane_scale"] = self._plane_scale[object_idx]
        if "plane_scale_offset" in args:
            # Set the data
            self._plane_scale_offset.data[object_idx] = args.pop(
                "plane_scale_offset").clone().detach()
        args["_plane_scale_offset"] = self._plane_scale_offset[object_idx]
        return args

    def push_to_objects(self) -> None:
        """
        Updates the local tensor values of positions and other modalities which are centerally managed (to allow speed up using batching)
        on the respective nodes. Nodes will afterwards have a view on the parameters.
        """
        with torch.no_grad():
            if (self._node_indices == -1).all():
                raise ValueError(
                    "No objects have been registered. Please register the objects before pushing them to the scene.")

            objects = list(self.objects) + [self.camera]

            for i, obj in enumerate(objects):
                obj: ModuleSceneNode3D
                idx = obj.get_index()
                pos_idx = self.get_oid_to_positions_mask(idx)
                if isinstance(obj, TimedDiscreteSceneNode3D):
                    if not isinstance(obj, BackgroundPlaneSceneNode3D) and not isinstance(obj, LearnedAberrationPlaneSceneNode3D):
                        obj._translation.data = self._translations[pos_idx].squeeze(
                            0)
                        obj._orientation.data = self._orientation[pos_idx].squeeze(
                            0)
                    elif isinstance(obj, LearnedAberrationPlaneSceneNode3D):
                        obj._translation.data = self._camera_aberration_translation
                        obj._orientation.data = self._camera_aberration_orientation
                    else:
                        obj._translation.data = self._background_translation
                        obj._orientation.data = self._background_orientation
                    obj._times.data = (self._times).detach().clone()
                    obj._equidistant_times = self._equidistant_times
                    obj._interpolation = "cubic"
                if isinstance(obj, LearnedOffsetSceneNode3D):
                    obj._offset_translation.data = self._offset_translation[pos_idx].squeeze(
                        0)
                    obj._offset_rotation_vector.data = self._offset_rotation_vector[pos_idx].squeeze(
                        0)
                    obj._offset_times.data = self._offset_times
                    obj._translation_offset_weight.data = self._offset_translation_weight[pos_idx].squeeze(
                        0)
                    obj._rotation_offset_weight.data = self._offset_rotation_weight[pos_idx].squeeze(
                        0)
                    obj._is_translation_learnable = self._offset_translation_learned_mask[pos_idx].squeeze(
                        0)
                    obj._is_rotation_learnable = self._offset_rotation_vector_learned_mask[
                        pos_idx].squeeze(0)
                if isinstance(obj, TimedPlaneSceneNode3D):
                    cidx = self.get_oid_to_plane_mask(
                        obj.get_index()).argwhere().squeeze(0).item()
                    obj._plane_scale.data = self._plane_scale[cidx]
                    obj._plane_scale_offset.data = self._plane_scale_offset[cidx]
                pass

    # endregion

    # region Plotting

    def _plot_object_positions(
            self,
            objects_mapping: Mapping[TimedDiscreteSceneNode3D, int],
            positions: torch.Tensor,
            times: torch.Tensor,
            offset_times: torch.Tensor,
            t: torch.Tensor,
            size: float = 5,
            view_points: Dict[TimedDiscreteSceneNode3D,
                              Literal["camera", "world"]] = None,
            **kwargs) -> Figure:
        from tools.viz.matplotlib import get_mpl_figure
        from nag.model.learned_offset_scene_node_3d import plot_position
        if view_points is None:
            view_points = dict()
        cols = 2
        objects = objects_mapping.keys()
        rows = len(objects)
        fig = plt.figure(figsize=(size * cols, size * rows))

        for i, obj in enumerate(objects):
            idx = objects_mapping[obj].item()
            ax1 = fig.add_subplot(rows, 2, 2*i+1)
            ax2 = fig.add_subplot(rows, 2, 2*i+2)

            plot_position(positions[idx], times,
                          offset_times,
                          ax=[ax1, ax2], t=t, **kwargs)

            axw = fig.add_subplot(rows, 1, 1 + i, frameon=False)
            axw.set_title("View " + view_points.get(obj,
                          "world").capitalize() + ": " + obj.get_name())
            axw.axis("off")
        return fig

    @saveable(default_dpi=150)
    def plot_object_positions(self,
                              object_idx: Optional[List[int]] = None,
                              t: Optional[torch.Tensor] = None,
                              size: float = 5,
                              view_point: Literal["camera",
                                                  "world"] = "camera",
                              **kwargs):
        import matplotlib.pyplot as plt
        from tools.viz.matplotlib import get_mpl_figure
        from nag.model.timed_discrete_scene_node_3d import plot_position
        objects = []
        if object_idx is None:
            objects = self.objects + [self.camera]
        else:
            objects = [self.objects[i] for i in object_idx]
        if t is None:
            t = self._times
        rows = len(objects)
        size = 5
        cols = 2

        nodes = self.get_node_indices()
        camera = self.camera
        nodes_lookup = {n: torch.argwhere(
            nodes == n.get_index()) for n in objects}
        with torch.no_grad():
            position = self.get_global_positions(t)

        view_points = dict()

        if view_point == "camera":
            camera = self.camera
            cam_idx = self.camera_idx
            mat = torch.ones_like(position[:, 0, 0, 0]).bool()
            mat[cam_idx] = False
            position[mat] = global_to_local_mat(
                position[~mat], position[mat], other_include_time=True)
            view_points = {n: ("camera" if n != camera else "world")
                           for n in objects}

        return self._plot_object_positions(nodes_lookup, position, self._times, self._offset_times, t=t, size=size,
                                           view_points=view_points, **kwargs)

    @saveable(default_name="plot_objects_positions", is_figure_collection=True, default_dpi=150)
    def plot_objects_positions(self,
                               object_idx: List[List[int]],
                               t: Optional[torch.Tensor] = None,
                               size: float = 5,
                               view_point: Literal["camera",
                                                   "world"] = "camera",
                               **kwargs) -> List[Figure]:
        """Plot object positions for multiple objects in multiple figures.

        Parameters
        ----------
        object_idx : List[List[int]]
            List of lists of object indices to plot.

        t : Optional[torch.Tensor], optional
            The timesteps where the positions should be plotted, by default None
        size : float, optional
            Size in inches of every graph, by default 5

        Returns
        -------
        List[Figure]
            List of figures corresponding to the object indices.
        """
        import matplotlib.pyplot as plt
        from tools.viz.matplotlib import get_mpl_figure
        from nag.model.timed_discrete_scene_node_3d import plot_position
        objects = []
        nodes = self.get_node_indices()
        index_object_mapping = self.get_index_object_mapping()
        figs = []
        if t is None:
            t = self._times
        with torch.no_grad():
            position = self.get_global_positions(t).cpu()
        size = 5
        times = self._times.detach().cpu()
        offset_times = self._offset_times.detach().cpu()
        t = t.cpu()

        view_points = dict()

        if view_point == "camera":
            camera = self.camera
            cam_idx = self.camera_idx
            mat = torch.ones_like(position[:, 0, 0, 0]).bool()
            mat[cam_idx] = False
            position[mat] = global_to_local_mat(
                position[~mat], position[mat], other_include_time=True)

        for olist in object_idx:
            objects = [index_object_mapping[i] for i in olist]
            view_points = {n: ("camera" if n != camera else "world")
                           for n in objects}
            nodes_lookup = {n: torch.argwhere(
                nodes == n.get_index()) for n in objects}
            fig = self._plot_object_positions(
                nodes_lookup, position, times, offset_times, t, size=size, view_points=view_points, **kwargs)
            figs.append(fig)
        return figs
    # endregion
