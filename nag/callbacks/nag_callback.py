import math
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import numpy as np
import os

from nag.config.nag_config import NAGConfig
from nag.dataset.nag_dataset import NAGDataset
from nag.model.background_image_plane_scene_node_3d import BackgroundImagePlaneSceneNode3D
from nag.model.background_plane_scene_node_3d import BackgroundPlaneSceneNode3D
from nag.model.background_image_plane_scene_node_3d import BackgroundImagePlaneSceneNode3D
from nag.model.learned_camera_scene_node_3d import LearnedCameraSceneNode3D
from nag.model.nag_functional_model import simple_zorder
from nag.model.nag_model import NAGModel
from nag.model.view_dependent_image_plane_scene_node_3d import ViewDependentImagePlaneSceneNode3D
from nag.model.view_dependent_background_image_plane_scene_node_3d import ViewDependentBackgroundImagePlaneSceneNode3D
from tools.metric.metric import Metric
from tools.util.format import parse_type
import torch
import pytorch_lightning as pl
from tools.util.numpy import numpyify
from tools.util.format import get_leading_zeros_format_string
from tools.util.progress_factory import ProgressFactory
from tools.util.torch import tensorify
from tools.run.config_runner import ConfigRunner
from tools.transforms.geometric.transforms3d import flatten_batch_dims, rotmat_to_unitquat
from tools.agent.util.tracker import Tracker
from tools.logger.logging import logger as logging
from tools.transforms.numpy.min_max import MinMax
from tools.io.image import put_text, linear_segmented_smoothing, n_layers_alpha_compositing, load_image_stack_generator, save_image_stack, save_image
from tools.video.utils import write_mp4_generator
from tools.util.path_tools import replace_unallowed_chars
from tools.model.module_scene_node_3d import ModuleSceneNode3D
from matplotlib import pyplot as plt
from nag.model.learned_aberration_plane_scene_node_3d import LearnedAberrationPlaneSceneNode3D
from typing import List
import torch
import gc
from tools.util.format import parse_format_string
from tools.util.path_tools import replace_unallowed_chars
from tools.transforms.to_numpy_image import ToNumpyImage
import numpy as np
from nag.utils.viz import flow_to_color
from nag.model.timed_plane_scene_node_3d import get_linear_segmented_smoothing_fnc
import pandas as pd


def generate_batched_outputs_and_save(
        model: NAGModel,
        resolution: np.ndarray,
        t: torch.Tensor,
        t_indices: torch.Tensor,
        t_real_indices: torch.Tensor,
        batch_size: int = 10,
        save_format_path: Optional[str] = None,
        return_images: bool = False,
        progress_bar: bool = False,
        progress_factory: Optional[ProgressFactory] = None,
        disable_view_dependency: bool = False,
        disable_camera_aberration: bool = False
) -> Union[List[str], Tuple[List[str], torch.Tensor]]:
    if save_format_path is None:
        save_format_path = os.path.join(model.config.output_path,
                                        "in_training",
                                        "{epoch:02d}_epoch_{t}_t_{idx}_idx_full.png")
    old_view_dependency_weights = dict()
    try:
        # If view dependency is disabled, we need to disable it
        if disable_view_dependency:
            for o in model.objects:
                if isinstance(o, (ViewDependentImagePlaneSceneNode3D, ViewDependentBackgroundImagePlaneSceneNode3D)):
                    old_view_dependency_weights[o] = o.view_dependence_weight.detach(
                    ).clone()
                    o.view_dependence_weight[...] = 0.0
        if disable_camera_aberration:
            for o in model.objects:
                if isinstance(o, LearnedAberrationPlaneSceneNode3D):
                    o.visible = False

        fnc = model.generate_outputs
        from tools.util.torch import batched_generator_exec
        batch_fnc = batched_generator_exec(batched_params=['t'],
                                           default_batch_size=batch_size,
                                           default_multiprocessing=False)(fnc)
        paths = []
        it = None
        if progress_bar:
            if progress_factory is None:
                progress_factory = ProgressFactory()
            it = progress_factory.bar(
                total=math.ceil(len(t_indices) / batch_size), desc="Generating images", is_reusable=True, tag="generate_images")
        lead_zeros_fmt = get_leading_zeros_format_string(
            t_indices.amax().item())
        real_lead_zeros_fmt = get_leading_zeros_format_string(
            t_real_indices.amax().item())

        images = None
        if return_images:
            images = torch.zeros(len(
                t_indices), 3, resolution[1], resolution[0], device=model.device, dtype=torch.uint8)

        for i, img_stack_out in enumerate(batch_fnc(model.config, resolution=resolution, t=t, progress_bar=progress_bar, progress_factory=progress_factory)):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(t_indices))
            t_cur_idx = t_indices[start_idx:end_idx]
            t_real_idx = t_real_indices[start_idx:end_idx]
            img_stack_out = (img_stack_out * 255).clamp(0, 255).byte()

            out_paths = save_image_stack(img_stack_out,
                                         save_format_path,
                                         additional_filename_variables_list=[
                                             {"t": lead_zeros_fmt.format(int(k)), "idx": real_lead_zeros_fmt.format(t_real_idx[j])} for j, k in enumerate(t_cur_idx)],
                                         additional_filename_variables={
                                             "epoch": model.current_epoch},
                                         override=True)
            if return_images:
                if img_stack_out.device != images.device:
                    images = images.to(img_stack_out.device)
                images[start_idx:end_idx] = img_stack_out

            paths.extend(out_paths)
            if it is not None:
                it.update()
    finally:
        # If view dependency is disabled, restore the old weights
        if disable_view_dependency:
            for o, w in old_view_dependency_weights.items():
                o.view_dependence_weight[...] = w

        if disable_camera_aberration:
            for o in model.objects:
                if isinstance(o, LearnedAberrationPlaneSceneNode3D):
                    o.visible = True

    if return_images:
        return paths, images
    return paths


def generate_batched_outputs_per_object_and_save(
    model: NAGModel,
    resolution: np.ndarray,
    t: torch.Tensor,
    t_indices: torch.Tensor,
    t_real_indices: torch.Tensor,
    objects: Optional[List[ModuleSceneNode3D]] = None,
    batch_size: int = 5,
    progress_bar: bool = False,
    progress_factory: Optional[ProgressFactory] = None
):
    fnc = model.generate_outputs
    from tools.util.torch import batched_generator_exec
    batch_fnc = batched_generator_exec(batched_params=['t'],
                                       default_batch_size=batch_size,
                                       default_explicit_garbage_collection=True,
                                       default_multiprocessing=False)(fnc)

    bar = None
    if objects is None:
        objects = [o for o in model.objects if (not isinstance(
            o, BackgroundPlaneSceneNode3D) or isinstance(o, BackgroundImagePlaneSceneNode3D))]

    if progress_bar:
        if progress_factory is None:
            progress_factory = ProgressFactory()
        bar = progress_factory.bar(total=(
            len(objects) * math.ceil(len(t) / batch_size)), desc="Saving object, images", is_reusable=True, tag="save_objects")

    lead_zeros_fmt = get_leading_zeros_format_string(t_indices.amax().item())
    real_lead_zeros_fmt = get_leading_zeros_format_string(
        t_real_indices.amax().item())

    object_images_paths = {i: [] for i in range(len(objects))}

    # Save the images to disk
    for i, img_stack_out in enumerate(batch_fnc(model.config,
                                      resolution=resolution,
                                      t=t,
                                      objects=[[o] for o in objects],
                                      progress_bar=progress_bar, progress_factory=progress_factory)):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(t_indices))
        t_cur_idx = t_indices[start_idx:end_idx]
        img_stack_out = (img_stack_out * 255).clamp(0, 255).byte()
        t_real_idx = t_real_indices[start_idx:end_idx]

        for i, o in enumerate(objects):
            sub_dir_path = get_leading_zeros_format_string(
                len(objects)).format(i)
            name = o.get_name()
            if name is not None:
                sub_dir_path += "_" + name
            sub_dir_path = replace_unallowed_chars(
                sub_dir_path, allow_dot=False)

            op = save_image_stack(img_stack_out[i],
                                  os.path.join(model.config.output_path,
                                               "final",
                                               sub_dir_path,
                                               "{t}_t_{idx}.png"),
                                  additional_filename_variables_list=[
                                      {"t": lead_zeros_fmt.format(k), "idx": real_lead_zeros_fmt.format(t_real_idx[j])} for j, k in enumerate(t_cur_idx)],
                                  additional_filename_variables={
                "epoch": model.current_epoch},
                override=True)
            object_images_paths[i].extend(op)

            if bar is not None:
                bar.update()
    return object_images_paths


def calculate_metrics(
    outputs: torch.Tensor,
    times: torch.Tensor,
    metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    dataset: NAGDataset
) -> Dict[str, torch.Tensor]:
    T, HO, WO, C = outputs.shape
    from torchvision.transforms import Resize

    results = {}

    indices = {i: t.item() for i, t in enumerate(
        dataset._frame_timestamps) if t.item() in set(times.detach().cpu().tolist())}

    if len(set(indices.keys())) != len(outputs):
        raise ValueError("Not all indices are in the dataset")

    targets = None
    resize_target = None
    resize_output = None
    outputs_scaled = None
    for i, t in indices.items():
        ti = torch.argwhere(times == t).squeeze()
        target = dataset.load_image(torch.tensor(i), native_size=True)[
            0]  # Loading the target image in native size
        _, H, W = target.shape

        if HO != H or WO != W:
            if resize_target is None and H < HO or W < WO:
                logging.warning(
                    "Target image is smaller than output size. Resizing to output size.")
                resize_target = Resize((HO, WO))
            if H < HO or W < WO:
                target = resize_target(target).permute(2, 0, 1)

            if resize_output is None and H > HO or W > WO:
                logging.info(
                    "Output image is smaller than target size. Resizing to target size.")
                resize_output = Resize((H, W))

            if H > HO or W > WO:
                if outputs_scaled is None:
                    outputs_scaled = torch.zeros(
                        len(indices), H, W, C, device=outputs.device, dtype=outputs.dtype)
                outputs_scaled[ti] = resize_output(
                    outputs[i].permute(2, 0, 1)).permute(1, 2, 0)

        if targets is None:
            targets = torch.zeros(
                len(indices), H, W, C, device=outputs.device, dtype=outputs.dtype)
        targets[ti] = target.permute(1, 2, 0)

    if outputs_scaled is not None:
        outputs = outputs_scaled
    # Permute to NCHW
    outputs = outputs.permute(0, 3, 1, 2)
    targets = targets.permute(0, 3, 1, 2)

    for name, metric in metrics.items():
        if isinstance(metric, torch.nn.Module):
            metric = metric.to(outputs.device)
        results[name] = metric(outputs, targets)
    return results


def calculate_mask_metrics(
    outputs: torch.Tensor,
    times: torch.Tensor,
    metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    dataset: NAGDataset,
    also_only_mask: bool = True
) -> Dict[str, torch.Tensor]:
    T, HO, WO, C = outputs.shape
    from torchvision.transforms import Resize

    results = {}

    indices = {i: t.item() for i, t in enumerate(
        dataset._frame_timestamps) if t.item() in set(times.detach().cpu().tolist())}

    if len(set(indices.keys())) != len(outputs):
        raise ValueError("Not all indices are in the dataset")

    targets = None
    masks = None
    resize_target = None
    resize_output = None
    outputs_scaled = None
    for i, t in indices.items():
        ti = torch.argwhere(times == t).squeeze()
        target = dataset.load_image(torch.tensor(i), native_size=True)[
            0]  # Loading the target image in native size
        _, H, W = target.shape
        mask = dataset.load_mask(torch.tensor(i), init_size=False)[0]

        if HO != H or WO != W:
            if resize_target is None and H < HO or W < WO:
                logging.warning(
                    "Target image is smaller than output size. Resizing to output size.")
                resize_target = Resize((HO, WO))
            if H < HO or W < WO:
                target = resize_target(target).permute(2, 0, 1)
                mask = resize_target(mask.float()).permute(2, 0, 1).bool()

            if resize_output is None and H > HO or W > WO:
                logging.info(
                    "Output image is smaller than target size. Resizing to target size.")
                resize_output = Resize((H, W))

            if H > HO or W > WO:
                if outputs_scaled is None:
                    outputs_scaled = torch.zeros(
                        len(indices), H, W, C, device=outputs.device, dtype=outputs.dtype)
                outputs_scaled[ti] = resize_output(
                    outputs[i].permute(2, 0, 1)).permute(1, 2, 0)

        if targets is None:
            targets = torch.zeros(
                len(indices), H, W, C, device=outputs.device, dtype=outputs.dtype)
        if masks is None:
            masks = torch.zeros(
                len(indices), H, W, len(dataset.mask_ids), device=outputs.device, dtype=torch.bool)
        targets[ti] = target.permute(1, 2, 0)
        masks[ti] = mask.permute(1, 2, 0)

    if outputs_scaled is not None:
        outputs = outputs_scaled
    # Permute to NCHW
    outputs = outputs.permute(0, 3, 1, 2)
    targets = targets.permute(0, 3, 1, 2)
    masks = masks.permute(0, 3, 1, 2)

    B, C, H, W = outputs.shape
    _, O, _, _ = masks.shape

    for name, metric in metrics.items():
        if isinstance(metric, torch.nn.Module):
            metric = metric.to(outputs.device)
        res = torch.full((B, O), fill_value=torch.nan,
                         device=outputs.device, dtype=outputs.dtype)
        res_om = torch.full((B, O), fill_value=torch.nan, device=outputs.device,
                            dtype=outputs.dtype) if also_only_mask else None

        for b in range(B):
            for o in range(O):
                mask = masks[b, o].unsqueeze(0)  # [1, H, W]
                coords = torch.argwhere(mask)

                if len(coords) == 0:
                    # If there are no pixels in the mask, skip
                    continue

                y0, x0 = coords.amin(dim=0)[1:]
                y1, x1 = coords.amax(dim=0)[1:] + 1

                output = outputs[b, :, y0:y1, x0:x1]
                target = targets[b, :, y0:y1, x0:x1]

                res[b, o] = metric(output.unsqueeze(0), target.unsqueeze(0))

                if also_only_mask:
                    cropped_mask = mask[:, y0:y1, x0:x1]
                    output_om = output.clone()
                    output_om[~cropped_mask.expand(
                        3, -1, -1)] = 0.  # Set nonmask pixels to 0 to only calculate the metric on the mask
                    target_om = target.clone()
                    target_om[~cropped_mask.expand(3, -1, -1)] = 0.
                    res_om[b, o] = metric(
                        output_om.unsqueeze(0), target_om.unsqueeze(0))

        results[name] = res
        if also_only_mask:
            results[name + "_only_mask"] = res_om
    return results


def calculate_metrics_from_paths(
    paths: list,
    times: torch.Tensor,
    metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    dataset: NAGDataset
) -> Dict[str, torch.Tensor]:
    from tools.io.image import load_image_stack
    T = len(paths)
    if T != len(times):
        raise ValueError("Times and paths do not match")
    img = torch.tensor(load_image_stack(
        sorted_image_paths=paths)).float().div(255.0)
    return calculate_metrics(img, times, metrics, dataset)


def calculate_mask_metrics_from_paths(
    paths: list,
    times: torch.Tensor,
    metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    dataset: NAGDataset
) -> Dict[str, torch.Tensor]:
    from tools.io.image import load_image_stack
    T = len(paths)
    if T != len(times):
        raise ValueError("Times and paths do not match")
    img = torch.tensor(load_image_stack(
        sorted_image_paths=paths)).float().div(255.0)
    return calculate_mask_metrics(img, times, metrics, dataset)


class NAGCallback(pl.Callback):

    def __init__(self, runner: ConfigRunner):
        super().__init__()
        from nag.run.nag_runner import NAGRunner
        self.runner: NAGRunner = runner

    @property
    def config(self):
        return self.runner.config

    def calc_sin_epoch(self,
                       epoch: int, model: NAGModel):
        max_coarse_to_fine = model.config.coarse_to_fine_end_epoch
        min_coarse_to_fine = model.config.coarse_to_fine_start_epoch

        if max_coarse_to_fine is None:
            max_coarse_to_fine = model.config.max_epochs
        if min_coarse_to_fine is None:
            min_coarse_to_fine = 0

        if (max_coarse_to_fine == 0) or (max_coarse_to_fine < min_coarse_to_fine):
            return tensorify(1.0, device=model.device, dtype=model.config.dtype)

        if max_coarse_to_fine > self.runner.config.max_epochs:
            max_coarse_to_fine = self.runner.config.max_epochs
            self.runner.config.coarse_to_fine_end_epoch = max_coarse_to_fine
            logging.warning(
                "Coarse to fine end epoch is greater than max epochs. Setting to max epochs.")

        ep = tensorify(min(1.0, 0.05 + np.sin(min(1, (epoch - min_coarse_to_fine) /
                                              ((max_coarse_to_fine - (1 if max_coarse_to_fine > 1 else 0) - min_coarse_to_fine))) * np.pi/2)),
                       device=model.device, dtype=model.config.dtype)

        return ep.clamp(0, 1)

    def check_times(self, times: np.ndarray, model: NAGModel, check_duplicates: bool = False) -> np.ndarray:
        from tools.util.torch import index_of_first
        # Check if the times are in the dataset
        ttimes = torch.tensor(times).clone()
        times_idx = index_of_first(
            self.runner.dataset.frame_timestamps, ttimes)
        invalid_specified_times = (times_idx == -1)
        new_plot_times = times.copy().astype(float)
        need_reset = False

        non_existing = invalid_specified_times.clone()
        invalid_indices = invalid_specified_times.argwhere()

        if invalid_specified_times.any():
            # Check if non-existing times are floats and in the range of 0 to 1
            in_range_missing = (ttimes[invalid_specified_times] >= 0) & (
                ttimes[invalid_specified_times] <= 1)

            if in_range_missing.any():
                non_existing[invalid_specified_times] = in_range_missing
                invalid_times = flatten_batch_dims(
                    ttimes[non_existing], -2)[0].repeat(len(self.runner.dataset.frame_timestamps), 1)
                test_times = self.runner.dataset.frame_timestamps.unsqueeze(
                    -1).expand_as(invalid_times)
                replace_idx = torch.abs(
                    (test_times - invalid_times)).argmin(dim=0)
                new_times = self.runner.dataset.frame_timestamps[replace_idx]
                significant = torch.abs(
                    ttimes[non_existing] - new_times) > 1e-3
                new_plot_times[non_existing] = new_times
                if significant.any():
                    sig_old = ttimes[non_existing][significant]
                    sig_new = new_times[significant]
                    fmt = "{:.3f}"
                    logging.warning("The following fp plot times are not in the dataset and beeing replaced: \n" + "\n".join(
                        [(fmt.format(o) + " -> " + fmt.format(n)) for o, n in zip(sig_old, sig_new)]))
                    need_reset = True
                checked_indices = invalid_indices[in_range_missing]
                non_existing[checked_indices] = False

            if (~in_range_missing).any() and non_existing.any():
                # Account these time as int indices of the original dataset
                to_check = invalid_indices[~in_range_missing].squeeze()
                invalid_times = ttimes[to_check]
                invalid_times, _ = flatten_batch_dims(invalid_times, -1)
                # Check that all non_ex
                results = self.runner.dataset.frame_indices_to_times(
                    invalid_times)
                invalid_results = (results == -1)
                valid_patched_idx = to_check[~invalid_results]
                valid_times = results[~invalid_results]
                if len(valid_patched_idx) > 0:
                    new_plot_times[valid_patched_idx] = valid_times
                    need_reset = True
                non_existing[valid_patched_idx] = False
                # Log the Replacement
                if len(valid_times) > 0:
                    logging.warning("The following integer plot times beeing replaced by their float correspondence: \n" + "\n".join(
                        [f"{o} -> {n:.3f}" for o, n in zip(invalid_times[~invalid_results], valid_times)]))

        if non_existing.any():
            invalid_times = ttimes[non_existing]
            new_plot_times = new_plot_times[~non_existing]
            need_reset = True
            logging.warning("The following plot times are not in the dataset and will be removed: \n" + "\n".join(
                [f"{t:.3f}" for t in invalid_times]))

        if check_duplicates:
            ut = np.unique(new_plot_times, return_counts=True)
            unique_times = ut[0]
            if len(unique_times) < len(new_plot_times):
                kv = {k: v for k, v in zip(ut[0], ut[1]) if v > 1}
                logging.warning(
                    "The following plot times are duplicated multiple times and will be just printed ones: \n" + "\n".join(
                        [f"{v:.3f}: {int(c)}" for v, c in kv.items()]))
                new_plot_times = unique_times

        if need_reset:
            model.config.in_training_plot_times = new_plot_times
        return new_plot_times

    def on_train_epoch_start(self, trainer: pl.Trainer, model: NAGModel):
        self.runner.epoch = model.current_epoch

        # Set num batches in model
        model.num_batches = torch.tensor(len(trainer.train_dataloader))

        with torch.no_grad():
            if model.next_sin_epoch is not None and model.current_epoch != 0:
                model.sin_epoch = model.next_sin_epoch
            else:
                model.sin_epoch = self.calc_sin_epoch(
                    model.current_epoch, model)

            if (model.current_epoch + 1) < model.config.max_epochs:
                model.next_sin_epoch = self.calc_sin_epoch(
                    model.current_epoch + 1, model)
            else:
                model.next_sin_epoch = torch.tensor(
                    1.0, device=model.device, dtype=model.config.dtype)

            # trainer.train_dataloader.dataset.sin_epoch = model.sin_epoch

            # Check if phase should be changed
            self.change_phase(
                model, save_after_phase_change=self.config.save_after_phase_change)

            # if model.sin_epoch > 0.4 and not self.flow_unlocked:
            #     # unlock flow model
            #     logging.info("Unlocking flow model.")
            #     model.enable_flow(True)
            #     self.flow_unlocked = True

            # if model.current_epoch >= self.config.pretraining_epochs and not self.positions_unlocked:
            #     logging.info("Unlocking position learning.")
            #     model.enable_position_learning(True)
            #     self.positions_unlocked = True

            if model.config.fast:  # skip tensorboarding except for beginning and end
                if model.current_epoch == model.config.max_epochs - 1 or model.current_epoch == 0:
                    pass
                else:
                    return

            if model.config.in_training_checkpoint_interval is not None and (model.current_epoch % model.config.in_training_checkpoint_interval) == 0:
                if model.current_epoch == 0 != self.config.max_epochs:
                    self.save_model(trainer, model)

            save_to_disk = model.config.in_training_save_plots_and_images

            # for i, frame in enumerate([0, model.bundle.num_frames//2, model.bundle.num_frames-1]): # can sample more frames
            if model.current_epoch % model.config.in_training_epoch_plot_interval == 0:
                if model.current_epoch == 0 and not model.config.in_training_plot_on_first_epoch:
                    return
                # Plotting of stuff
                resolution = model.camera._image_resolution.flip(
                    -1).detach().cpu().numpy()

                resolution = (
                    resolution * model.config.in_training_plot_resolution_factor).astype(int)

                # Plot the hole thing
                times = numpyify(model.config.in_training_plot_times)
                times = self.check_times(times, model, check_duplicates=True)
                if len(times) > 0:
                    ds: NAGDataset = self.runner.dataset
                    t_indices = ds.times_to_indices(
                        tensorify(times, device="cpu"))

                    t_real_indices = ds.indices_to_frame_indices(t_indices)

                    lead_zeros_fmt = get_leading_zeros_format_string(
                        t_indices.amax().item())

                    paths, img = generate_batched_outputs_and_save(
                        model, resolution, tensorify(times, device=model.device), t_indices, t_real_indices=t_real_indices, return_images=True,
                        progress_bar=model.config.use_progress_bar, progress_factory=model.config.progress_factory)

                    if model.config.in_training_epoch_plot_log:
                        images = img.permute(0, 2, 3, 1).detach().cpu().numpy()
                        for i, t in enumerate(times):
                            tag = "in_training/" + \
                                f"{lead_zeros_fmt.format(t_indices[i])}_t_full"
                            self.runner.experiment_logger.add_image(
                                tag, images[i], dataformats="HWC", step=model.current_epoch, epoch=model.current_epoch)
                        if model.config.in_training_epoch_plot_log_l1:
                            diff_save_path = os.path.join(model.config.output_path,
                                                          "in_training_diff",
                                                          "{epoch:02d}_epoch_{t}_t_diff.png")
                            self.generate_diff_images_and_save(
                                images, tensorify(times, device=model.device), t_indices, ds, save_to_disk=save_to_disk, save_format_path=diff_save_path, log_in_logger=True)

                            # gt = np.zeros_like(images)
                            # from tools.io.image import resize_image
                            # for i in range(times.shape[0]):
                            #     gti = (
                            #         ds[t_indices[i].item()][0] * 255).permute(1, 2, 0).numpy().astype(np.uint8)
                            #     if gti.shape != images.shape[1:]:
                            #         gti = resize_image(
                            #             gti, size=images.shape[1:3])
                            #     gt[i] = gti
                            # diff = np.abs(images.astype(
                            #     float) - gt.astype(float)).mean(axis=-1, keepdims=True)
                            # minmax = MinMax(0, 255, axis=(0, 1, 2))
                            # norm_diff = minmax.fit_transform(
                            #     diff).astype(np.uint8)

                            # inpainted_images = np.zeros_like(gt)
                            # for i in range(times.shape[0]):
                            #     tag = "in_training/" + \
                            #         f"{lead_zeros_fmt.format(t_indices[i])}_t_diff"

                            #     inp_img = put_text(norm_diff[i].repeat(3, axis=-1),
                            #                        f"Min: {minmax.min.item():.3f} Max: {minmax.max.item():.3f}",
                            #                        placement="top-right",
                            #                        size=.5,
                            #                        thickness=1,
                            #                        color="k",
                            #                        margin=10,
                            #                        background_color="white",
                            #                        background_stroke=2,
                            #                        background_stroke_color="k",
                            #                        padding=7
                            #                        )
                            #     inpainted_images[i] = inp_img

                            #     self.runner.experiment_logger.add_image(
                            #         tag, inp_img, step=model.current_epoch,
                            #         epoch=model.current_epoch,
                            #         dataformats="HWC")
                            # if save_to_disk:
                            #     save_image_stack(inpainted_images,
                            #                      os.path.join(model.config.output_path,
                            #                                   "in_training_diff",
                            #                                   "{epoch:02d}_epoch_{t}_t_diff.png"),
                            #                      additional_filename_variables_list=[
                            #                          {"t": lead_zeros_fmt.format(int(k))} for k in t_indices],
                            #                      additional_filename_variables={
                            #                          "epoch": model.current_epoch},
                            #                      override=True)
                    plt.close("all")

            if model.current_epoch % model.config.in_training_save_geometry_interval == 0:
                with torch.no_grad():
                    node_indices = model.get_node_indices()
                    tracker: Tracker = self.runner.tracker
                    ind_names = {m.get_index(): m.get_name()
                                 for m in model.objects}
                    ind_names[model.camera.get_index()] = "camera"
                    positions = model.get_global_positions(model.camera._times)
                    quaternions = rotmat_to_unitquat(
                        positions[..., :3, :3]).detach().cpu().numpy()
                    translations = positions[..., :3, 3].detach().cpu().numpy()
                    for i in range(positions.shape[0]):
                        index = node_indices[i]
                        name = ind_names[index.item()]
                        name_quat = f"{name}_quat"
                        name_trans = f"{name}_trans"
                        tracker.epoch_metric(
                            name_quat, quaternions[i], in_training=True, step=model.current_epoch)
                        tracker.epoch_metric(
                            name_trans, translations[i], in_training=True, step=model.current_epoch)
                # Save tracker
                self.save_tracker()
                plt.close("all")
            if (model.current_epoch % model.config.in_training_plot_geometry_interval == 0) or (model.current_epoch == (self.config.max_epochs-1)):
                objs = []
                labels = []

                ds: NAGDataset = self.runner.dataset
                t_indices = ds.times_to_indices(
                    tensorify(ds.frame_timestamps, device="cpu"))

                t_real_indices = ds.indices_to_frame_indices(t_indices)

                if model.config.in_training_plot_geometry_camera:
                    objs.append([model.camera.get_index()])
                    labels.append("camera")
                indices_candidates = model.config.in_training_plot_geometry_object_indices
                if indices_candidates is None:
                    for o in model.objects:
                        if isinstance(o, BackgroundPlaneSceneNode3D):
                            continue
                        idx = o.get_index()
                        objs.append([idx])
                        labels.append("object_" + str(idx))
                else:
                    for ind in indices_candidates:
                        if isinstance(ind, int):
                            labels.append("object_" + str(ind))
                            objs.append([ind])
                        else:
                            labels.append("objects_" + "_".join(ind))
                            objs.append(ind)
                figs = model.plot_objects_positions(
                    objs, save=save_to_disk, path=f"{{index:02d}}_{model.current_epoch:02d}_object_positions", t_real=t_real_indices)
                for label, fig in zip(labels, figs):
                    tag = "geometry/" + label
                    self.runner.experiment_logger.add_figure(
                        tag, fig,
                        step=model.current_epoch,
                        epoch=model.current_epoch)
                plt.close("all")

    def on_train_epoch_end(self, trainer: pl.Trainer, model: NAGModel):
        self.runner.tracker.epoch(in_training=True)

    def generate_diff_images_and_save(
        self,
        images: np.ndarray,
        times: torch.Tensor,
        t_indices: torch.Tensor,
        ds: NAGDataset,
        save_to_disk: bool,
        save_format_path: Optional[str] = None,
        log_in_logger: bool = True
    ):
        model = self.runner.model
        if save_to_disk and save_format_path is None:
            save_format_path = os.path.join(model.config.output_path,
                                            "in_training_diff",
                                            "{epoch:02d}_epoch_{t}_t_diff.png")

        lead_zeros_fmt = get_leading_zeros_format_string(
            t_indices.amax().item())
        gt = np.zeros_like(images)
        from tools.io.image import resize_image
        for i in range(times.shape[0]):
            gti = (
                ds[t_indices[i].item()][0] * 255).permute(1, 2, 0).numpy().astype(np.uint8)
            if gti.shape != images.shape[1:]:
                gti = resize_image(
                    gti, size=images.shape[1:3])
            gt[i] = gti
        diff = np.abs(images.astype(
            float) - gt.astype(float)).mean(axis=-1, keepdims=True)
        minmax = MinMax(0, 255, axis=(0, 1, 2))
        norm_diff = minmax.fit_transform(
            diff).astype(np.uint8)

        inpainted_images = np.zeros_like(gt)
        for i in range(times.shape[0]):
            tag = "in_training/" + \
                f"{lead_zeros_fmt.format(t_indices[i])}_t_diff"

            inp_img = put_text(norm_diff[i].repeat(3, axis=-1),
                               f"Min: {minmax.min.item():.3f} Max: {minmax.max.item():.3f}",
                               placement="top-right",
                               size=.5,
                               thickness=1,
                               color="k",
                               margin=10,
                               background_color="white",
                               background_stroke=2,
                               background_stroke_color="k",
                               padding=7
                               )

            inpainted_images[i] = inp_img

            if log_in_logger:
                self.runner.experiment_logger.add_image(
                    tag, inp_img,
                    step=model.current_epoch,
                    epoch=model.current_epoch,
                    dataformats="HWC")

        if save_to_disk:
            save_image_stack(inpainted_images,
                             save_format_path,
                             additional_filename_variables_list=[
                                 {"t": lead_zeros_fmt.format(int(k))} for k in t_indices],
                             additional_filename_variables={
                                 "epoch": model.current_epoch},
                             override=True)

    def evaluate_against_gt(
        self,
        output_folders: Union[str, List[str]],
        metrics: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        times: Optional[torch.Tensor] = None,
        filename_format: str = "{t}.tiff",
        resolution: Optional[Tuple[int, int]] = None,
        progress_bar: bool = True,
    ):
        from tools.io.image import resize_image, save_image
        model = self.runner.model
        config = self.runner.config
        dataset = self.runner.dataset
        progress_factory = config.progress_factory
        if not isinstance(metrics, List):
            metrics = [metrics]
        if isinstance(output_folders, str):
            output_folders = [output_folders]

        if resolution is None:
            resolution = dataset.image_shape[1:3]

        resolution = tensorify(
            resolution, device="cpu").flip(-1).int().tolist()
        res_xy = resolution
        res_yx = resolution[::-1]

        if times is None:
            times = dataset.frame_timestamps
        frame_t = dataset.times_to_frame_indices(
            tensorify(times, device="cpu"))
        frame_t_indices = dataset.times_to_indices(
            tensorify(times, device="cpu"))
        bar = None
        if progress_bar:
            if progress_factory is None:
                progress_factory = ProgressFactory.global_instance()
            bar = progress_factory.bar(
                total=len(times), desc="Evaluating metrics", is_reusable=True, tag="evaluate_metrics")
        with torch.no_grad():
            for i, t in enumerate(times):
                tidx = frame_t_indices[i]
                output = model.generate_outputs(
                    config=config, resolution=res_xy, t=t, progress_bar=progress_bar, progress_factory=progress_factory)
                target = dataset.load_image(
                    tidx, native_size=True, init_size=False)
                if target.shape[-2:] != tuple(res_yx):
                    target = resize_image(target, size=res_yx)
                for km, m in enumerate(metrics):
                    out_p = os.path.join(config.output_path, output_folders[km], parse_format_string(
                        filename_format, obj_list=[{"t": frame_t_indices[i].item()}])[0])
                    calc = m(output, target)
                    if len(calc.shape) == 4 and calc.shape[0] == 1:
                        calc = calc.squeeze(0)
                    save_image(calc, out_p, override=True)
                if bar is not None:
                    bar.update()

    def _parse_metrics(self, metrics: List[Union[str, type, Metric, Tuple[Union[str, type, Metric], Dict[str, Any]]]]) -> Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
        ret = dict()

        def _check_obj_vals(mt: Metric):
            if hasattr(met, "data_range"):
                met.data_range = 1.
            if hasattr(met, "max_value"):
                met.max_value = torch.tensor(1.)

        for i, metric_type in enumerate(metrics):
            if isinstance(metric_type, str) or isinstance(metric_type, type):
                metric_type = parse_type(metric_type, Metric)
                met = metric_type()
            elif isinstance(metric_type, tuple):
                mt = metric_type[0]
                args = metric_type[1] if len(metric_type) > 1 else dict()
                if not isinstance(args, dict):
                    raise ValueError(
                        f"If metric is a tuple, the second element must be a dict, invalid for index {i}")
                metric_type = parse_type(mt, (Metric, torch.nn.Module))
                met = metric_type(**args)
            elif isinstance(metric_type, Metric):
                met = metric_type
            else:
                raise ValueError(
                    f"Unknown metric type: {metric_type}")
            _check_obj_vals(met)
            ret[Tracker.get_metric_name(met)] = met
        return ret

    def parse_final_metrics(self) -> Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
        config: NAGConfig = self.config
        return self._parse_metrics(config.final_evaluation_metrics)

    def parse_final_mask_metrics(self) -> Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
        config: NAGConfig = self.config
        return self._parse_metrics(config.final_mask_evaluation_metrics)

    def save_initial(self, model: NAGModel):
        if not self.config.initial_plot:
            logging.info("Initial plot is disabled. Skipping.")
            return
        times = numpyify(self.runner.dataset.frame_timestamps)

        resolution = model.camera._image_resolution.flip(
            -1).detach().cpu().numpy()
        resolution = (
            resolution * model.config.initial_plot_resolution_factor).astype(int)

        t_indices = self.runner.dataset.times_to_indices(
            tensorify(times, device="cpu"))
        t_real_indices = tensorify(
            self.runner.dataset._index["index"].values, device="cpu")

        # img = model.generate_outputs(model.config,
        #                              t=times,
        #                              resolution=resolution,
        #                              progress_bar=model.config.use_progress_bar,
        #                              progress_factory=model.config.progress_factory)

        save_initial_fmt_path = os.path.join(model.config.output_path,
                                             "initial",
                                             "complete",
                                             "{t}_t_{idx}_idx.png")

        generated_initial_paths = generate_batched_outputs_and_save(
            model,
            resolution,
            t=tensorify(times, device=model.device),
            t_indices=t_indices,
            t_real_indices=t_real_indices,
            batch_size=5,
            save_format_path=save_initial_fmt_path,
            return_images=self.config.initial_diff_plot,
            progress_bar=model.config.use_progress_bar,
            progress_factory=model.config.progress_factory
        )
        gen_images = None
        if self.config.initial_diff_plot:
            generated_initial_paths, gen_images = generated_initial_paths
            images = gen_images.permute(0, 2, 3, 1).detach().cpu().numpy()
            diff_save_path = os.path.join(model.config.output_path,
                                          "initial",
                                          "diff",
                                          "{epoch:02d}_epoch_{t}_t_diff.png")
            self.generate_diff_images_and_save(
                images, tensorify(times, device=model.device), t_indices, self.runner.dataset, save_to_disk=True, save_format_path=diff_save_path, log_in_logger=False)

        if self.config.initial_plot_video:
            try:
                # Complete scene video
                gen = load_image_stack_generator(
                    sorted_image_paths=generated_initial_paths, progress_bar=False)
                write_mp4_generator(gen, os.path.join(
                    model.config.output_path, "initial", "complete_scene.mp4"), fps=5, progress_bar=True, frame_counter=True)
            except Exception as e:
                logging.error(f"Could not save complete video: {e}")

    def save_final(self,
                   model: NAGModel,
                   generate_final_images: bool = True,
                   generate_per_object_images: bool = True,
                   evaluate_metrics: bool = True,
                   evaluate_mask_metrics: bool = True,
                   scan_existing: bool = True
                   ):
        # Save the images in high resolution of all time steps

        times = numpyify(self.runner.dataset.frame_timestamps)

        resolution = model.camera._image_resolution.flip(
            -1).detach().cpu().numpy()
        resolution = (
            resolution * model.config.final_plot_resolution_factor).astype(int)

        t_indices = self.runner.dataset.times_to_indices(
            tensorify(times, device="cpu"))

        t_real_indices = tensorify(
            self.runner.dataset._index["index"].values, device="cpu")

        # img = model.generate_outputs(model.config,
        #                              t=times,
        #                              resolution=resolution,
        #                              progress_bar=model.config.use_progress_bar,
        #                              progress_factory=model.config.progress_factory)

        if generate_final_images:
            logging.info("Generating final images.")
            save_final_fmt_path = os.path.join(model.config.output_path,
                                               "final",
                                               "complete",
                                               "{t}_t.png")
            exists = False
            generated_final_paths = None
            if scan_existing and not self.config.final_diff_plot:
                from tools.util.path_tools import read_directory
                from tools.util.torch import index_of_first
                paths = read_directory(os.path.join(
                    model.config.output_path, "final", "complete"), pattern=r"(?P<t>\d+)_t\.png", parser=dict(t=int))
                found_ts = torch.tensor([p["t"] for p in paths])
                order = index_of_first(found_ts, t_indices)
                exists = (order > -1).all()
                if exists:
                    generated_final_paths = np.array([x["path"] for x in paths])[
                        order].tolist()

            if not exists:
                generated_final_paths = generate_batched_outputs_and_save(
                    model,
                    resolution,
                    t=tensorify(times, device=model.device),
                    t_indices=t_indices,
                    t_real_indices=t_real_indices,
                    batch_size=5,
                    save_format_path=save_final_fmt_path,
                    return_images=self.config.final_diff_plot,
                    progress_bar=model.config.use_progress_bar,
                    progress_factory=model.config.progress_factory
                )

            gen_images = None
            if self.config.final_diff_plot:
                logging.info("Generating final diff images.")
                generated_final_paths, gen_images = generated_final_paths
                images = gen_images.permute(0, 2, 3, 1).detach().cpu().numpy()
                diff_save_path = os.path.join(model.config.output_path,
                                              "final",
                                              "diff",
                                              "{epoch:02d}_epoch_{t}_t_diff.png")
                self.generate_diff_images_and_save(
                    images, tensorify(times, device=model.device), t_indices, self.runner.dataset, save_to_disk=True, save_format_path=diff_save_path, log_in_logger=False)

            if self.config.final_plot_without_view_dependency:
                logging.info(
                    "Generating final images without view dependency.")
                save_final_fmt_path_nv = os.path.join(model.config.output_path,
                                                      "final",
                                                      "complete_no_view_dependency",
                                                      "{t}_t.png")

                generated_final_paths_nv = generate_batched_outputs_and_save(
                    model,
                    resolution,
                    t=tensorify(times, device=model.device),
                    t_indices=t_indices,
                    t_real_indices=t_real_indices,
                    batch_size=5,
                    save_format_path=save_final_fmt_path_nv,
                    return_images=False,
                    progress_bar=model.config.use_progress_bar,
                    progress_factory=model.config.progress_factory,
                    disable_view_dependency=True
                )

            if self.config.has_camera_aberration_plane:
                logging.info(
                    "Generating final images without camera aberration.")
                save_final_fmt_path_abr = os.path.join(model.config.output_path,
                                                       "final",
                                                       "complete_no_aberration",
                                                       "{t}_t.png")
                generated_final_paths_no_abrr = generate_batched_outputs_and_save(
                    model,
                    resolution,
                    t=tensorify(times, device=model.device),
                    t_indices=t_indices,
                    t_real_indices=t_real_indices,
                    batch_size=5,
                    save_format_path=save_final_fmt_path_abr,
                    return_images=False,
                    progress_bar=model.config.use_progress_bar,
                    progress_factory=model.config.progress_factory,
                    disable_camera_aberration=True
                )

        if evaluate_metrics:
            logging.info("Evaluate final metrics.")
            metrics = self.parse_final_metrics()
            if len(metrics) > 0:
                from tools.util.torch import batched_generator_exec
                t_b_size = 10
                bm_exec = batched_generator_exec(
                    batched_params=['paths', 'times'],
                    default_batch_size=t_b_size,
                    default_explicit_garbage_collection=True,
                    default_multiprocessing=True)(
                    calculate_metrics_from_paths
                )
                result_iter = bm_exec(generated_final_paths,
                                      tensorify(times).detach().clone(),
                                      metrics=metrics,
                                      dataset=self.runner.dataset
                                      )
                merged_results = dict()
                for i, item_result in enumerate(result_iter):
                    for k, v in item_result.items():
                        if k not in merged_results:
                            merged_results[k] = []
                        merged_results[k].append(v)
                merged_results = {k: torch.cat(v, dim=0)
                                  for k, v in merged_results.items()}

                for name, result in merged_results.items():
                    tag_mean = "metrics/" + "Mean" + name
                    mean_value = result.mean().item()
                    tag_single = "metrics/" + name
                    if self.runner.experiment_logger is not None:
                        self.runner.experiment_logger.add_scalar(
                            tag_mean, mean_value, step=model.current_epoch, epoch=model.current_epoch)
                    self.runner.tracker.epoch_metric(
                        "Mean" + name, mean_value, in_training=True, step=model.current_epoch)
                    self.runner.tracker.epoch_metric(
                        name, result, in_training=True, step=model.current_epoch)
                    for i, t in enumerate(times):
                        tag = tag_single + f"-final"
                        if self.runner.experiment_logger is not None:
                            self.runner.experiment_logger.add_scalar(
                                tag, result[i].item(), step=i, epoch=model.current_epoch, batch=i)
                self.save_tracker()

        if evaluate_mask_metrics:
            logging.info("Evaluate mask final metrics.")
            metrics = self.parse_final_mask_metrics()
            if len(metrics) > 0:
                from tools.util.torch import batched_generator_exec
                t_b_size = 10
                bm_exec = batched_generator_exec(
                    batched_params=['paths', 'times'],
                    default_batch_size=t_b_size,
                    default_explicit_garbage_collection=True,
                    default_multiprocessing=True)(
                    calculate_mask_metrics_from_paths
                )
                result_iter = bm_exec(generated_final_paths,
                                      tensorify(times).detach().clone(),
                                      metrics=metrics,
                                      dataset=self.runner.dataset
                                      )
                merged_results = dict()
                for i, item_result in enumerate(result_iter):
                    for k, v in item_result.items():
                        if k not in merged_results:
                            merged_results[k] = []
                        merged_results[k].append(v)
                merged_results = {k: torch.cat(v, dim=0)
                                  for k, v in merged_results.items()}

                mask_ids = self.runner.dataset.mask_ids
                fmt_id = get_leading_zeros_format_string(len(mask_ids))

                for name, result in merged_results.items():
                    B, O = result.shape
                    for o in range(O):
                        oname = "Mask_" + \
                            fmt_id.format(mask_ids[o].item()) + "_" + name
                        tag_mean = "metrics/" + "Mean_" + oname
                        mean_value = result[:, o].nanmean().item()
                        tag_single = "metrics/" + oname
                        if self.runner.experiment_logger is not None:
                            self.runner.experiment_logger.add_scalar(
                                tag_mean, mean_value, step=model.current_epoch, epoch=model.current_epoch)
                        self.runner.tracker.epoch_metric(
                            "Mean_" + oname, mean_value, in_training=True, step=model.current_epoch)
                        self.runner.tracker.epoch_metric(
                            oname, result[:, o], in_training=True, step=model.current_epoch)
                        for i, t in enumerate(times):
                            tag = tag_single + f"-final"
                            if self.runner.experiment_logger is not None:
                                val = result[i, o].item()
                                if val != torch.nan:
                                    self.runner.experiment_logger.add_scalar(
                                        tag, val, step=i, epoch=model.current_epoch, batch=i)
                self.save_tracker()
                try:
                    self.compute_masked_grouped_metrics()
                    self.save_tracker()
                except Exception as e:
                    logging.error(f"Could group metrics: {e}")

        if generate_per_object_images:
            logging.info("Generating per object images.")
            object_images_path = generate_batched_outputs_per_object_and_save(
                model,
                resolution,
                tensorify(times, device=model.device),
                t_indices,
                t_real_indices=t_real_indices,
                batch_size=5,
                progress_bar=model.config.use_progress_bar,
                progress_factory=model.config.progress_factory
            )

        if self.config.final_modalities_save:
            logging.info("Saving object modalities")
            self.save_object_modalities(model)

        if self.config.save_video:
            logging.info("Saving videos.")
            if generate_final_images:
                try:
                    # Complete scene video
                    gen = load_image_stack_generator(
                        sorted_image_paths=generated_final_paths, progress_bar=False)
                    write_mp4_generator(gen, os.path.join(
                        model.config.output_path, "final", "complete_scene.mp4"), fps=5, progress_bar=True)
                except Exception as e:
                    logging.error(f"Could not save complete video: {e}")

            if generate_per_object_images:
                for i, paths in object_images_path.items():
                    try:
                        gen = load_image_stack_generator(
                            sorted_image_paths=paths, progress_bar=False)
                        write_mp4_generator(gen, os.path.join(
                            model.config.output_path, "final", f"object_{i}_scene.mp4"), fps=5, progress_bar=True)
                    except Exception as e:
                        logging.error(
                            f"Could not save video for object {i}: {e}")

        # if model.config.use_progress_bar:
        #     if factory is None:
        #         factory = ProgressFactory()
        #     bar = factory.bar(total=(
        #         len(objects)), desc="Saving object, images", is_reusable=True, tag="save_objects")

        # images = model.generate_outputs(
        #     model.config,
        #     t=times,
        #     resolution=resolution,
        #     objects=[[o] for o in objects],
        #     progress_bar=model.config.use_progress_bar,
        #     progress_factory=model.config.progress_factory)
        # # Save the images to disk
        # for i, o in enumerate(objects):
        #     save_image_stack((images[i] * 255.0).clamp(0, 255).byte(),
        #                      os.path.join(model.config.output_path,
        #                                   "final",
        #                                   get_leading_zeros_format_string(
        #                                       len(objects)).format(i),
        #                                   "{t}_t.png"),
        #                      additional_filename_variables_list=[{"t": get_leading_zeros_format_string(len(
        #                          model.camera._times)).format(k)} for k in range(len(model.camera._times))],
        #                      additional_filename_variables={
        #                          "epoch": model.current_epoch},
        #                      override=True)

        #     if bar is not None:
        #         bar.update()

    def save_object_modalities(self, model: NAGModel):
        runner = self.runner

        def gather(gen,
                   times,
                   object_labels: List[str],
                   times_indices: List[int],
                   resolution=(510, 384),
                   cache: bool = False,
                   save: bool = True,
                   base_path: str = None,
                   filename_format: str = f"{{index:03d}}_t.png"
                   ):
            numpyify_image = ToNumpyImage(output_dtype=np.uint8)
            numpyify_image_raw = ToNumpyImage(output_dtype=np.float32)
            colors, alphas, flows = None, None, None

            pf = runner.config.progress_factory
            bar = None
            if runner.config.use_progress_bar and pf is not None:
                bar = pf.bar(desc="Gathering outputs")

            num_objects = len(object_labels)

            if cache:
                flows = torch.zeros(num_objects, len(
                    times), 2, resolution[1], resolution[0])
                colors = torch.zeros(num_objects, len(
                    times), 3, resolution[1], resolution[0])
                alphas = torch.zeros(num_objects, len(
                    times), 1, resolution[1], resolution[0])

            for i, (out, t, tidx) in enumerate(zip(gen, times, times_indices)):
                color, alpha, flow = out

                if color is not None:
                    color = color.detach().cpu()
                if alpha is not None:
                    alpha = alpha.detach().cpu()
                if flow is not None:
                    flow = flow.detach().cpu()

                if cache:
                    if color is not None:
                        colors[:, i] = color[:, 0]
                    if alpha is not None:
                        alphas[:, i] = alpha[:, 0]
                    if flow is not None:
                        flows[:, i] = flow[:, 0]

                if save:
                    for object_idx in range(num_objects):
                        object_folder = replace_unallowed_chars(
                            object_labels[object_idx], allow_dot=False)
                        if not os.path.exists(base_path):
                            os.makedirs(base_path)
                        file_name = parse_format_string(filename_format,
                                                        obj_list=[dict(t=t)],
                                                        index_offset=tidx)[0]

                        if color is not None:
                            color_path = os.path.join(
                                base_path, object_folder, "color", file_name)
                            save_image(numpyify_image(
                                color[object_idx, 0]), color_path)
                        if alpha is not None:
                            alpha_path = os.path.join(
                                base_path, object_folder, "alpha", file_name)
                            save_image(numpyify_image(
                                alpha[object_idx, 0])[..., 0], alpha_path)
                        if flow is not None:
                            flow_path = os.path.join(
                                base_path, object_folder,  "flow", file_name)

                            file_name_raw = os.path.splitext(file_name)[
                                0] + ".tif"
                            raw_flow_path = os.path.join(
                                base_path, object_folder, "raw_flow", file_name_raw)
                            cflow = flow[object_idx]
                            if (cflow == 0).all():
                                cflow = torch.ones_like(cflow[0])
                                fcolor = torch.cat(
                                    [cflow, torch.ones_like(cflow[:, :1])], dim=1)
                            else:
                                fcolor = flow_to_color(
                                    cflow[0].permute(1, 2, 0).numpy())
                            save_image(numpyify_image(fcolor), flow_path)

                            rflow = torch.cat(
                                [cflow[0], torch.zeros_like(cflow[0, :1])], dim=0)
                            save_image(numpyify_image_raw(
                                rflow), raw_flow_path)

                if bar is not None:
                    bar.update()

                del color, alpha, flow
                gc.collect()
                torch.cuda.empty_cache()
            return colors, alphas, flows

        selected_objects = runner.model.objects
        times = runner.dataset.frame_timestamps
        times_indices = runner.dataset.indices_to_frame_indices(
            runner.dataset.times_to_indices(times))

        n_obj_format = get_leading_zeros_format_string(len(selected_objects))

        labels = [n_obj_format.format(i) + "_" + x.get_name()
                  for i, x in enumerate(selected_objects)]

        logging.info("Query and saves object modalities.")
        gather(model.generate_modality_outputs(runner.config,
                                               t=times,
                                               objects=selected_objects,
                                               query_color=True,
                                               query_alpha=True,
                                               query_flow=True,
                                               resolution=model.camera._image_resolution.flip(-1).tolist()),
               times,
               object_labels=labels,
               resolution=model.camera._image_resolution.flip(-1).tolist(),
               times_indices=times_indices,
               base_path=os.path.join(runner.config.output_path, "modalities"),
               )  # Shape (num_objects, num_frames, C, H, W)

    def change_phase(self,
                     model: NAGModel,
                     log: bool = True,
                     save_after_phase_change: bool = False):
        # Activate training phase
        time = model.current_epoch
        current_phase_idx = model.active_training_phase
        if current_phase_idx == -1:
            # find the active phase
            new_phase_idx, phase = next(((i, x) for i, x in enumerate(
                model.training_phases) if x.is_active(time)))
            phase.change_phase(None, model, log=log)
        else:
            # Check if the active phase is still active
            current_phase = model.training_phases[current_phase_idx]
            if current_phase.is_active(time):
                return
            # Find the next active phase
            new_phase_idx, phase = next((((current_phase_idx + 1) + i, x) for i, x in enumerate(
                model.training_phases[current_phase_idx + 1:]) if x.is_active(time)))
            phase.change_phase(current_phase, model, log=log)
            if save_after_phase_change:
                logging.info(
                    f"Saving model after phase change to {phase.name}")
                self.save_model(self.runner.trainer, model)
                path = os.path.join("tracker-phase-change", str(time))
                self.save_tracker(path)
        model.active_training_phase = new_phase_idx

    def activate_phases(self, model: NAGModel):
        """Activates all existing phases based on the config to make sure model is in correct state."""
        time = model.current_epoch
        to_activate = [x for x in model.training_phases if x.is_past(time)]
        last_phase = None
        for phase in to_activate:
            if last_phase is not None:
                phase.change_phase(last_phase, model, log=False)
            else:
                phase.change_phase(None, model, log=False)
            last_phase = phase
        # new current phase
        new_phase_idx, phase = next(((i, x) for i, x in enumerate(
            model.training_phases) if x.is_active(time)))
        phase.change_phase(last_phase, model, log=False)
        model.active_training_phase = new_phase_idx

    def save_model(self, trainer: pl.Trainer, model: NAGModel, prefix: str = ""):
        checkpoint_dir = os.path.join(model.config.output_path, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        cpath = os.path.join(
            checkpoint_dir, f"{prefix}{model.current_epoch}.ckpt")
        trainer.save_checkpoint(cpath)

    def save_tracker(self, path: str = "tracker"):
        tracker_dir = os.path.join(self.config.output_path, path)
        self.runner.tracker.save_to_directory(tracker_dir)

    def save_config(self, model: NAGModel):
        config_dir = os.path.join(
            model.config.output_path, "training_config.yaml")
        model.config.save_to_file(config_dir, make_dirs=True, no_uuid=True)

    def on_train_start(self, trainer, model):
        # config = model.config.convert_to_yaml_str(
        #     model.config, handle_unmatched="jsonpickle")
        self.runner.experiment_logger.log_experiment_config(self.runner.config)
        self.save_config(model)
        self.save_initial(model)
        self.activate_phases(model)

    def on_train_end(self, trainer, model):
        self.save_tracker()
        self.save_model(trainer, model, prefix="final-")
        self.save_final(
            model, generate_per_object_images=self.config.final_per_object_save)
        self.save_tracker()

    # region Texture Editing

    def save_texture_editing_background(self, model: NAGModel):
        from tools.io.image import texture_grid, load_image
        from tools.util.path_tools import replace_unallowed_chars
        from tools.transforms.to_tensor_image import ToTensorImage
        from tools.transforms.to_numpy_image import ToNumpyImage
        from tools.video.inpaint_writer import InpaintWriter
        logging.info("Create background tile texture")
        numpyify_image = ToNumpyImage(output_dtype=np.uint8)
        back_node = next((x for x in model.objects if isinstance(
            x, BackgroundImagePlaneSceneNode3D)), None)

        if back_node is None:
            logging.error(
                "No background node found. Cannot save texture editing.")
            return

        ref_time = back_node.flow_reference_time.cpu()
        ref_time, _ = flatten_batch_dims(ref_time, -1)

        filename = replace_unallowed_chars(back_node.get_name(
        ) + f"_grid_texture_{self.runner.dataset.times_to_indices(ref_time[0]).item()}", allow_dot=False) + ".png"
        texture_path = os.path.join(
            self.runner.config.output_path, "edits", filename)

        exists = False
        texture = None
        if os.path.exists(texture_path):
            exists = True
            logging.info(f"Texture already exists: {texture_path}. Loading.")
            tensorify_image = ToTensorImage(output_dtype=torch.float32)
            texture = tensorify_image(load_image(texture_path))
            back_node._create_plain_texture_map(texture)
        else:
            texture_cam = texture_grid(
                model.camera._image_resolution.detach().cpu(), square_size=20)
            logging.info(f"Creating texture map this might take a while ...")
            back_node.create_texture_map(
                texture_cam, t=ref_time, camera=model.camera)
            texture = back_node.texture_map.detach().cpu()

        # Save the texture map to disk
        if not exists:
            save_image(numpyify_image(texture), texture_path)
            logging.info(f"Saved texture to {texture_path}")

        # Render model with the texture map
        back_node.render_texture_map = True

        times = self.runner.dataset.frame_timestamps
        bar = self.runner.config.progress_factory.bar(
            desc="Generating texture editing images", total=len(times), is_reusable=True, tag="texture-editing")

        texture_video_path = os.path.join(
            self.runner.config.output_path, "edits", "texture_grid_background.mp4")
        base_path = os.path.join(
            self.runner.config.output_path, "edits", "texture_grid_background")
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        if not os.path.exists(os.path.dirname(texture_video_path)):
            os.makedirs(os.path.dirname(texture_video_path))
        try:
            gen = model.query(t=times)
            with InpaintWriter(texture_video_path, fps=10, inpaint_counter=False) as writer:
                for i, t in enumerate(times):
                    image = next(gen)
                    image = numpyify_image(image)[0]
                    save_image(image, os.path.join(
                        base_path, f"{i:03d}_t.png"))
                    writer.write(image)
                    bar.update()
        finally:
            back_node.render_texture_map = False

    def save_texture_editing_oid(self, model: NAGModel, oid: int, render_only_object: bool = True, tight_grid: bool = False):
        from tools.io.image import texture_grid, load_image
        from tools.util.path_tools import replace_unallowed_chars
        from tools.transforms.to_tensor_image import ToTensorImage
        from tools.transforms.to_numpy_image import ToNumpyImage
        from tools.video.inpaint_writer import InpaintWriter

        logging.info("Create tile texture for object: " + str(oid))
        numpyify_image = ToNumpyImage(output_dtype=np.uint8)
        fg_node = next(
            (x for x in model.objects if x.get_index() == oid), None)

        if fg_node is None:
            logging.error(
                "No node found. Cannot save texture editing.")
            return

        ref_time = fg_node.flow_reference_time.cpu()
        ref_time, _ = flatten_batch_dims(ref_time, -1)

        ref_time_idx = torch.isclose(
            self.runner.dataset.frame_timestamps, ref_time, atol=1e-5).argwhere().squeeze()
        logging.info(
            f"Reference time index: {ref_time_idx} time: {ref_time[0]}")

        filename = replace_unallowed_chars(fg_node.get_name(
        ) + f"_grid_texture{'_tight' if tight_grid else ''}_{self.runner.dataset.times_to_indices(ref_time[0]).item()}", allow_dot=False) + ".png"
        texture_path = os.path.join(
            self.runner.config.output_path, "edits", filename)

        exists = False
        texture = None
        if os.path.exists(texture_path):
            exists = True
            logging.info(f"Texture already exists: {texture_path}. Loading.")
            tensorify_image = ToTensorImage(output_dtype=torch.float32)
            texture = tensorify_image(load_image(texture_path))
            fg_node._create_plain_texture_map(texture)
        else:
            if not tight_grid:
                texture_cam = texture_grid(
                    model.camera._image_resolution.detach().cpu(), square_size=20)
            else:
                masks = self.runner.dataset.load_mask(ref_time_idx)
                if len(masks.shape) == 4:
                    masks = masks[0]
                mask = masks[oid]
                coords = torch.nonzero(mask).argwhere()
                min_coords = coords.min(dim=0)
                max_coords = coords.max(dim=0)
                span = max_coords - min_coords
                text_inp = texture_grid((span[0], span[1]), square_size=20)
                texture_cam = torch.zeros(
                    (4,) + tuple(model.camera._image_resolution.detach().cpu().tolist()), device=torch.float32)
                texture_cam[:, min_coords[0]:max_coords[0],
                            min_coords[1]:max_coords[1]] = text_inp

            logging.info(f"Creating texture map this might take a while ...")
            fg_node.create_texture_map(
                texture_cam, t=ref_time, camera=model.camera)
            texture = fg_node.texture_map.detach().cpu()

        # Save the texture map to disk
        if not exists:
            save_image(numpyify_image(texture), texture_path)
            logging.info(f"Saved texture to {texture_path}")

        # Render model with the texture map
        fg_node.render_texture_map = True

        times = self.runner.dataset.frame_timestamps
        bar = self.runner.config.progress_factory.bar(
            desc="Generating texture editing images", total=len(times), is_reusable=True, tag="texture-editing")

        texture_video_path = os.path.join(
            self.runner.config.output_path, "edits", f"texture_grid_{oid}.mp4")
        base_path = os.path.join(
            self.runner.config.output_path, "edits", f"texture_grid_{oid}")
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        if not os.path.exists(os.path.dirname(texture_video_path)):
            os.makedirs(os.path.dirname(texture_video_path))

        objects = None
        if render_only_object:
            objects = [[fg_node]]

        try:
            gen = model.query(t=times, objects=objects)
            with InpaintWriter(texture_video_path, fps=10, inpaint_counter=False) as writer:
                for i, t in enumerate(times):
                    image = next(gen)
                    if render_only_object:
                        image = image.squeeze(0)  # Remove N object dim
                    image = numpyify_image(image)[0]
                    save_image(image, os.path.join(
                        base_path, f"{i:03d}_t.png"))
                    writer.write(image)
                    bar.update()
        finally:
            fg_node.render_texture_map = False

    def perform_texture_editing(self, model: NAGModel, textures: Dict[int, List[Union[Tuple[str, int], Tuple[str, int, bool]]]], output_path: str):
        from tools.io.image import texture_grid, load_image
        from tools.util.path_tools import replace_unallowed_chars
        from tools.transforms.to_tensor_image import ToTensorImage
        from tools.transforms.to_numpy_image import ToNumpyImage
        from tools.video.inpaint_writer import InpaintWriter
        logging.info("Loading texture editing images")
        numpyify_image = ToNumpyImage(output_dtype=np.uint8)

        texture_buffer: Dict[int, List[torch.Tensor, int, str, bool]] = dict()

        def load_texture(path: str) -> torch.Tensor:
            tensorify_image = ToTensorImage(output_dtype=torch.float32)
            if not os.path.exists(path):
                logging.error(f"Texture path does not exist: {path}.")
                return None
            texture = tensorify_image(load_image(
                path, size=model.camera._image_resolution.detach().cpu().tolist()))
            return texture

        def load_textures_for_node(node: Any, text: List[Union[Tuple[str, int], Tuple[str, int, bool]]]):
            texture_buffer[node.get_index()] = list()
            for i, item in enumerate(text):
                if len(item) == 2:
                    t, time = item
                    is_projected = False
                elif len(item) == 3:
                    t, time, is_projected = item
                loaded_texture = load_texture(t)
                if loaded_texture is not None:
                    texture_buffer[node.get_index()].append(
                        (loaded_texture, time, t, is_projected))
                else:
                    raise ValueError(f"Texture {t} could not be loaded.")

        for k, text in textures.items():
            # Check if object index exist.
            found_node = None
            if k == -1:
                # Background node
                found_node = next((x for x in model.objects if isinstance(
                    x, BackgroundImagePlaneSceneNode3D)), None)
            else:
                found_node = next(
                    (x for x in model.objects if x.get_index() == k), None)
            if found_node is None:
                raise ValueError(f"Object with index {k} not found.")
            # Load the textures for the node
            load_textures_for_node(found_node, text)

        for node_idx, texture_items in texture_buffer.items():

            node = next(
                (x for x in model.objects if x.get_index() == node_idx), None)
            if node is None:
                raise ValueError(f"Object with index {node_idx} not found.")

            for texti, (texture_image, time_index, texture_path, is_projected) in enumerate(texture_items):
                ref_time = self.runner.dataset.frame_timestamps[time_index]
                ref_time, _ = flatten_batch_dims(ref_time, -1)

                org_filename = os.path.splitext(
                    os.path.basename(texture_path))[0]
                filename = replace_unallowed_chars(node.get_name(
                ) + f"_texture_{time_index}_{org_filename}", allow_dot=False) + ".png"

                texture_path = os.path.join(
                    self.runner.config.output_path, "edits", filename)

                exists = False
                texture = None
                if os.path.exists(texture_path) and texti == 0:
                    exists = True
                    logging.info(
                        f"Texture already exists: {texture_path}. Loading.")
                    tensorify_image = ToTensorImage(output_dtype=torch.float32)
                    texture = tensorify_image(load_image(texture_path))
                    node._create_plain_texture_map(texture)
                else:
                    if is_projected:
                        logging.info(f"Apply already projected texture.")
                        node._create_plain_texture_map(texture_image)
                        texture = node.texture_map.detach().cpu()
                        exists = True
                    else:
                        logging.info(
                            f"Creating texture map this might take a while ...")
                        node.create_texture_map(
                            texture_image, t=ref_time, camera=model.camera)
                        texture = node.texture_map.detach().cpu()

                # Save the texture map to disk
                if not exists and texti == 0:
                    save_image(numpyify_image(texture), texture_path)
                    logging.info(f"Saved texture to {texture_path}")

            # Render model with the texture map
            node.render_texture_map = True

        times = self.runner.dataset.frame_timestamps
        bar = self.runner.config.progress_factory.bar(
            desc="Generating texture editing images", total=len(times), is_reusable=True, tag="texture-editing")

        texture_video_path = os.path.join(output_path, "video.mp4")
        base_path = os.path.join(output_path)

        if not os.path.exists(base_path):
            os.makedirs(base_path)
        if not os.path.exists(os.path.dirname(texture_video_path)):
            os.makedirs(os.path.dirname(texture_video_path))
        try:
            gen = model.query(t=times)
            with InpaintWriter(texture_video_path, fps=10, inpaint_counter=False) as writer:
                for i, t in enumerate(times):
                    image = next(gen)
                    image = numpyify_image(image)[0]
                    save_image(image, os.path.join(
                        base_path, f"{i:03d}_t.png"))
                    writer.write(image)
                    bar.update()
        finally:
            for node in model.objects:
                node.render_texture_map = False

    def compute_masked_grouped_metrics(self) -> pd.DataFrame:
        pattern = r"""(?P<folder>train/epoch/)(?P<is_mean>Mean_)?Mask_(?P<mask_id>\d+)_(?P<metric>MPSNR|SSIM)?(?:(?P<only_mask>_only_mask))?"""
        import re
        human_classes = ["Cyclist", "Pedestrian",
                         "Pedestrian Object", "Motorcyclist"]
        vehicle_classes = ["Car", "Truck", "Bus", "Motorcycle",
                           "Bicycle", "Trailer", "Other Large Vehicle"]
        runner = self.runner

        def get_human_vehicle_mapping(mapping: Dict[int, str]) -> Tuple[List[int], List[int]]:
            ret = dict()
            human = set(human_classes)
            veh = set(vehicle_classes)
            for k, v in mapping.items():
                if v in human:
                    ret[k] = "Human"
                elif v in veh:
                    ret[k] = "Vehicle"
                else:
                    print(f"Unmapped class {v} for id {k} ignoring")
            return ret

        def split_metrics(name: str) -> Optional[Dict[str, Any]]:
            match = re.search(pattern, name)
            if match is None:
                return None
            groups = match.groupdict()
            if "only_mask" in groups and groups["only_mask"] is not None:
                groups["only_mask"] = True
            else:
                groups["only_mask"] = False
            if "is_mean" in groups and groups["is_mean"] is not None:
                groups["is_mean"] = True
            else:
                groups["is_mean"] = False
            del groups["folder"]
            groups["name"] = name

            groups["mask_id"] = int(groups["mask_id"])
            return groups
        tracker = runner.tracker
        all_metrics = [split_metrics(
            x) for x in tracker.metrics.keys() if split_metrics(x)]
        mean_metrics = [x for x in all_metrics if x["is_mean"]]
        all_ids = list(set([x["mask_id"] for x in mean_metrics]))
        mappings = runner.dataset.get_semantic_mapping(all_ids)
        hum_vehic_mapping = get_human_vehicle_mapping(mappings)

        for m in mean_metrics:
            m["class"] = hum_vehic_mapping.get(m["mask_id"])
            m["value"] = tracker.metrics[m["name"]].values.iloc[-1]["value"]
            m["metric"] = m["class"] + " " + m["metric"]
            del m["is_mean"]
            del m["name"]

        vehicle_metrics = [x for x in mean_metrics if x["class"] == "Vehicle"]
        human_metrics = [x for x in mean_metrics if x["class"] == "Human"]
        frames = []

        if len(vehicle_metrics) > 0:
            vdf = pd.DataFrame(vehicle_metrics)
            vdf.drop(columns=["class"], inplace=True)
            agg_vdf = vdf.groupby(["metric", "only_mask"]).agg(
                {"mask_id": "count", "value": "mean"}).reset_index()
            frames.append(agg_vdf)

        if len(human_metrics) > 0:
            hdf = pd.DataFrame(human_metrics)
            hdf.drop(columns=["class"], inplace=True)
            agg_hdf = hdf.groupby(["metric", "only_mask"]).agg(
                {"mask_id": "count", "value": "mean"}).reset_index()
            frames.append(agg_hdf)

        if len(frames) == 0:
            print("No metrics found for model!")
            return pd.DataFrame(columns=["metric", "only_mask", "Object Count", "value"])

        agg_df = pd.concat(frames, axis=0).reset_index(drop=True)
        agg_df.rename(columns={"mask_id": "Object Count"}, inplace=True)

        for i, row in agg_df.iterrows():
            class_name = row["metric"].split(" ")[0]
            is_only_mask = row["only_mask"]
            metric = row["metric"].split(" ")[1]
            count = row["Object Count"]
            tag = metric + "_" + class_name
            value = row["value"]
            if is_only_mask:
                tag += "_only_mask"
            tracker.epoch_metric(tag, value, in_training=False,
                                 step=runner.model.current_epoch)
            if runner.experiment_logger is not None:
                runner.experiment_logger.add_scalar(
                    tag, value, step=runner.model.current_epoch, epoch=runner.model.current_epoch)
        return agg_df
    # endregion

# Editing Utility Functions


def compose_multi_objects(images: torch.Tensor,
                          object_nodes: Any,
                          cam: LearnedCameraSceneNode3D,
                          t: torch.Tensor,
                          smooth_borders: bool = True,
                          smooth_threshold: bool = True):
    t, _ = flatten_batch_dims(t, -1)
    ret = torch.zeros_like(images[0])
    T = t.shape[0]
    for tidx in range(T):
        zorder = simple_zorder(object_nodes, cam, t[tidx]).cpu()
        visible = (zorder > -1)
        zorder = zorder[visible]
        visible_objs = images[visible, tidx]
        # Order objects by zorder
        visible_objs = visible_objs[zorder]
        object_nodes_sorted = [object_nodes[i] for i in zorder]

        smoothing_fnc = get_linear_segmented_smoothing_fnc(
            threshold_lower=0, threshold_upper=20, slope=1/5)

        for i, obj in enumerate(visible_objs[:]):
            if smooth_borders:
                smooth_mask = object_nodes_sorted[i].get_smoothed_occluded_image_space(
                    cam, t[tidx], smoothing_fnc)
                visible_objs[i, 3:4] *= smooth_mask.squeeze(0)
            if smooth_threshold:
                visible_objs[i, 3:4] = linear_segmented_smoothing(
                    visible_objs[i, 3:4], (0.00, 0.08), (0, 0.08), slope=2, shift=-0.8)

        ret[tidx] = n_layers_alpha_compositing(visible_objs.permute(
            0, 2, 3, 1), torch.arange(len(visible_objs))).permute(2, 0, 1)
    return ret

# endregion
