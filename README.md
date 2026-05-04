<div align="center" style="padding: 20px 0; border-bottom: 2px solid #EEEEEE; margin-bottom: 20px;">
    <h1 style="color: #2c3e50; font-size: 2.2em; margin-bottom: 0.2em;">
        Neural Atlas Graphs </br> for Dynamic Scene Decomposition and Editing
    </h1>
    <p style="font-size: 1.1em; color: #34495e; margin-top: 15px; margin-bottom: 5px;">
        <a href="https://github.com/jp-schneider" style="color: #34495e; text-decoration: none;"><strong>Jan Philipp Schneider</strong></a><sup>1,2</sup>,
        <a href="./README.md" style="color: #34495e; text-decoration: none;"><strong>Pratik Singh Bisht</strong></a><sup>1</sup>,
        <a href="https://ilyac.info/" style="color: #34495e; text-decoration: none;"><strong>Ilya Chugunov</strong></a><sup>2</sup>,
        <a href="https://www.cg.informatik.uni-siegen.de/de/kolb-andreas" style="color: #34495e; text-decoration: none;"><strong>Andreas Kolb</strong></a><sup>1</sup>,
        <a href="https://sites.google.com/site/michaelmoellermath/" style="color: #34495e; text-decoration: none;"><strong>Michael Moeller</strong></a><sup>1,3</sup>,
        <a href="https://www.cs.princeton.edu/~fheide/" style="color: #34495e; text-decoration: none;"><strong>Felix Heide</strong></a><sup>2,4</sup>
    </p>
    <p style="font-size: 0.9em; color: #7f8c8d; line-height: 1.6;">
        <sup>1</sup>University of Siegen &nbsp;  &nbsp;
        <sup>2</sup>Princeton University &nbsp;  &nbsp;
        <sup>3</sup>Lamarr Institute &nbsp;  &nbsp;
        <sup>4</sup>Torc Robotics
        <br>
    </p>
    <p>&#x1F389 NeurIPS 2025 (spotlight) &#x1F389</p>
</div>


<p align="center">
    <a href="https://princeton-computational-imaging.github.io/nag/" alt="Project Page">
        <img src="https://img.shields.io/badge/Project%20Page-gray?logo=computer&logoColor=red" /></a>
    <a href="https://neurips.cc/virtual/2025/loc/san-diego/poster/115926">
        <img src="./data/static/badge/neurips_2025.svg"/>
    </a>
    <a href="./static/pdfs/nag.pdf" alt="Paper PDF File">
        <img src="https://img.shields.io/badge/Paper-gray?logo=files&logoColor=red" /></a>
    <a href="./static/pdfs/nag-supplementary.pdf" alt="Supplementary PDF File">
        <img src="https://img.shields.io/badge/Supplementary-gray?logo=files&logoColor=red" /></a>
    <a href="https://arxiv.org/abs/2509.16336">
        <img src="https://img.shields.io/badge/arXiv-2509.16336-b31b1b.svg" />
    </a>
    <a href="https://openreview.net/forum?id=pkuVonMwhT" alt="OpenReview">
    <img src="https://img.shields.io/badge/OpenReview-8c1b13?logo=file" /></a>

</p>
<p align="center">
    <h3 align="center">"Neural Atlas Graphs enable high-quality dynamic scene decomposition and intuitive 2D appearance editing, with use-cases in autonomous driving and videography."
    </h3>
</p>
<br>
<div style="margin: 0 auto; padding: 0px;">
    <table border="0" cellpadding="0" cellspacing="0" style="width: 100%; border-collapse: collapse;">
        <tbody>
            <!-- Row 1 -->
            <tr>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/W203/W203_060_002_GT.png" alt="Ground Truth Image 1 - Time 002" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/W203/W203_060_005_GT.png" alt="Ground Truth Image 2 - Time 005" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/W203/W203_060_015_GT.png" alt="Ground Truth Image 3 - Time 015" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/W203/W203_060_025_GT.png" alt="Ground Truth Image 4 - Time 025" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/W203/W203_060_removed_objects.png" alt="Decomposed Objects" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
            </tr>
            <!-- Row 2 -->
            <tr>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/W203/W203_060_002_edit.png" alt="Edited Image 1 - Time 002" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/W203/W203_060_005_edit.png" alt="Edited Image 2 - Time 005" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/W203/W203_060_015_edit.png" alt="Edited Image 3 - Time 015" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/W203/W203_060_025_edit.png" alt="Edited Image 4 - Time 025" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/W203/speed_limit_zebra_resized.png" alt="Edit Texture" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
            </tr>
        </tbody>
    </table>
    <h5 align="left">Dynamic Scene Editing in Waymo S-203. Ground Truth (top) vs. NAG Edits (bottom) across four frames. The final column shows the texture source (bottom) and removed objects (top). Note the realistic, consistent blending of the foreground car and its shadow into the edited scene.</h5>
</div>
<br>
<div style="margin: 0 auto; padding: 0px;">
    <table border="0" cellpadding="0" cellspacing="0" style="width: 100%; border-collapse: collapse;">
        <tbody>
            <tr>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/blackswan/gt/00005.jpg" alt="Ground Truth Image 1 - Time 05" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/blackswan/gt/00015.jpg" alt="Ground Truth Image 2 - Time 15" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/blackswan/gt/00030.jpg" alt="Ground Truth Image 3 - Time 30" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/blackswan/gt/00045.jpg" alt="Ground Truth Image 4 - Time 45" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
            </tr>
            <tr>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/blackswan/white/005_t.png" alt="Edited Image 1 - Time 05" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/blackswan/white/015_t.png" alt="Edited Image 2 - Time 15" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/blackswan/white/030_t.png" alt="Edited Image 3 - Time 30" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/blackswan/white/045_t.png" alt="Edited Image 4 - Time 45" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
            </tr>
            <tr>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/blackswan/rainbow/005_t.png" alt="Edited Image 1 - Time 05" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/blackswan/rainbow/015_t.png" alt="Edited Image 2 - Time 15" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/blackswan/rainbow/030_t.png" alt="Edited Image 3 - Time 30" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
                <td align="center" style="padding: 2px;"><a href="#"><img src="./data/static/images/blackswan/rainbow/045_t.png" alt="Edited Image 4 - Time 45" style="width: 100%; height: auto; border-radius: 4px;"></a></td>
            </tr>
        </tbody>
    </table>
    <h5 align="left"><bold>Seamless Texture Transfer and Propagation. We utilize image generation models to create new textures for the black swan (top). The NAG effectively projects textures (white, rainbow) onto the dynamic 3D object, ensuring robust temporal coherence throughout the sequence (bottom rows).</h5>
</div>
        
-----    
This repository contains the official implementation of **Neural Atlas Graphs for Dynamic Scene Decomposition and Editing**, a novel hybrid scene representation for learning editable high-resolution dynamic scenes. Neural Atlas Graphs (NAG) integrate the editability of neural atlases with the complex spatial reasoning of scene graphs, where each graph node is a view-dependent neural atlas. This allows for both intuitive 2D appearance editing and consistent 3D ordering and positioning of scene elements.

## 🛠️ Installation

Please refer to the [installation instructions](docs/installation.md) for setting up the repository, its dependencies and data.

## 🚀 NAG Training

Given a proper python environment setup, one can run our method using:

```bash
python nag/scripts/run_nag.py --config-path [path-to-config]
```

For more details, please refer to the [training instructions](docs/training.md).

During the training process and afterwards, the model is evaluated to produce outputs for all frames, calculating metrics, as well as scene decompositions for every object.

### 🔁 Reproducibility

We are committed to full reproducibility of the results presented in our paper. All configuration files and training procedures are provided in this repository. We provide detailed instructions on how to reproduce our experiments in the [reproducibility](docs/reproducibility.md) document.
Further, we provide the datasets and an explanation how to set these up in our [datasets setup](docs/datasets.md) document.

In the future, we plan to provide further scripts to convert additional Waymo segments and Davis sequences into our used formats. Create a GitHub issue if you are interested in this or have any questions.

### 📝 Working with NAGs

We provide a [Jupyter Notebook](notebooks/working_with_nags.ipynb) showcasing how to load a pre-trained NAG model, decompose scenes into objects, and perform texture editing. This notebook serves as a practical guide for utilizing the capabilities of Neural Atlas Graphs.

## 🧠 NAG Code Structure

To briefly outline the code structure of our repository, we provide a high-level overview of the main components and their locations within the codebase.

Our model training and evaluations are encapsulated using a dedicated [runner](nag/run/nag_runner.py), which holds instances of the model, dataset, and all other training related components. The runner can be created to train a new model, or load an existing one to further evaluate it. As we are relying on pytorch lightning for training, we implemented a [callback class](nag/callbacks/nag_callback.py), which control training progress and handles the evaluation of the model.General tools and utility functions are within a dedicated [tools](tools/)
library, which need to be included using git-submodules.

### NAG Core Components
Further, we briefly point out the location of the NAG core components within the repository.

- The NAG Model is located at [nag/model/nag_functional_model.py](nag/model/nag_functional_model.py) yielding the compositions of all the nodes, and contains the rendering code.
- The forground nodes implementation is located at [nag/model/view_dependent_image_plane_scene_node_3d.py](nag/model/view_dependent_image_plane_scene_node_3d.py) and its base classes up to the definition of [nag/model/learned_image_plane_scene_node_3d.py](nag/model/learned_image_plane_scene_node_3d.py) which includes the networks definitions.
- The background node is implemented within [nag/model/view_dependent_background_image_plane_scene_node_3d.py](nag/model/view_dependent_background_image_plane_scene_node_3d.py) and its base class [nag/model/background_image_plane_scene_node_3d.py](nag/model/background_image_plane_scene_node_3d.py).
- The editing functionalities are implemented within a mixin class [nag/model/texture_mappable_scene_node_3d.py](nag/model/texture_mappable_scene_node_3d.py).
- The camera is implemented within [nag/model/learned_camera_scene_node_3d.py](nag/model/learned_camera_scene_node_3d.py) and its base class [nag/model/timed_camera_scene_node_3d.py](nag/model/timed_camera_scene_node_3d.py).

Surely, there is way more to explore & explain, so feel free to open issues or discussions on GitHub if you have any questions regarding the code structure or implementation details.

## 📜 Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{Schneider2025NAG,
    author = {Schneider, Jan Philipp and Bisht, Pratik S. and Chugunov, Ilya and Kolb, Andreas and Moeller, Michael and Heide, Felix},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {D. Belgrave and C. Zhang and H. Lin and R. Pascanu and P. Koniusz and M. Ghassemi and N. Chen},
    pages = {14481--14500},
    publisher = {Curran Associates, Inc.},
    title = {Neural Atlas Graphs for Dynamic Scene Decomposition and Editing},
    url = {https://proceedings.neurips.cc/paper_files/paper/2025/file/1534649513e77f7de77772f143799b13-Paper-Conference.pdf},
    volume = {38},
    year = {2025}
}
```

Thanks for your interest in our work! We hope you find Neural Atlas Graphs as exciting and useful as we do. If you have any questions, suggestions, or feedback, please don't hesitate to reach out via GitHub issues or discussions.

