# PARIS3D: Reasoning-based 3D Part Segmentation Using Large Multimodal Model
This is the official implementation of "PARIS3D: Reasoning-based 3D Part Segmentation Using Large Multimodal Model".

We propose a model that is capable of segmenting parts of 3D objects based on implicit textual queries and generating natural language explanations corresponding to 3D object segmentation requests. Experiments show that our method achieves competitive performance to models that use explicit queries, with the additional abilities to identify part concepts, reason about them, and complement them with world knowledge.


<p align="center">
<img src="fig/redintro.png" alt="teaser">
results on PartNetE dataset
</p>

<p align="center">
<img src="fig/realpc.png" alt="real_pc">
results on real-world (iPhone-scanned) point clouds
</p>

## Abstract 
Recent advancements in 3D perception systems have significantly improved their ability to perform visual recognition tasks such as segmentation. However, these systems still heavily rely on explicit human instruction to identify target objects or categories, lacking the capability to actively reason and comprehend implicit user intentions. We introduce a novel segmentation task known as reasoning part segmentation for 3D objects, aiming to output a segmentation mask based on complex and implicit textual queries about specific parts of a 3D object.

## Installation

### Create a conda envrionment and install dependencies.
```
conda env create -f environment.yml
conda activate paris3d
```
### Install LISA
```
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```
### Install PyTorch3D

We utilize [PyTorch3D](https://github.com/facebookresearch/pytorch3d) for rendering point clouds. Please install it by the following commands or its [official guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md):
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git" 
```
### Install cut-pursuit
We utilize [cut-pursuit](https://github.com/loicland/superpoint_graph) for computing superpoints. Please install it by the following commands or its official guide:
```
CONDAENV=YOUR_CONDA_ENVIRONMENT_LOCATION
cd partition/cut-pursuit
mkdir build
cd build
cmake .. -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.9.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.9 -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
make
```

## Quick-Demo
### Download pretrained checkpoints
You can find the pre-trained checkpoint from [here](https://huggingface.co/Amrinkar/PARIS3D).

### Inference
install yacs==0.1.8
We provide 5 example point cloud files in `examples/`. You can use the following command to run both zero-shot and few-shot inferences for them after downloading 5+1 checkpoint files.
```
python3 demo.py
```
The script will generate following files:
```
rendered_img/: rendering of the input point cloud from 10 different views.
glip_pred/: GLIP predicted bouning boxes for each view.
superpoint.ply: Generated super points for the input point cloud used for converting bounding boxes to 3D segmentation. Different super points are in different colors.
semantic_seg/: visualization of semantic segmentation results for each part. Colored in white or black.
instance_seg/: visualization of instance segmentation results for each part. Different part instances in different color.
```
You can also find the example output from [here](https://huggingface.co/datasets/minghua/PartSLIP/tree/main/).

### Evaluation
export LD_LIBRARY_PATH=/home/amrin.kareem/.conda/envs/partslip/lib:$LD_LIBRARY_PATH
`sem_seg_eval.py` provides a script to calculate the mIoUs reported in the paper. 

## RPSeg Dataset
You can find the PartNet-Ensembled dataset used in the paper from [here](https://huggingface.co/datasets/minghua/PartSLIP/tree/main/).
```
PartNetE_meta.json: part names trained and evaluated of all 45 categories.
split: 
    - test.txt: list of models used for testing (1,906 models)
    - few-shot.txt: list of models used for few-shot training (45x8 models)
    - train.txt: list of models used for training (extra 28k models for some baselines)
data:
    - test
        - Chair
            - 179
                - pc.ply: input colored point cloud file
                - images: rendered images of the 3D mesh, used for generation of the input point cloud (multi-view fusion)
                - label.npy: ground truth segmentation labels
                    - semantic_seg: (n,), semantic segmentation labels, 0-indexed, corresponding to the part names in PartNetE_meta.json, -1 indicates not belonging to any parts.
                    - instance_seg: (n,), instance segmentation labels, 0-indexed, each number indicates a instance, -1 indicates not belonging to any part instances.
            ...
        ...
    - few_shot
```

## Tips:
1. We assume dense and colored input point cloud, which is typically available in real-world applications.
2. You don't need to load the same checkpoints multiple times when batch evaluation.
3. You can reuse the superpoint results across different evaluations (e.g., zero- and few-shot) for the same input point cloud.
4. If you find the results unsatisfactory (e.g., when you change the number of input points or change to other datasets), you may want to tune the following paramters:

    a. point cloud rendering: `point_size` in `src/render_pc.py::render_single_view()`. You can change the point size to ensure realistic point cloud renderings.

    b. superpoint generation: `reg` in `src/gen_superpoint.py::gen_superpoint()`. This parameters adjust the granuarlity of the super point generation. You may want to ensure the generated superpoints are not too coarse-grained (e.g., multiple chair legs are not segmented) or fine-grained (e.g., too many small super points).

5.  For zero-shot text prompt, simply concatenating all part names (e.g., "arm, back, seat, leg, wheel") is sometimes better than including the object category as well (e.g., "arm, back, seat, leg, wheel of a chair", used in the paper). The mIoUs are 27.2 and 34.8 in our experiments.
6.  For zero-shot inference, you can change the prompts without extra training. Whereas for few-shot inference, changing prompts requires retraining.
 
## Citation

If you find our code helpful, please cite our paper:

```
@article{liu2022partslip,
  title={PartSLIP: Low-Shot Part Segmentation for 3D Point Clouds via Pretrained Image-Language Models},
  author={Liu, Minghua and Zhu, Yinhao and Cai, Hong and Han, Shizhong and Ling, Zhan and Porikli, Fatih and Su, Hao},
  journal={arXiv preprint arXiv:2212.01558},
  year={2022}
}
```
