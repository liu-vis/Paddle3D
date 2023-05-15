# NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction

## Contents

* [Reference](#1)
* [Introduction](#2)
* [Model Zoo](#3)
* [Training Configuration](#4)
* [Tutorials](#5)
    * [Data Preparation](#51)
    * [Training](#52)
    * [Evaluation](#53)

## <h2 id="1">Reference</h2>

```bibtex
@article{wang2021neus,
  title={NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction},
  author={Wang, Peng and Liu, Lingjie and Liu, Yuan and Theobalt, Christian and Komura, Taku and Wang, Wenping},
  journal={arXiv preprint arXiv:2106.10689},
  year={2021}
```

## <h2 id="2">Introduction</h2>

NeuS

NeuS is an approach to static object/scene reconstruction, published in NeurIPS 2021. The authors are affiliated with The University of Hong Kong, the Max Planck Institute for Informatics, and Texas A&M University. Given a set of images of a static object and the corresponding camera poses, NeuS is capable of reconstructing the surface geometry of the object. It learns an SDF (Signed Distance Field) using an MLP (Multi-Layer Perceptron) supervised by the provided RGB images. It's noteworthy that the quality of the images and the accuracy of the camera poses influence the reconstruction results.

## <h2 id="3">Model Zoo</h2>

- Benchmarks on Blender Dataset.

| Scene |  PSNR   |  SSIM  |                                               Download                                               |                      Configuration                      |
|:-----:|:-------:|:------:|:----------------------------------------------------------------------------------------------------:|:-------------------------------------------------------:|
| dtu105 | 31.23 | 0.7889 | [model](-- TO BE UPLOADED) | [config](../../../configs/neus/llff_data.yml) |


## <h2 id="4">Training Configuration</h2>

For training configuration on open source datasets, refer
to [NeuS Training Configuration](../../../configs/neus). Among them, `llff_data_efficient.yml` uses the ray sampling strategy proposed in [Instant-ngp](https://arxiv.org/abs/2201.05989), which speedups the training process by almost 3 times [TO BE UPDATED].

## <h2 id="5">Tutorials</h2>

### <h3 id="51">Data Preparation</h3>

Soft link the dataset file to `PaddleRender/data/` or specify the dataset path
in [configuration file](../../../configs/neus).

### <h3 id="52">Training</h3>

At `PaddleRendering/`, execute:

```shell
export PYTHONPATH='.'

# Train on single GPU
python tools/train.py \
  --config configs/neus/llff_data.yml \
  --save_dir neus_dtu \
  --log_interval 1000 \
  --save_interval 10000

# Train on multiple GPUs (GPU 0, 1 as an example)
python -m paddle.distributed.launch --devices 0,1 \
    tools/train.py \
    --config configs/neus/llff_data.yml \
    --save_dir neus_dtu \
    --log_interval 1000 \
    --save_interval 10000
```

The training script accepts the following arguments and options:

| Arguments & Options         | Explanation                                                                                                                                                                                                                                                                                               | Required | Defaults                         |
|:----------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------|:---------------------------------|
| config                      | Configuration file.                                                                                                                                                                                                                                                                                       | YES      | -                                |
| image_batch_size            | Batch size of images, from which rays are sampled every iteration.<br>If `-1`, rays are sampled from the entire dataset.                                                                                                                                                                                  | NO       | -1                               |
| ray_batch_size              | Batch size of rays, the number of rays sampled from image mini-batch every iteration.                                                                                                                                                                                                                     | NO       | 1                                |
| image_resampling_interval   | To accelerate training, each GPU maintains a image buffer (image mini-batch is prefetched, rays are sampled from the buffer every iteration).<br>This argument specifies the interval of updating the image buffer. If `-1`, the buffer is never updated (for the case where `image_batch_size` is `-1`). | NO       | -1                               |
| use_adaptive_ray_batch_size | Whether to use an adaptive `ray_batch_size`.<br>If enabled, the number of valid samples fed to the model is stable at `2^18`, which accelerates model convergence.                                                                                                                                        | NO       | FALSE                            |
| iters                       | The number of iterations.                                                                                                                                                                                                                                                                                 | NO       | Specified in configuration file. |
| learning_rate               | Learning rate.                                                                                                                                                                                                                                                                                            | NO       | Specified in configuration file. |
| save_dir                    | Directory where models and VisualDL logs are saved.                                                                                                                                                                                                                                                       | NO       | output                           |
| save_interval               | Interval of saving checkpoints.                                                                                                                                                                                                                                                                           | NO       | 1000                             |
| do_eval                     | Whether to do evaluation after checkpoints are saved.                                                                                                                                                                                                                                                     | NO       | FALSE                            |
| resume                      | Whether to resume interrupted training.                                                                                                                                                                                                                                                                   | NO       | FALSE                            |
| model                       | Path to pretrained model file (`.pdparams`).                                                                                                                                                                                                                                                              | NO       | No pretrained model.             |
| log_interval                | Interval for logging.                                                                                                                                                                                                                                                                                     | NO       | 500                              |
| keep_checkpoint_max         | The maximum number of saved checkpoints (When the number of saved checkpoint exceeds the limit, the oldest checkpoint is automatically deleted).                                                                                                                                                          | NO       | 5                                |

### <h3 id="53">Evaluation</h3>

At `PaddleRendering/`, execute:

```shell
export PYTHONPATH='.'
python tools/evaluate.py \
  --config configs/neus/llff_data.yml \
  --model neus_dtu/iter_300000/model.pdparams
```

At the end of the evaluation, the rendering results will be saved in the directory specified by `--model`.

The evaluation script accepts the following arguments and options:

| Arguments & Options | Explanation                                                                                        | Required | Defaults |
|:--------------------|:---------------------------------------------------------------------------------------------------|:---------|:---------|
| config              | Configuration file.                                                                                | YES      | -        |
| model               | Model to be evaluated.                                                                             | YES      | -        |
| ray_batch_size      | Ray batch size.                                                                                    | NO       | 256    |
| num_workers         | The number of subprocess to load data, `0` for no subprocess used and loading data in main process | NO       | 0        |
