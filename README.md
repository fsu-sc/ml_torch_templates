# PyTorch Template Project
PyTorch deep learning project made easy.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [PyTorch Template Project](#pytorch-template-project)
  - [Requirements](#requirements)
  - [Features](#features)
  - [Folder Structure](#folder-structure)
  - [Usage](#usage)
    - [Config file format](#config-file-format)
    - [Using config files](#using-config-files)
    - [Resuming from checkpoints](#resuming-from-checkpoints)
    - [Using Multiple GPU](#using-multiple-gpu)
  - [Speed optimizations](#speed-optimizations)
    - [Model compilation](#model-compilation)
    - [Mixed precision training](#mixed-precision-training)
    - [Matrix multiplication precision](#matrix-multiplication-precision)
  - [Customization](#customization)
    - [Project initialization](#project-initialization)
    - [Custom CLI options](#custom-cli-options)
    - [Data Loader](#data-loader)
    - [Trainer](#trainer)
    - [Model](#model)
    - [Loss](#loss)
    - [Metrics](#metrics)
    - [Additional logging](#additional-logging)
    - [Testing](#testing)
    - [Validation data](#validation-data)
    - [Checkpoints](#checkpoints)
    - [Tensorboard Visualization](#tensorboard-visualization)
  - [Contribution](#contribution)
  - [TODOs](#todos)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements
* Python >= 3.5 (3.6 recommended)
* PyTorch >= 0.4 (1.2 recommended)
* tqdm (Optional for `test.py`)
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))

## Features
* Clear folder structure which is suitable for many deep learning projects.
* `.yml` config file support for convenient parameter tuning.
* Customizable command line options for more convenient parameter tuning.
* Checkpoint saving and resuming.
* Abstract base classes for faster development:
  * `BaseTrainer` handles checkpoint saving/resuming, training process logging, and more.
  * `BaseDataLoader` handles batch generation, data shuffling, and validation data splitting.
  * `BaseModel` provides basic model summary.

## Folder Structure
  ```
  pytorch-template/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  │
  ├── config.yml - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── new_project.py - initialize new project with template files
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```

## Usage
The code in this repo is an MNIST example of the template.
Try `python train.py -c config.yml` to run code.

### Config file format
Config files are in `.yml` format:
```yaml
name: Mnist_LeNet
n_gpu: 1
amp: true
matmul_precision: medium
compile: false

arch:
  type: MnistModel
  args: {}

data_loader:
  type: MnistDataLoader
  args:
    data_dir: /Net/work/ozavala/DATA/ML_Template/
    batch_size: 10
    shuffle: true
    validation_split: 0.1
    num_workers: 2

optimizer:
  type: Adam
  args:
    lr: 0.001
    weight_decay: 0
    amsgrad: true

loss: nll_loss
metrics:
  - accuracy
  - top_k_acc

lr_scheduler:
  type: StepLR
  args:
    step_size: 50
    gamma: 0.1

trainer:
  epochs: 10
  save_dir: outputs/
  save_period: 10
  verbosity: 2
  monitor: min val_loss
  early_stop: 10
  tensorboard: true 
```

Add addional configurations if you need.

### Using config files
Modify the configurations in `.yml` config files, then run:

  ```
  python train.py --config config.yml
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```

### Using Multiple GPU
You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```
  python train.py --device 2,3 -c config.yml
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py
  ```

## Speed optimizations

### Model compilation

If you are using pytorch 1.1 or higher, you can use `torch.compile` to compile your model. You can turn on the option in the config file.

  ```yaml
  compile: true
  ```

### Mixed precision training

If you are using pytorch 1.1 or higher, you can use `torch.cuda.amp.autocast` to enable mixed precision training. You can turn on the option in the `config.yml` file.

  ```yaml
  amp: true
  ```

### Matrix multiplication precision

If you are using pytorch 1.1 or higher, you can use `torch.set_float32_matmul_precision` to set the precision of matrix multiplication. You can turn on the option in the config file.

```yaml
matmul_precision: medium
```

-The precision can be one of the following:
- `highest` – Maximum precision (Default), use FP32.
- `high` – Mixed precision. Either use TensorFloat32 or two bfloat16 to represent float32.
- `medium` – Reduced precision for speed. Use bfloat16 if allowed.

## Customization

### Project initialization
Use the `new_project.py` script to make your new project directory with template files.
`python new_project.py ../NewProject` then a new project folder named 'NewProject' will be made.
This script will filter out unneccessary files like cache, git files or readme file. 

### Custom CLI options

Changing values of config file is a clean, safe and easy way of tuning hyperparameters. However, sometimes
it is better to have command line options if some values need to be changed too often or quickly.

This template uses the configurations stored in the yml file by default, but by registering custom options as follows
you can change some of them using CLI flags.

  ```python
  # simple class-like object having 3 attributes, `flags`, `type`, `target`.
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
      CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
      CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
      # options added here can be modified by command line flags.
  ]
  ```
`target` argument should be sequence of keys, which are used to access that option in the config dict. In this example, `target` 
for the learning rate option is `('optimizer', 'args', 'lr')` because `config['optimizer']['args']['lr']` points to the learning rate.
`python train.py -c config.yml --bs 256` runs training with options given in `config.yml` except for the `batch size`
which is increased to 256 by command line options.


### Data Loader
* **Writing your own data loader**

1. **Inherit ```BaseDataLoader```**

    `BaseDataLoader` is a subclass of `torch.utils.data.DataLoader`, you can use either of them.

    `BaseDataLoader` handles:
    * Generating next batch
    * Data shuffling
    * Generating validation data loader by calling
    `BaseDataLoader.split_validation()`

* **DataLoader Usage**

  `BaseDataLoader` is an iterator, to iterate through batches:
  ```python
  for batch_idx, (x_batch, y_batch) in data_loader:
      pass
  ```
* **Example**

  Please refer to `data_loader/data_loaders.py` for an MNIST data loading example.

### Trainer
* **Writing your own trainer**

1. **Inherit ```BaseTrainer```**

    `BaseTrainer` handles:
    * Training process logging
    * Checkpoint saving
    * Checkpoint resuming
    * Reconfigurable performance monitoring for saving current best model, and early stop training.
      * If config `monitor` is set to `max val_accuracy`, which means then the trainer will save a checkpoint `model_best.pth` when `validation accuracy` of epoch replaces current `maximum`.
      * If config `early_stop` is set, training will be automatically terminated when model performance does not improve for given number of epochs. This feature can be turned off by passing 0 to the `early_stop` option, or just deleting the line of config.

2. **Implementing abstract methods**

    You need to implement `_train_epoch()` for your training process, if you need validation then you can implement `_valid_epoch()` as in `trainer/trainer.py`

* **Example**

  Please refer to `trainer/trainer.py` for MNIST training.

* **Iteration-based training**

  `Trainer.__init__` takes an optional argument, `len_epoch` which controls number of batches(steps) in each epoch.

### Model
* **Writing your own model**

1. **Inherit `BaseModel`**

    `BaseModel` handles:
    * Inherited from `torch.nn.Module`
    * `__str__`: Modify native `print` function to prints the number of trainable parameters.

2. **Implementing abstract methods**

    Implement the foward pass method `forward()`

* **Example**

  Please refer to `model/model.py` for a LeNet example.

### Loss
Custom loss functions can be implemented in 'model/loss.py'. Use them by changing the name given in "loss" in config file, to corresponding name.

### Metrics
Metric functions are located in 'model/metric.py'.

You can monitor multiple metrics by providing a list in the configuration file, e.g.:
  ```yaml
  metrics:
    - accuracy
    - top_k_acc
  ```

### Additional logging
If you have additional information to be logged, in `_train_epoch()` of your trainer class, merge them with `log` as shown below before returning:

  ```python
  additional_log = {"gradient_norm": g, "sensitivity": s}
  log.update(additional_log)
  return log
  ```

### Testing
You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument.

### Validation data
To split validation data from a data loader, call `BaseDataLoader.split_validation()`, then it will return a data loader for validation of size specified in your config file.
The `validation_split` can be a ratio of validation set per total data(0.0 <= float < 1.0), or the number of samples (0 <= int < `n_total_samples`).

**Note**: the `split_validation()` method will modify the original data loader
**Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`

### Checkpoints
You can specify the name of the training session in config files:
  ```yaml
  name: MNIST_LeNet
  ```

The checkpoints will be saved in `save_dir/name/timestamp/checkpoint_epoch_n`, with timestamp in mmdd_HHMMSS format.

A copy of config file will be saved in the same folder.

**Note**: checkpoints contain:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.mnt_best,
    'config': self.config
  }
  ```

### Tensorboard Visualization
This template supports Tensorboard visualization by using either  `torch.utils.tensorboard` or [TensorboardX](https://github.com/lanpa/tensorboardX).

1. **Install**

    If you are using pytorch 1.1 or higher, install tensorboard by 'pip install tensorboard>=1.14.0'.

    Otherwise, you should install tensorboardx. Follow installation guide in [TensorboardX](https://github.com/lanpa/tensorboardX).

2. **Run training** 

    Make sure that `tensorboard` option in the config file is turned on.

    ```
     "tensorboard" : true
    ```

3. **Open Tensorboard server** 

    Type `tensorboard --logdir saved/log/` at the project root, then server will open at `http://localhost:6006`

By default, values of loss and metrics specified in config file, input images, and histogram of model parameters will be logged.
If you need more visualizations, use `add_scalar('tag', data)`, `add_image('tag', image)`, etc in the `trainer._train_epoch` method.
`add_something()` methods in this template are basically wrappers for those of `tensorboardX.SummaryWriter` and `torch.utils.tensorboard.SummaryWriter` modules. 

**Note**: You don't have to specify current steps, since `WriterTensorboard` class defined at `logger/visualization.py` will track current steps.

## Contribution
Feel free to contribute any kind of function or enhancement, here the coding style follows PEP8

Code should pass the [Flake8](http://flake8.pycqa.org/en/latest/) check before committing.

## TODOs

- [ ] More models
- [ ] More examples
- [ ] Multiple optimizers
- [ ] Support more tensorboard functions

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgements
This project is inspired by the project [Pythorch-Template](https://github.com/victoresque/pytorch-template) by [Victor Huang](https://github.com/victoresque), which was inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)
