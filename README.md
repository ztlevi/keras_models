# Keras Models

This repo contains trainig/evaluating/testing codes for various models using tensorflow keras.

## Installation

```python
conda create -n tf python=3.6
source activate tf
conda install -c conda-forge tensorflow-gpu=1.12.0 scikit-image
conda install -c menpo opencv3
```

## Basic Information

- Developer: Ting Zhou

## Project Structure

- dataset: various dataset creation scripts
- net: ML base models
- outputs: where the project's outputs go
- training: training scripts seperated into different modules, containing training from scratch and
  fine tuning scripts
- utils: utilities scripts, e.g. prerequsites funcitons and model freezing

## Known Issues

1. `TypeError: '<' not supported between instances of 'dict' and 'float'`:

   Error when using keras to load mobilnet checkpoint. The tensorflow version I am using is 1.12.
   You can solve this problem refering to the answer
   [here](https://github.com/tensorflow/tensorflow/issues/22697).

   **Temporary Fix**: After the `super()` call in the ReLu `init()` function in
   `tensorflow/python/keras/layers/advanced_activations.py` (around line 310), add the following
   lines of code:

```python
    if type(max_value) is dict:
        max_value = max_value['value']
    if type(negative_slope) is dict:
        negative_slope = negative_slope['value']
    if type(threshold) is dict:
        threshold = threshold['value']
```
