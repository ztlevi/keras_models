# Age Estimation

## Dataset

1. [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/): 505310 automatically annotated
   images. Labels including age and gender are not very accuracte.

   The age of a person can be calculated based on the date of birth and the time when the photo was
   taken (note that we assume that the photo was taken in the middle of the year):

   ```
   [age,~]=datevec(datenum(wiki.photo_taken,7,1)-wiki.dob);
   ```

2. [UTKFace](http://aicip.eecs.utk.edu/wiki/UTKFace): 23705 manually annotated images.

## Training

Using CORAL loss function defined in [this paper](https://arxiv.org/abs/1901.07884).

1.  Fine tuning on MobilenetV1 pre-trained weights for 3 epochs with only last FC layer trainable.
2.  Train the model on IMDB-WIKI dataset for 12 epochs. Pick the checkpoint with lowest `val_mae`.
3.  Fine-tuned on Audience dataset for 60 epochs. Select the checkpoint with lowest `val_mae`.

Training scripts refer to the repo
[here](http://rnd-github-usa-g.huawei.com/BostonResearchCenter/keras_models/tree/master/training/age).

## CORAL Loss

- Weight importance

```python
def task_importance_weights(label_array, num_classes):
    uniq = np.unique(label_array)
    num_examples = label_array.shape[0]

    m = np.zeros(num_classes-1)

    for i, t in enumerate(np.arange(np.min(uniq), np.max(uniq))):
        m_k = np.max([label_array[label_array > t].shape[0], num_examples - label_array[label_array > t].shape[0]])
        m[i] = np.sqrt(m_k)

    imp = m / np.max(m)
    return imp
```

- loss function

```python
def coral_loss(imp):
    def loss(levels, logits):
        val = -K.sum(
            (K.log(K.sigmoid(logits)) * levels + (K.log(K.sigmoid(logits)) - logits) * (1 - levels))
            * tf.convert_to_tensor(imp, dtype=tf.float32),
            axis=1,
            )
        return K.mean(val)

    return loss
```

- Ground truth:

```python
levels = [[1] * label + [0] * (self.num_classes - 1 - label) for label in batch_y]
```

for example: age 10 is represented as `[1]*10 + [0]*(101-1-10)`

- Layer design

```python
model = keras.applications.mobilenet.MobileNet(
    input_shape=input_shape, weights="imagenet", include_top=False
)

x = model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.001)(x)
x = keras.layers.Dense(1, use_bias=False)(x)
x = Linear_1_bias(num_classes)(x)
```

## Performance

MAE on IMDB-WIKI 8.072

MAE on UTKFace 5.16
