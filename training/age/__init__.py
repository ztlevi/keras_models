from tensorflow import keras


# MAE
def mae_pred(y_true, y_pred):
    return keras.backend.sum(keras.backend.abs(y_true - y_pred)) / 64
