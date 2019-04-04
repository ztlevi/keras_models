import math

import numpy as np
from PIL import Image
from tensorflow import keras

import cv2
from training.age import Linear_1_bias, coral_loss, mae_pred

loss = coral_loss(np.ones(101 - 1))
model = keras.models.load_model(
    "/home/ztlevi/Developer/keras_models/outputs/checkpoints/age_mobilenet_v1_utkface/ckpt.h5",
    custom_objects={"loss": loss, "mae_pred": mae_pred, "Linear_1_bias": Linear_1_bias},
)
video = cv2.VideoCapture(0)

y_preds = []
while True:
    _, frame = video.read()

    # Convert the captured frame into RGB
    im = Image.fromarray(frame, "RGB")
    w, h = im.size
    im = im.crop((200, 130, w - 200, h - 130))
    cv2.imshow("cropped", np.array(im))

    # Resizing into 128x128 because we trained the model with this image size.
    im = im.resize((224, 224))
    img = np.array(im)
    norm_img = cv2.normalize(
        img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    # Our keras model used a 4D tensor, (images x height x width x channel)
    # So changing dimension 128x128x3 into 1x128x128x3
    img_array = np.expand_dims(norm_img, axis=0)

    # Calling the predict method on model to predict 'me' on the image
    pred = model.predict(img_array)
    pred = pred > 0.5
    y_pred = np.sum(pred)

    y_preds += [y_pred]
    if len(y_preds) > 15:
        y_preds.pop(0)
    pred_age = int(np.mean(y_preds))

    # if prediction is 0, which means I am missing on the image, then show the frame in gray color.
    # if prediction == 0:
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Age: {}".format(pred_age), (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("AGE", frame)
    # cv2.imshow("AGE", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
video.release()
cv2.destroyAllWindows()
