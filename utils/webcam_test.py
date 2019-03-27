import numpy as np
from PIL import Image

import cv2
# from keras import models
from tensorflow import keras

# Load the saved model
from training.age import mae_pred

model = keras.models.load_model(
    "/home/ztlevi/Developer/keras_models/outputs/checkpoints/age_mobilenet_v1_utkface/ckpt.h5", custom_objects={'mae_pred': mae_pred}
)
video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()

    # Convert the captured frame into RGB
    im = Image.fromarray(frame, "RGB")
    w, h = im.size
    im = im.crop((200, 130, w-200, h-130))
    cv2.imshow("cropped", np.array(im))

    # Resizing into 128x128 because we trained the model with this image size.
    im = im.resize((224, 224))
    img = np.array(im)
    norm_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Our keras model used a 4D tensor, (images x height x width x channel)
    # So changing dimension 128x128x3 into 1x128x128x3
    img_array = np.expand_dims(norm_img, axis=0)

    # Calling the predict method on model to predict 'me' on the image
    prediction = np.argmax(model.predict(img_array), axis=1)[0]

    # if prediction is 0, which means I am missing on the image, then show the frame in gray color.
    # if prediction == 0:
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Age: {}".format(prediction), (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
    print(prediction)

    cv2.imshow("AGE", frame)
    # cv2.imshow("AGE", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
video.release()
cv2.destroyAllWindows()
