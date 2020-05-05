import json

from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.applications.xception import preprocess_input, decode_predictions
import numpy as np

xception_model = None

with open("data/labels.json", "r") as f:
    labels = json.load(f)
    print(labels)
    labels = [l[1] for l in labels.values()]
    print(labels)


def init_xception(include_top = True):
    global xception_model
    xception_model = Xception(include_top=include_top, weights='imagenet')


def xception_process(image_path, k = 5, threshold=None):
    """
    Extracts returns the k top predictions.

    :param image_path:
    :param k:
    :return:
    """
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = xception_model.predict(x)
    preds = decode_predictions(preds, top=k)[0]

    if threshold is not None:
        preds = [dict(desc=p[1], prob=float(p[2]) ) for p in preds if p[2] >= threshold]
    else:
        preds = [dict(desc=p[1], prob=float(p[2])) for p in preds]
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)

    return preds


def xception_feature(image_path):
    """
    Extracts returns the k top predictions.

    :param image_path:
    :param k:
    :return:
    """
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = xception_model.predict(x)

    return np.reshape(preds, preds.shape[0] * preds.shape[1] * preds.shape[2] * preds.shape[3])
