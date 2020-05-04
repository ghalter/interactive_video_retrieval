import json
import cv2
import os

import numpy as np
from sklearn.cluster import KMeans


with open(os.path.abspath(os.path.split(__file__)[0]) + "/colours.json", "r") as f:
    simple_colors = json.load(f)


class ColorClassifier:
    def __init__(self, colors=simple_colors):
        self.colors = np.array([c['lab'] for c in colors])
        self.names = [c['name'] for c in colors]

    def classify(self, lab):
        """
        Since the LABColorSpace is perceptually uniform (at least more or less),
        we can apply a simple euclidean distance to map a given color to the closest given color in the set.

        :param lab:
        :return:
        """
        t = np.argmin(np.linalg.norm(self.colors-lab, axis=1))
        return self.names[t]


class KMeanPaletteClassifier:
    """
    Computes a color palette and extracts color names for the given image.
    Pipeline: LAB -> SeedsSuperPixel -> KMEANS Palette -> Color Names mapping
    """
    def __init__(self):
        self.model = None
        self.model_shape = None
        self.ccl = ColorClassifier()

    def fit(self, bgr):
        lab_uint = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab_float = cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)

        if lab_uint.shape != self.model_shape:
            self.model = cv2.ximgproc.createSuperpixelSEEDS(lab_uint.shape[1], lab_uint.shape[0], 3, 200,
                                                                     num_levels=6, histogram_bins=8)
        self.model.iterate(lab_uint)
        mask = self.model.getLabels()

        q = np.zeros(shape=lab_float.shape[:2])
        colors = []
        c = 0

        for lbl in np.unique(self.model.getLabels()):
            indices = np.where(mask == lbl)
            col = np.mean(lab_float[indices], axis=0)
            q[indices] = c
            c += 1
            colors.append(col)

        colors = np.array(colors, dtype=np.float32)
        kmeans = KMeans(n_clusters=20, random_state=0).fit(colors)
        res = np.zeros_like(lab_float)
        color_names = []

        for i in range(len(colors)):
            indices = np.where(q == i)
            col = kmeans.cluster_centers_[kmeans.labels_[i]]

            name = self.ccl.classify(col)
            color_names.append(name)

            res[indices] = col
        color_names = list(set(color_names))
        return color_names, res
