import os

import numpy as np
import cv2
import json

from src.hdf5_manager import hdf5_writer
from src.database import session, dump_entry, get_by_hdf_index, Entry
from src.palette_kmeans import KMeanPaletteClassifier

# EXTRACTION #
objects = []
def compute_feature_over_db():
    cl = KMeanPaletteClassifier()
    total = len(session.query(Entry).all())
    for i, d in enumerate(session.query(Entry).all()): #type:Entry
        img = cv2.imread(d.thumbnail_path)
        pred = cl.fit(img)[0]
        print(i, total)
        d.color_labels = json.dumps(pred)

    session.commit()


# hdf5_writer.set_path("data/test-features.hdf5", mode="r+")
# hdf5_writer.initialize_dataset("xception_features", shape=(10*10*2048, ), dtype=np.float16)
try:
    compute_feature_over_db()
    hdf5_writer.on_close()
except Exception as e:
    session.rollback()
    raise e


# TESTING #
from random import sample
t = session.query(Entry).all()
t = sample(t, 100)

for k in t: #type:Entry
    print(k)
    cv2.imshow("out", cv2.imread(k.thumbnail_path))
    print(k.get_colors())
    cv2.waitKey()<