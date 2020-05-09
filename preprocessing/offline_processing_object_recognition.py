import os

import numpy as np
import cv2
import json

from src.hdf5_manager import hdf5_writer
from src.database import session, dump_entry, get_by_hdf_index, Entry
from src.object_recognition import init_xception, xception_process



objects = []
def compute_feature_over_db():
    total = len(session.query(Entry).all())
    for i, d in enumerate(session.query(Entry).all()): #type:Entry
        print(i, total)
        pred = xception_process(d.thumbnail_path)
        break
        d.xception_string = json.dumps(pred)

    # session.commit()



# hdf5_writer.set_path("data/test-features.hdf5", mode="r+")
# hdf5_writer.initialize_dataset("xception_features", shape=(10*10*2048, ), dtype=np.float16)
try:
    init_xception(True)
    compute_feature_over_db()
    # hdf5_writer.on_close()
except Exception as e:
    session.rollback()
    raise e

from keras.applications.xception import Xception, preprocess_input