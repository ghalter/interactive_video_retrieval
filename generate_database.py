import os

import numpy as np
import cv2

from src.hdf5_manager import hdf5_writer
from src.database import session, dump_entry, get_by_hdf_index, Entry
from src.spatial_histogram import calculate_spatial_histogram


def compute_feature_over_db(func,):
    for d in session.query(Entry).all(): #type:Entry

        frame = cv2.imread(d.thumbnail_path)
        if frame is not None:
            func(d, frame)
    session.commit()

def compute_histograms(entry, frame):

    frame_lab = cv2.cvtColor(frame.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    hist = calculate_spatial_histogram(frame_lab)
    hdf5_index = hdf5_writer.dump(hist, "histograms")
    entry.histogram_feature_index = hdf5_index


hdf5_writer.set_path("data/test-features.hdf5", mode="r+")
hdf5_writer.initialize_dataset("histograms", shape=(3,3,10,10,10), dtype=np.float16)
compute_feature_over_db(compute_histograms)
hdf5_writer.on_close()