import os

import numpy as np
import cv2

from src.hdf5_manager import hdf5_writer
from src.database import session, dump_entry, get_by_hdf_index, Entry
from src.spatial_histogram import calculate_spatial_histogram

from src.config import CONFIG

n_bins = CONFIG['n_hist_bins']
n_cols = CONFIG['n_hist_cols']
n_rows = CONFIG['n_hist_rows']


def compute_feature_over_db(func,):
    total = len(session.query(Entry).all())
    for i, d in enumerate(session.query(Entry).all()): #type:Entry
        if i % 10 == 0:
            print(i, total, i / np.round(total * 100, 2), "%")
        frame = cv2.imread(d.thumbnail_path)
        if frame is not None:
            func(d, frame)
    session.commit()


def compute_histograms(entry, frame):

    frame_lab = cv2.cvtColor(frame.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    hist = calculate_spatial_histogram(frame_lab, n_rows, n_cols, n_bins)
    hdf5_index = hdf5_writer.dump(hist, "histograms")
    entry.histogram_feature_index = hdf5_index


hdf5_writer.set_path("data/test-features.hdf5", mode="r+")
hdf5_writer.reset("histograms", shape=(n_rows, n_cols, n_bins, n_bins, n_bins), dtype=np.float16)
compute_feature_over_db(compute_histograms)
hdf5_writer.on_close()