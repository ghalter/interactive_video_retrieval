import os

import numpy as np
import cv2

from src.hdf5_manager import hdf5_writer
from src.database import session, dump_entry, get_by_hdf_index, Entry
from src.spatial_histogram import calculate_spatial_histogram

n_bins = 10
n_bins_l = 5
n_cols = 4
n_rows = 3


def compute_feature_over_db(func,):
    for d in session.query(Entry).all(): #type:Entry

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
hdf5_writer.initialize_dataset("histograms", shape=(n_rows, n_cols, n_bins, n_bins, n_bins), dtype=np.float16)
compute_feature_over_db(compute_histograms)
hdf5_writer.on_close()

# def compute_extends(entry, frame):
#     frame = cv2.cvtColor(frame.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
#
#     amin, amax = np.amin(frame[:,:,1]), np.amax(frame[:,:,1])
#     bmin, bmax =  np.amin(frame[:,:,2]), np.amax(frame[:,:,2])
#     global t_amax, t_amin, t_bmin, t_bmax
#
#     if amin < t_amin:
#         t_amin = amin
#
#     if bmin < t_bmin:
#         t_bmin = bmin
#
#     if amax > t_amax:
#         t_amax = amax
#
#     if bmax > t_bmax:
#         t_bmax = bmax
#
# t_amin = 128
# t_bmin = 128
# t_amax = -128
# t_bmax = -128
#
# compute_feature_over_db(compute_extends)
# print(t_amin, t_amax, t_bmin, t_bmax)
