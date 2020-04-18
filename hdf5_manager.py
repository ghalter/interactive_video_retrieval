"""
Numeric data hosted by the WebApp is stored in a HDF5 File.
The general idea is to have fast access to all the data contained within the
Storage with having to open it.

This file contains the definition of the HDF5_Manager,
taking care of the process of dumping and storing data in the process.

"""

import os
import h5py
import numpy as np
from threading import Lock

# Chunk Size
CHUNK_SIZE = (200,)

DS_COL_HIST = dict(dataset_name="histograms", dataset_shape=(16, 16, 16), dataset_dtype=np.float32)

ALL_ANALYSES = [
    DS_COL_HIST,
]

class HDF5Manager():
    """
    The HDF5 manager takes care on dumping and loading data from a
    hdf5 file.

    Generally, it contains an additional dict **_index** which holds the
    current last index set within the file. Each time something gets dumped into a specific
    file location, the index gets incremented by one.

    .. note:: While it is possible to clear a complete dataset, it is not possible to simply remove a single entry.

    """
    def __init__(self, filepath = None, mode = "r"):
        self.path = None
        self.h5_file = None
        self._index = dict()
        self.mode = mode
        if filepath is not None:
            self.set_path(filepath, mode=mode)

    # region -- IO --
    def set_path(self, path, mode="r"):
        """
        Opens the dataset in a given path, and initializes all dataset if not existing.
        :param path: The path to open
        :return: None
        """
        self.path = path
        init = False
        if self.h5_file is not None:
            self.h5_file.close()
        if not os.path.isfile(self.path):
            self.h5_file = h5py.File(self.path, mode = "w", libver='latest')
            self.h5_file.close()
        self.h5_file = h5py.File(self.path,  mode, libver='latest', swmr=False)

        for name in self.h5_file.keys():
            self._index[name] = self.h5_file[name].attrs['cursor_pos']
            print("Exists:", name.ljust(30), "\tCursor Pos:", self._index[name])
        return init

    def initialize_dataset(self, name, shape, dtype):
        """
        Initialises a single dataset
        :param name: the name of the dataset
        :param shape: the shape of the dataset entry
        :param dtype: the datatype of the dataset
        :return:
        """
        shape = CHUNK_SIZE + shape
        if name not in self.h5_file and self.h5_file.mode == "r+":
            self.h5_file.create_dataset(name=name, shape=shape, dtype=dtype, maxshape=(None,) + shape[1:],
                                        chunks=True)
            self._index[name] = 0
            self.h5_file[name].attrs['cursor_pos'] = 0
            print("INIT: Cursor Pos:", name, self._index[name])
        else:
            self._index[name] = self.h5_file[name].attrs['cursor_pos']
            print("Exists:", name.ljust(30), "\tCursor Pos:",self._index[name])

    def dump(self, d, dataset_name, flush = True):
        """
        Dums a data entry in a given dataset
        :param d: the entry to dump
        :param dataset_name: the dataset where d should be dumped
        :param flush: if the dataset should be flushed
        :return: the index where the data was dumped within a dataset
        """
        if self.h5_file is None:
            raise IOError("HDF5 File not opened yet")
        if dataset_name not in self.h5_file.keys():
            raise IOError("HDF5 File doesnt contain ", dataset_name)

        pos = self._index[dataset_name]
        if pos > 0 and pos % CHUNK_SIZE[0] == 0:
            self.h5_file[dataset_name].resize((pos + CHUNK_SIZE[0],) + self.h5_file[dataset_name].shape[1:])

        self.h5_file[dataset_name][pos] = d
        self._index[dataset_name] += 1
        self.h5_file[dataset_name].attrs['cursor_pos'] = self._index[dataset_name]

        if flush:
            self.h5_file.flush()
        return int(pos)

    def load(self, dataset_name, pos):
        """
        Loads a datapoint from dataset at a given position
        :param dataset_name: the name of the dataset
        :param pos: the index of the dataset (this can also be a list of indices)
        :return: a loaded datapoint
        """
        if self.h5_file is None:
            raise IOError("HDF5 File not opened yet")
        return self.h5_file[dataset_name][pos]

    def fit(self, X, dataset_name, k=100, window=1000):
        """
        Finds a ranked list of closest feature vectors.

        :param X: The feature vector to find the closest for
        :param dataset_name: The dataset to look in
        :param k: The number of closest
        :param window: The search window width
        :return: A ranked list of HDF5 indices
        """
        stop = self._index[dataset_name]

        c = 0
        ranked_indices = None
        ranked_mse = None

        mse_axis = list(range(len(self.h5_file[dataset_name].shape)))[1:]
        mse_axis = tuple(mse_axis)

        while c < stop:
            x0, x1 = c, np.clip(c + window, None, stop)
            Y = self.h5_file[dataset_name][x0:x1]

            indices = np.arange(x0, x1, dtype=np.uint64)
            mse = np.nan_to_num(((Y - X) ** 2)).mean(axis=mse_axis)

            sorting = np.argsort(mse)

            # We keep the k best matches
            mse = mse[sorting][:np.clip(k, None, mse.shape[0])]
            indices = indices[sorting][:np.clip(k, None, mse.shape[0])]

            if ranked_indices is None:
                ranked_indices = indices
                ranked_mse = mse
            else:
                ranked_indices = np.concatenate((ranked_indices, indices))
                ranked_mse = np.concatenate((ranked_mse, mse))
            c += window

        new_ranked = np.argsort(ranked_mse)
        ranked_indices = ranked_indices[new_ranked][:np.clip(k, None, ranked_indices.shape[0])]
        ranked_mse = ranked_mse[new_ranked][:np.clip(k, None, ranked_mse.shape[0])]

        return ranked_indices, ranked_mse


    def on_close(self):
        if self.h5_file is not None:
            self.h5_file.flush()
            self.h5_file.close()
            self.h5_file = None



hdf5_writer = HDF5Manager()
hdf5_file = HDF5Manager()