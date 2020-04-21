import os

import numpy as np
import cv2

from src.hdf5_manager import HDF5Manager
from src.database import session, dump_entry, get_by_hdf_index

movies = ["C:/Users/gaude/Documents/VIAN/projects/trailer.mp4"]
resolution = 10


def calculate_histogram(frame_lab, n_bins=16):
    """
    Compute a color histogram of a given CIELab frame.

    :param frame_lab:
    :param n_bins:
    :return:
    """
    frame_lab = np.reshape(cv2.cvtColor(frame_lab, cv2.COLOR_BGR2LAB), (frame_lab.shape[0] * frame_lab.shape[1], 3))
    hist = cv2.calcHist([frame_lab[:, 0], frame_lab[:, 1], frame_lab[:, 2]], [0, 1, 2], None,
                        [n_bins, n_bins, n_bins],
                        [0, 100, -128, 128,
                         -128, 128])
    return hist


## Extraction ##
# We first create a HDF5 File to store all feature vectors
ds = HDF5Manager("data/features.hdf5", mode="r+")


# Initialize a new feature vector dataset, in this case color histograms.
ds.initialize_dataset("histograms", (16,16,16), dtype=np.float16)

# Let's iterate over all movies
for m in movies:

    # Open the movie in opencv
    cap = cv2.VideoCapture(m)
    movie_name = os.path.split(m)[1]

    # For each frame in a given resolution
    for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), resolution):

        # Get the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame_bgr = cap.read()

        # Create a new entry in the database for the given frame
        db_entry = dump_entry(movie_name=movie_name, movie_path=m, frame_pos=i, frame=frame_bgr)

        ## HISTOGRAM ##
        # Convert it to the CIELab color space  to compute a color histogram
        frame_lab = frame_bgr.astype(np.float32) / 255
        hist = calculate_histogram(frame_lab).astype(np.float16)

        # Store the feature vector in the HDF5 File
        h_index = ds.dump(hist, dataset_name="histograms")

        # Keep the location of the feature vector in the HDF5, in the SQL Database.
        db_entry.histogram_feature_index = h_index

        ## END HISTOGRAM ##
        cv2.imshow("output", frame_bgr)
        cv2.waitKey(10)

session.commit()
ds.h5_file.close()



## Retrieval ##
# Open the feature vectors again
ds = HDF5Manager("data/features.hdf5", mode="r")

test_images = []
for m in movies:
    cap = cv2.VideoCapture(m)
    for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), resolution):
        if i % 40 == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame_bgr = cap.read()
            test_images.append(frame_bgr)

# TEST the retrieval
for test_image in test_images:

    # Compute a histogram again
    frame_lab = test_image.astype(np.float32) / 255
    hist = calculate_histogram(frame_lab).astype(np.float16)

    hdf5_indices, mse = ds.fit(hist, "histograms", k=3, window=100)

    for rank, q in enumerate(hdf5_indices):
        r = get_by_hdf_index(q, "histograms")

        cv2.imshow("Input", test_image)
        if r is not None:
            cv2.imshow("Retrieved " + str(rank), cv2.imread(r.thumbnail_path))

    cv2.waitKey()