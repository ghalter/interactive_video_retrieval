import numpy as np
import cv2

from src.config import CONFIG


def calculate_histogram(frame_lab, n_bins=10):
    """
    Compute a color histogram of a given CIELab frame.

    :param frame_lab:
    :param n_bins:
    :return:
    """
    if frame_lab.shape[2] > 3:
        remove_alpha = np.reshape(frame_lab, (frame_lab.shape[0] * frame_lab.shape[1], 4))
        frame_lab = remove_alpha[np.where(remove_alpha[:, 3] > 0)]
        frame_lab = frame_lab[:, :3]

    else:
        frame_lab = np.reshape(frame_lab, (frame_lab.shape[0] * frame_lab.shape[1], 3))

    frame_lab[:, 1] = np.clip(frame_lab[:, 1], -60, 60)
    frame_lab[:, 2] = np.clip(frame_lab[:, 2], -60, 60)

    hist = cv2.calcHist([frame_lab[:, 0],
                         frame_lab[:, 1],
                         frame_lab[:, 2]],
                        [0, 1, 2],
                        None,
                        [n_bins, n_bins, n_bins],
                        [0, 100, -60, 60,
                         -60, 60]).astype(np.float16)
    return hist / frame_lab.shape[0]


def calculate_spatial_histogram(frame_lab, n_rows = 3, n_cols = 3, n_bins = 10):
    row_wnd = int(np.floor(frame_lab.shape[0] / n_rows))
    col_wnd = int(np.floor(frame_lab.shape[1] / n_cols))
    d = np.zeros(shape=(n_rows, n_cols, n_bins, n_bins, n_bins))
    for x in range(n_rows):
        for y in range(n_cols):
            d[x, y] = calculate_histogram(frame_lab[
                                          x * row_wnd : (x + 1) * row_wnd,
                                          y * col_wnd : (y + 1) * col_wnd
                                          ], n_bins=n_bins)
    return d

from scipy.spatial import distance as dist

def histogram_comparator(X, Y):
    n_rows = Y.shape[1]
    n_cols = Y.shape[2]

    row_wnd = int(np.floor(X.shape[0] / n_rows))
    col_wnd = int(np.floor(X.shape[1] / n_cols))

    mse = []
    c = 0
    for x in range(n_rows):
        for y in range(n_cols):
            h = calculate_histogram(X[
                                x * row_wnd: (x + 1) * row_wnd,
                                y * col_wnd: (y + 1) * col_wnd
                                ], n_bins=CONFIG['n_hist_bins'])
            if np.sum(h) > 0:
                r = []
                for i in range(Y.shape[0]):
                    d = cv2.compareHist(Y[i, x, y].astype(np.float32), h.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
                    r.append(d)
                mse.append(r)
                c += 1

    mse = np.array(mse)
    mse = np.log(np.mean(mse, axis=0))
    return mse


if __name__ == '__main__':
    import glob
    from src.hdf5_manager import HDF5Manager

    test_images = glob.glob("../data/thumbnails/*.jpg")[:100]
    ds = HDF5Manager("../data/hist-test.hdf5", mode="r+")
    ds.initialize_dataset("histograms", shape=(3,3,10,10,10), dtype=np.float16)
    for p in test_images:
        img = cv2.imread(p)

        lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
        hists = calculate_spatial_histogram(lab)
        ds.dump(hists, "histograms")

    img = cv2.imread(test_images[50])

    alpha = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.float32)
    alpha[-50:, -50:] = 1.0

    lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    lab = np.dstack((lab, alpha))
    print(lab.shape)
    indices, distances = ds.fit(lab, "histograms", func=histogram_comparator)

    print(indices)
    # img = cv2.imread(test_images[50])
    img[np.where(alpha == 0.0)] = [0, 0, 0]
    cv2.imshow("input", img)
    for i in range(10):
        img = cv2.imread(test_images[indices[i]])
        cv2.imshow("output" + str(i), img)
    cv2.waitKey()