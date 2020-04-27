import numpy as np
import cv2


def calculate_histogram(frame_lab, n_bins=10):
    """
    Compute a color histogram of a given CIELab frame.

    :param frame_lab:
    :param n_bins:
    :return:
    """
    frame_lab = np.reshape(cv2.cvtColor(frame_lab, cv2.COLOR_BGR2LAB), (frame_lab.shape[0] * frame_lab.shape[1], 3))
    hist = cv2.calcHist([frame_lab[:, 0],
                         frame_lab[:, 1],
                         frame_lab[:, 2]],
                        [0, 1, 2],
                        None,
                        [n_bins, n_bins, n_bins],
                        [0, 100, -100, 100,
                         -100, 100]).astype(np.float16)
    return hist / frame_lab.shape[0]


def calculate_spatial_histogram(frame_lab, n_rows = 3, n_cols = 3, n_bins = 10):
    row_wnd = int(np.floor(frame_lab.shape[0] / n_rows))
    col_wnd = int(np.floor(frame_lab.shape[1] / n_cols))
    d = np.zeros(shape=(n_rows,n_cols, n_bins, n_bins, n_bins))
    for x in range(n_rows):
        for y in range(n_cols):
            d[x, y] = calculate_histogram(frame_lab[
                                          x * row_wnd : (x + 1) * row_wnd,
                                          y * col_wnd : (y + 1) * col_wnd
                                          ], n_bins=n_bins)
    return d



def histogram_comparator(X, Y):
    return np.nan_to_num(((Y - X) ** 2)).mean(axis=(1,2,3,4,5))


if __name__ == '__main__':
    import glob
    from src.hdf5_manager import HDF5Manager

    test_images = glob.glob("../data/thumbnails/*.jpg")[:100]
    ds = HDF5Manager("../data/hist-test.hdf5", mode="r+")
    ds.initialize_dataset("histograms", shape=(3,3,10,10,10), dtype=np.float16)
    for p in test_images:
        img = cv2.imread(p)
        cv2.imshow("t", img)
        cv2.waitKey(50)
        lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
        hists = calculate_spatial_histogram(lab)
        ds.dump(hists, "histograms")

    img = cv2.imread(test_images[50])
    lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    hists = calculate_spatial_histogram(lab)
    indices, distances = ds.fit(hists, "histograms")

    print(indices)
    img = cv2.imread(test_images[50])
    cv2.imshow("input", img)
    img = cv2.imread(test_images[indices[0]])
    cv2.imshow("output", img)
    cv2.waitKey()