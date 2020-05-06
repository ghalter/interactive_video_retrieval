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

    frame_lab[:, 1] = np.clip(frame_lab[:, 1], -60, 59)
    frame_lab[:, 2] = np.clip(frame_lab[:, 2], -60, 59)

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
    # X = calculate_spatial_histogram(X, n_rows, n_cols, CONFIG['n_hist_bins'])
    # idxs = np.where(X > 0)
    # print(X.shape, Y.shape)
    # diff = Y - X
    # diff = np.moveaxis(diff, 0, len(diff.shape) - 1)
    #
    # diff = diff[idxs]
    #
    # # Manhattan
    # # res = np.sum(np.abs(res), axis=(0,1,2,3,4))
    #
    # print(diff.shape)
    #
    # res = -np.sum(diff**2 / diff, axis=(0))
    #
    # X = X.reshape()
    # res = np.reshape(res, (CONFIG['n_hist_bins']**3 * n_rows * n_cols, res.shape[-1:][0]))
    # mse = res

    # BHATTACHARYYA
    for x in range(n_rows):
        for y in range(n_cols):
            h = calculate_histogram(X[
                                x * row_wnd: (x + 1) * row_wnd,
                                y * col_wnd: (y + 1) * col_wnd
                                ], n_bins=CONFIG['n_hist_bins']).astype(np.float16)

            r = []
            if np.sum(h) > 0:
                if X.shape[2] == 3:
                    for i in range(Y.shape[0]):
                        a = h
                        b = Y[i, x, y]
                        d = cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
                        r.append(d)
                else:
                    idxs = np.where(h > 0)
                    for i in range(Y.shape[0]):
                        a = h[idxs]
                        b = Y[i, x, y][idxs]
                        d = cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
                        r.append(d)
                c += 1
            else:
                # r = [np.inf] * Y.shape[0]
                r = [0] * Y.shape[0]
            mse.append(r)


    # mse = np.array(mse)
    # mean = np.mean(mse[np.where(mse!=np.inf)])
    #
    # for i in range(n_rows * n_cols):
    #     if np.any(mse[i] == np.inf):
    #         print("replacing")
    #         mse[i, :] = mean
    # mse = np.mean(mse, axis=0)

    mse = np.sum(np.array(mse), axis=0)
    return mse


if __name__ == '__main__':
    import glob
    from src.hdf5_manager import HDF5Manager

    test_images = glob.glob("../data/thumbnails/*.jpg")[:5000]
    ds = HDF5Manager("../data/hist-test.hdf5", mode="r+")
    ds.initialize_dataset("histograms", shape=(3,3,10,10,10), dtype=np.float16)
    for p in test_images:
        img = cv2.imread(p)

        lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
        hists = calculate_spatial_histogram(lab)
        ds.dump(hists, "histograms")

    for i in range(10):
        img = cv2.imread(test_images[i * 30])

        # alpha = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.float32)
        # alpha[-50:, -50:] = 1.0

        lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
        # lab = np.dstack((lab, alpha))

        indices, distances = ds.fit(lab, "histograms", func=histogram_comparator)

        # img = cv2.imread(test_images[50])
        # img[np.where(alpha == 0.0)] = [0, 0, 0]
        print( indices)
        cv2.imshow("input", img)
        for i in range(10):
            print(indices[i], len(test_images))
            img = cv2.imread(test_images[indices[i]])
            cv2.imshow("output" + str(i), img)
        cv2.waitKey()