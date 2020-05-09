from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from src.palette_kmeans import ColorClassifier, KMeanPaletteClassifier

##KMeanPaletteClassifier
# import cv2
#
# for k in [5,10,20]:
#     cl = KMeanPaletteClassifier(k=k)
#     example = cv2.imread("example.jpg")
#     cl.fit(example)

####
#
#
# def randrange(n, vmin, vmax):
#     '''
#     Helper function to make an array of random numbers having shape (n, )
#     with each number distributed Uniform(vmin, vmax).
#     '''
#     return (vmax - vmin)*np.random.rand(n) + vmin
#
# fig = plt.figure(figsize=plt.figaspect(1.0))
# ax = fig.add_subplot(111, projection='3d')
#
# n = 100
#
# cl = ColorClassifier()
# names = cl.names
# values = cl.colors
#
#
# # For each set of style and range settings, plot n random points in the box
# # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# # for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
# l, a, b, col, lbl = [],[],[],[],[]
#
# for n, lab in zip(names, values):
#     l.append(lab[0])
#     a.append(lab[1])
#     b.append(lab[2])
#     col.append(cv2.cvtColor(np.array([[lab]], dtype=np.float32), cv2.COLOR_LAB2RGB)[0, 0].tolist() + [1.0] )
#
# ax.scatter(a, b, l, c=col, marker='o')
#
# ax.set_xlabel('A - Channel')
# ax.set_ylabel('B - Channel')
# ax.set_zlabel('L - Channel')
# ax.set_xlim3d(-100, 100)
# ax.set_ylim3d(-100,100)
# ax.set_zlim3d(0,100)
#
# plt.title("Classified Colors in CIE-Lab Colorspace")
#
# plt.show()

import cv2
from src.hilbert import create_hilbert_transform
from src.spatial_histogram import calculate_spatial_histogram
example = cv2.imread("example.jpg")
tf, cols = create_hilbert_transform(16, as_float=True)

frame_lab = cv2.cvtColor(example.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
print(frame_lab.shape)
hists = calculate_spatial_histogram(frame_lab, 3,4, 16)
hists *= example.shape[0] * example.shape[1]

fig, axs = plt.subplots(3, 4)
fig.set_size_inches(20,10)
cols = [(c[0], c[1], c[2], 1.0) for c in cols]

row_wnd = int(np.floor(frame_lab.shape[0] / 3))
col_wnd = int(np.floor(frame_lab.shape[1] / 4))

for x in range(3):
    example[x * row_wnd, :] = [0,0,0]
    for y in range(4):
        example[:, y * col_wnd] = [0, 0, 0]
        # h = hists[x, y]
        # h = h[tf]
        # h = np.reshape(h, newshape=h.shape[0]) + 0.01
        # # h = np.log10(h)
        # # h += 10
        # axs[x, y].bar(range(h.shape[0]), h, color=cols)
        # axs[x, y].set_yscale('log')

cv2.imwrite("example_grid.jpg", example)
# plt.yscale("log")
# plt.savefig("histograms")
# plt.show()
