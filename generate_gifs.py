import imageio


import os

import cv2
import pandas as pd
from src.database import session, dump_entry, Entry

def ms_to_frames(time, fps):
    """
    Converts Miliseconds to Frames
    :param time: Time MS
    :param fps: FPS of the Film
    :return: returns a FRAME IDX
    """
    return int(round(float(time) / 1000 * fps))

def resize_with_aspect(img, width = None, height = None, mode=cv2.INTER_CUBIC):
    if width is not None:
        fy = width / img.shape[1]
    elif height is not None:
        fy = height / img.shape[0]
    else:
        raise ValueError("Either width or height have to be given")

    return cv2.resize(img, None, None, fy, fy, mode)

MOVIES_ROOT = "E:/Programming/Datasets/V3C/videos/videos"
EXPORT_DIR  = "E:/Programming/Datasets/V3C/shots/"

df = pd.read_csv("data/scenes.csv")

curr_movie = None
curr_movie_name = None
cap = None
curr_fps = None

scr_in_movie_counter = 0
c=0

for r in df.itertuples():
    n = os.path.join(MOVIES_ROOT, r.MOVIE_PATH)

    if c % 100==0:
        print(c, df.shape[0])
    c+=1

    if n != curr_movie:
        if not os.path.isfile(n):
            print("Skipping", n)
            continue

        curr_movie = n
        curr_movie_name = os.path.split(r.MOVIE_PATH)[1].split(".")[0]
        scr_in_movie_counter = 0

        cap = cv2.VideoCapture(n)
        curr_fps = cap.get(cv2.CAP_PROP_FPS)

    try:
        print(r.START_MS, r.END_MS)
        start = ms_to_frames(float(r.START_MS) * 1000, curr_fps)
        end = ms_to_frames(float(r.END_MS) * 1000, curr_fps)

        center = (float(r.START_MS) + ((float(r.END_MS) - float(r.START_MS))) / 2) * 1000
        center_f = ms_to_frames(center, curr_fps)
    except Exception as e:
        print(e)
        continue

    step = int(end - start / 30)
    print(start, end)
    images = []
    for i in range(start, end, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if frame is None:
            print("Frame is None", curr_movie)
            continue
        else:
            frame = resize_with_aspect(frame, 300)
            # cv2.imshow("out", frame)
            # cv2.waitKey(30)
            images.append(frame[:,:,::-1])
    e = session.query(Entry).filter(Entry.movie_name == curr_movie_name).filter(Entry.frame_pos == center_f).one_or_none()
    print(e)
    with imageio.get_writer("data/gifs/" + str(e.id) + ".gif", mode='I') as writer:
        for f in images:
            writer.append_data(f)


# session.commit()


# for filename in filenames:
#     images.append(imageio.imread(filename))
# imageio.mimsave('/path/to/movie.gif', images)