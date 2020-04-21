import os

import cv2
import pandas as pd
from src.database import session, dump_entry

def ms_to_frames(time, fps):
    """
    Converts Miliseconds to Frames
    :param time: Time MS
    :param fps: FPS of the Film
    :return: returns a FRAME IDX
    """
    return int(round(float(time) / 1000 * fps))

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
        center = (float(r.START_MS) + ((float(r.END_MS) - float(r.START_MS))) / 2) * 1000
        center_f = ms_to_frames(center, curr_fps)
    except Exception as e:
        print(e)
        continue

    cap.set(cv2.CAP_PROP_POS_FRAMES, center_f)
    ret, frame = cap.read()

    if frame is None:
        print("Frame is None", curr_movie)
        continue
    else:
        dump_entry(curr_movie_name,
                   r.MOVIE_PATH,
                   center_f, frame,
                   thumbnail_path=str(curr_movie_name) + "_" + str(scr_in_movie_counter))
        scr_in_movie_counter += 1

session.commit()