import pandas as pd
from src.database import Entry, session


csv_path = 'data/captions.csv'
csv = pd.read_csv(csv_path)
_CAPTIONS = csv['caption'].tolist()
_IMAGE_IDS = csv['thumbnail_id'].tolist()

print(len(_CAPTIONS))
entries = session.query(Entry).all()

d = dict()
try:
    for e in entries: #type:Entry
        d[e.thumbnail_path.split("/")[2]] = e
    for (c, p) in zip(_CAPTIONS, _IMAGE_IDS):
        print(c, p)
        d[p].caption = c
    session.commit()
except Exception as e:
    session.rollback()
    raise e