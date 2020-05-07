import pandas as pd
from src.database import Entry, session
import pickle

dir_path = "data"
embeddings_path = dir_path + '/caption_embeddings.pkl'
file = open(embeddings_path, 'rb')
_caption_embeddings = pickle.load(file)
file.close()

csv_path = 'data/captions.csv'
csv = pd.read_csv(csv_path)
_CAPTIONS = csv['caption'].tolist()
_IMAGE_IDS = csv['thumbnail_id'].tolist()

print(len(_CAPTIONS))
entries = session.query(Entry).all()

res_emb = []
d = dict()
try:
    for e in entries: #type:Entry
        d[e.thumbnail_path.split("/")[2]] = e
    for (c, p, emb) in zip(_CAPTIONS, _IMAGE_IDS, _caption_embeddings):
        print(c, p)
        d[p].caption = c
        res_emb.append((d[p].id, emb))
        print(p)
    session.commit()
except Exception as e:
    session.rollback()
    raise e

res_emb = sorted(res_emb, key=lambda x:x[0])
res_emb = [r[1] for r in res_emb]
with open("embedding.pickle", "wb") as f:
    pickle.dump(res_emb, f)