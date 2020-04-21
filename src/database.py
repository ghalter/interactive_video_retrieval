import os
import uuid
import cv2

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

from src.config import CONFIG
from flask_sqlalchemy import SQLAlchemy

MAX_WIDTH = 730

Base = declarative_base()
db = SQLAlchemy()

def resize_with_aspect(img, width = None, height = None, mode=cv2.INTER_CUBIC):
    if width is not None:
        fy = width / img.shape[1]
    elif height is not None:
        fy = height / img.shape[0]
    else:
        raise ValueError("Either width or height have to be given")

    return cv2.resize(img, None, None, fy, fy, mode)

class Entry(Base):
    __tablename__ = 'entries'

    id = Column(Integer, primary_key=True)
    movie_name = Column(String)
    movie_path = Column(String)
    frame_pos = Column(Integer)
    thumbnail_path = Column(String)

    histogram_feature_index = Column(Integer, default=-1)

engine = create_engine("sqlite:///data/database.db", echo=False)
Base.metadata.create_all(engine)

session = sessionmaker(bind=engine)()

def dump_entry(movie_name, movie_path, frame_pos, frame, thumbnail_path = None) -> Entry:
    if thumbnail_path is None:
        thumbnail_path = os.path.join(CONFIG['thumbnail_folder'], str(uuid.uuid4()) + ".jpg")
    else:
        thumbnail_path = os.path.join(CONFIG['thumbnail_folder'], str(thumbnail_path) + ".jpg")

    if frame.shape[1] > MAX_WIDTH:
        frame = resize_with_aspect(frame, width=MAX_WIDTH)

    cv2.imwrite(thumbnail_path, frame)
    e = Entry(movie_name=movie_name,
              movie_path=movie_path,
              frame_pos=frame_pos,
              thumbnail_path=thumbnail_path)
    session.add(e)
    return e


def get_by_hdf_index(index, dataset) -> Entry:
    """
    Returns an Entry by a given feature vector index

    :param index:
    :param dataset:
    :return: Entry
    """
    if dataset == "histograms":
        return session.query(Entry).filter(Entry.histogram_feature_index == int(index)).one_or_none()

    return None