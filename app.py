"""
This is the web server
"""
import os
from typing import List
from random import sample

from flask import Flask, render_template, request, jsonify, make_response, send_file, url_for

import requests

from src.database import Entry, db, Base
from src.config import CONFIG
from src.hdf5_manager import hdf5_file

from src.spatial_histogram import histogram_comparator


n_bins = CONFIG['n_hist_bins']
n_cols = CONFIG['n_hist_cols']
n_rows = CONFIG['n_hist_rows']


app = Flask(__name__)
app.debug = True
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///data/test-database.db"
hdf5_file.set_path("data/test-features.hdf5")
# db.init_app(app)

Base.metadata.create_all()
def perform_query(string, k):
    # TODO implemented the actual query to perform
    print(string)
    res = []
    imgs = db.session.query(Entry).all() #type:List[Entry]

    imgs = sample(imgs, k=50)
    for r in imgs:
        res.append(r.to_json())
    return res


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/screenshot/<string:file_path>')
def get_screenshot(file_path):
    file_path = file_path.replace("|", "/")
    path = os.path.abspath(file_path)
    print(path)
    return send_file(path, mimetype='image/gif')


@app.route('/submit/<string:video>/<int:frame>')
@app.route('/submit/<string:video>/<int:frame>/')
def submit(video, frame):
    """
    Submits a given result to the VBS Server
    :param video: The name of the movie
    :param frame: The frame position of the result
    :return:
    """
    url = CONFIG['server'] \
          + "/vsb/submit?" \
          + "team=" + CONFIG['team'] \
          + "&member=" + CONFIG['member'] \
          + "&video=" + video \
          + "&frame=" + str(frame)
    print("Submit to:", url)

    requests.get(url)
    return make_response("Submitted", 200)


@app.route("/query/", methods=["POST"])
def query():
    """
    Performs a query and returns a list of results to the front end.
    :return:
    """
    q = request.json['query']
    return jsonify(perform_query(q, 10))

import numpy as np
import cv2
import re
from PIL import Image
import base64
import io

@app.route("/query-image/", methods=["POST"])
def query_image():
    """
    Performs a query and returns a list of results to the front end.
    :return:
    """

    data = request.values['imageBase64']
    data = re.sub('^data:image/.+;base64,', '', data)

    data = base64.b64decode(data)

    nparr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    alpha = np.zeros(shape = (img.shape[:2]), dtype=np.float32)
    alpha[np.where(np.sum(img, axis=2) > 0)] = 1.0
    img_lab = np.dstack((img_lab, alpha))

    indices, distances = hdf5_file.fit(img_lab, "histograms", func=histogram_comparator)

    cv2.imshow("tt", img)
    results = db.session.query(Entry).filter(Entry.histogram_feature_index.in_(indices.tolist())).all()
    results = [r.to_json() for r in results]
    return jsonify(results)


if __name__ == '__main__':
    app.run()

