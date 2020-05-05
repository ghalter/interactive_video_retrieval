"""
This is the web server
"""


import numpy as np
import cv2
import re
from PIL import Image
import base64
import io
import json

import os
from typing import List
from random import sample

from flask import Flask, render_template, request, jsonify, make_response, send_file, url_for

import requests

from src.database import Entry, db, Base
from src.config import CONFIG
from src.hdf5_manager import hdf5_file

from src.spatial_histogram import histogram_comparator
from src.object_recognition import labels
import json

with open("static/all_labels.json", "w") as f:
    print(labels)
    json.dump(labels, f)

n_bins = CONFIG['n_hist_bins']
n_cols = CONFIG['n_hist_cols']
n_rows = CONFIG['n_hist_rows']


app = Flask(__name__)
app.debug = True
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///data/test-database.db"
hdf5_file.set_path("data/test-features.hdf5")
db.init_app(app)

def subquery(entities, sub):
    if sub is None or len(sub) == 0:
        return entities
    sub_ids = [t['id'] for t in sub]
    return [e for e in entities if e.id in sub_ids]

def perform_query(string, k, sub = None):
    # TODO implemented the actual query to perform
    tokens = string.split(",")
    tokens = [t.strip().lower() for t in tokens]
    print(tokens)
    res = []

    imgs = db.session.query(Entry).all() #type:List[Entry]
    imgs = subquery(imgs, sub)

    for i in imgs:
        labels = i.get_query_strings()
        all_found = []
        for string in tokens:
            for k in labels:
                if string in k:
                    all_found.append(string)
                    break

        if all_found == tokens:
            res.append(i.to_json())
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
    sub = request.json['subquery']

    if len(sub) == 0:
        sub = None

    return jsonify(perform_query(q, 10, sub=sub))

@app.route("/similar/", methods=["POST"])
def similar():
    """
    Performs a query and returns a list of results to the front end.
    :return:
    """
    q = request.json['query']
    print(q)
    e = db.session.query(Entry).filter(Entry.id == q['id']).one_or_none()
    if e is None:
        return make_response("not found", 404)
    else:
        img = cv2.imread(e.thumbnail_path)

        cv2.imshow("q", img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        img_lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)

        indices, distances = hdf5_file.fit(img_lab, "histograms", func=histogram_comparator)
        print(distances)
        results = db.session.query(Entry).filter(Entry.histogram_feature_index.in_(indices.tolist())).all()
        # results = subquery(results, sub)

        results = [r.to_json() for r in results]
        return jsonify(results)

    # if len(sub) == 0:
    #     sub = None

    # return jsonify(perform_query(q, 10, sub=sub))


@app.route("/query-image/", methods=["POST"])
def query_image():
    """
    Performs a query and returns a list of results to the front end.
    :return:
    """


    data = request.values['imageBase64']
    data = re.sub('^data:image/.+;base64,', '', data)
    data = base64.b64decode(data)
    sub = json.loads(request.values['subquery'])


    nparr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    alpha = np.zeros(shape = (img.shape[:2]), dtype=np.float32)
    alpha[np.where(np.sum(img, axis=2) > 0)] = 1.0
    img_lab = np.dstack((img_lab, alpha))

    indices, distances = hdf5_file.fit(img_lab, "histograms", func=histogram_comparator)

    results = []
    for idx in indices:
        r = db.session.query(Entry).filter(Entry.histogram_feature_index == int(idx)).one_or_none()
        if r is not None:
            results.append(r)
        print(idx)
    # results = db.session.query(Entry).filter(Entry.histogram_feature_index.in_(indices.tolist())).all()
    results = subquery(results, sub)

    results = [r.to_json() for r in results]
    print("Done", results)
    return jsonify(results)


if __name__ == '__main__':
    app.run()

