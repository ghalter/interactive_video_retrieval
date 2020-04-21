"""
This is the web server
"""
import os
from typing import List
from random import sample

from flask import Flask, render_template, request, jsonify, make_response, send_file, url_for

import requests
from src.database import Entry, db

from src.config import CONFIG


app = Flask(__name__)
app.debug = True
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///data/database.db"

db.init_app(app)



def perform_query(string, k):
    # TODO implemented the actual query to perform
    print(string)
    res = []
    imgs = db.session.query(Entry).all() #type:List[Entry]

    imgs = sample(imgs, k=50)
    for i in imgs:
        res.append(dict(
            location = dict(movie=i.movie_name, frame_pos = i.frame_pos),
            thumbnail = url_for("get_screenshot", file_path = i.thumbnail_path.replace("/", "|"))
        ))
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


if __name__ == '__main__':
    app.run()

