"""
This is the web server
"""
import json


from flask import Flask, render_template, request, jsonify, make_response
import requests

from config import CONFIG

app = Flask(__name__)
app.debug = True


def perform_query(string, k):
    # TODO implemented the actual query to perform
    print(string)
    res = []
    for i in range(k):
        res.append(dict(
            location = dict(movie="movie-title", frame_pos = 10),
            thumbnail = "https://homepages.cae.wisc.edu/~ece533/images/airplane.png"
        ))
    return res


@app.route('/')
def index():
    return render_template("index.html")


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

