import json

try:
    with open("data/config.json", "r") as f:
        CONFIG = json.load(f)
except:
    with open("../data/config.json", "r") as f:
        CONFIG = json.load(f)