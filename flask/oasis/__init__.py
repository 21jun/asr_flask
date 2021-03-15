from flask import Flask
from flask_cors import CORS
from config import config_by_name

app = Flask(__name__)
CORS(app)
app.config.from_object(config_by_name["dev"])

from oasis import route