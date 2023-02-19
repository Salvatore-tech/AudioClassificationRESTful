from flask import Blueprint
from flask import render_template

INDEX_HTML_LOC = 'home.html'

views = Blueprint("views", __name__)


@views.route("/")
@views.route("/home")
def index():
    return render_template("home.html")

@views.route("/whaleDetected")
def show_positive_alert():
    return render_template("whale_alert.html")


