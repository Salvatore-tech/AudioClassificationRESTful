#!/bin/sh
export FLASK_APP=./src/__init__.py
export FLASK_ENV=development
flask run -h 0.0.0.0