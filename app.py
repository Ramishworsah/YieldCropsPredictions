from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
import pandas as pd


# creating flask app
app=Flask(__name__)


#python main

if __name__=='__main__':
    app.run(debug=True)