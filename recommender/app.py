from flask import Flask, request, render_template, redirect, flash, jsonify, url_for
import pandas as pd
import numpy as np
from surprise import dump
from db import DataBase
from recommender import Recommender

ITEMS = 50
app = Flask(__name__)
users = np.load('data/users.npy').item()

db = DataBase(users, pd.read_csv('data/attractions.csv', index_col='index'), pd.read_csv('data/ratings_reshaped.csv'), dump.load('data/funksvd')[1])
recommender = Recommender(db)
attractions = db.get_attractions()
topics = [
        {'id': '0', 'name': 'Landmarks', 'photo': '/static/landmarks.jpg'},
        {'id': '1', 'name': 'Art', 'photo': '/static/art.jpeg'},
        {'id': '2', 'name': 'Tours', 'photo': '/static/tours.jpg'},
        {'id': '3', 'name': 'Food', 'photo': '/static/food.jpg'},
        {'id': '4', 'name': 'Nature', 'photo': '/static/nature.jpg'},
        {'id': '5', 'name': 'Performing\narts', 'photo': '/static/performing.jpg'}
    ]

def add_rating_inline():
    username = request.form.get("username")
    data = request.form
    for element in data.items():
        if element[0] != 'username':
            item_id = int(element[0])
            rating = int(element[1])
            db.add_user(username, item_id, rating)

    db.retrain_svd()


@app.route('/', methods=['GET', 'POST'])
def index():
   

    if request.method == 'POST':
        topics_to_recommend = list(map(int, request.form.getlist('topic')))
        popularity = False if request.form.get('popularity') else True
        
        recommender.lda(db, popularity, topics_to_recommend)
        
        return redirect(url_for('get_attractions'))

    return render_template("index.html", topics=topics, ITEMS=ITEMS)

@app.route('/attractions', methods=['GET', 'POST'])
def get_attractions():
    if request.method == 'POST':
        add_rating_inline()
        return redirect(url_for("existing"))
    return render_template("attractions.html", attractions=recommender.recommendations[:10], ITEMS=ITEMS)

@app.route('/load_more', methods=["POST"])
def load_more():
    index = int(request.get_json()['index'])
    if index < ITEMS and index + 10 < ITEMS:
        return jsonify(recommender.recommendations[index:index+10])
    elif index < ITEMS:
        return jsonify(recommender.recommendations[index:ITEMS])
    else:
        return jsonify([])


@app.route('/existing', methods=['GET', 'POST'])
def existing():
    if request.method == 'POST':
        add_rating_inline()
        
        return redirect(url_for("existing"))


    return render_template('existing.html', attractions=recommender.recommendations[:10], ITEMS=ITEMS)

@app.route('/add', methods=['GET', 'POST'])
def add_rating():
    if request.method == 'POST':
        username = request.form.get("username")
        item_id = int(request.form.get("attraction"))
        rating = int(request.form.get("rating"))
        _ = db.add_user(username, item_id, rating, retrain=True)
        # _, rated_attractions = db.get_rated(uid)
        
        return redirect(url_for("index"))

    return render_template('new.html', attractions=attractions, ITEMS=ITEMS)

@app.route('/get_attractions', methods=["POST"])
def get_attractions_():
    return jsonify(attractions)

@app.route('/get_recommendations', methods=["POST"])
def get_recommendations():
    data = request.get_json()
    try:
        username = data['username']
        
        uid = db.get_user_id(username)
        _, items = db.get_rated(uid)
        if data['recommendations']:
            recommender.funksvd(db, uid)
            print(uid)
            return jsonify(recommender.recommendations[:10])
        return jsonify(items)
    except: return jsonify([])

@app.route('/nan', methods=['GET', 'POST'])
def nan():
    return render_template('existing.html', attractions=recommender.recommendations)
if __name__ == '__main__':
    app.run()