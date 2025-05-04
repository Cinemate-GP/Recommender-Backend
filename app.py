from flask import Flask, request, jsonify

app = Flask(__name__)

# Dummy recommender functions

def get_recommendations(user_id):
    return ["Item 1", "Item 2", "Item 3"]  # Example output

def get_test_list(gender, age, profession):
    return ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5"]  # Example output

def get_similar_movies(movie_id):
    return ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5", "Item 6", "Item 7"]  # Example output

######################################################################

# Hybrid Recommender System API

@app.route('/')
def index():
    return "Welcome to the Hybrid Recommender System!"

@app.route('/user/recommend', methods=['POST'])
def recommend_top_n():
    data = request.get_json()  # Expecting JSON input like {"user_id": 123}
    user_id = data.get('user_id')
    recommendations = get_recommendations(user_id)
    return jsonify({"recommendations": recommendations})

@app.route('/user/test', methods=['POST'])
def recommend_for_new_user():
    data = request.get_json()
    gender = data.get('gender')
    age = data.get('age')
    profession = data.get('profession')
    test_list = get_test_list(gender, age, profession)
    return jsonify({"test list of movies": test_list})

@app.route('/movie/similar', methods=['POST'])
def similar_movies():
    data = request.get_json()
    movie_id = data.get('movie_id')
    recommendations = get_similar_movies(movie_id)
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True)