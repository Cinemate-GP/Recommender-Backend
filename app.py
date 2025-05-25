"""
Cinemate Recommender API

This module defines the Flask API endpoints for the Cinemate recommender system.
Implementation logic is contained in the recommender_service module.
"""
from flask import Flask, request, jsonify
from recommender_service import (
    get_cf_recommendations,
    get_recommendations,
    get_test_list,
    get_similar_movies
)
import requests
import json
import sys

app = Flask(__name__)

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

@app.route('/user/cf_recommend/<int:user_id>', methods=['GET'])
def cf_recommend(user_id):
    """Endpoint for CF-based recommendations.
    
    Now accessible via GET request with user_id in the URL path:
    http://localhost:5000/user/cf_recommend/661
    """
    try:
        # Get recommendations
        movie_ids = get_cf_recommendations(user_id)
        
        return jsonify({
            "user_id": user_id,
            "count": len(movie_ids),
            "recommendations": movie_ids
        })
    except ValueError:
        return jsonify({"error": "Invalid user_id format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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

# Test CF recommendations endpoint
def test_cf_recommend(user_id=1):
    url = f"http://localhost:5000/user/cf_recommend/{user_id}"
    
    # Using GET request now instead of POST
    response = requests.get(url)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
    else:
        print(f"Error: {response.text}")

if __name__ == '__main__':
    # Check if called with test argument
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Default user ID
        user_id = 1
        
        # Check if user ID was provided as command-line argument
        if len(sys.argv) > 2:
            try:
                user_id = int(sys.argv[2])
                print(f"Testing with user_id: {user_id}")
            except ValueError:
                print(f"Invalid user_id: {sys.argv[2]}. Using default user_id: 1")
        
        test_cf_recommend(user_id)
    else:
        app.run(debug=True)