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
from sentiment_service import predict_sentiment

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

@app.route('/review/sentiment', methods=['POST'])
def analyze_sentiment():
    """
    Analyze sentiment of a movie review.
    
    Expected JSON input: {"review": "This movie was amazing!"}
    
    Returns:
        JSON response with sentiment analysis results:
        {
            "sentiment": "positive" or "negative",
            "confidence": 0.85,
            "probabilities": {
                "negative": 0.15,
                "positive": 0.85
            }
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        review_text = data.get('review')
        
        # Get sentiment prediction from service (includes validation)
        result = predict_sentiment(review_text)
          # Check if there was an error in prediction
        if "error" in result:
            return jsonify(result), 400 if "Missing" in result["error"] or "empty" in result["error"] or "invalid" in result["error"] else 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)