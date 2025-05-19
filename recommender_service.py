"""
Recommender System Implementation for Cinemate

This module contains the implementation of recommender algorithms and data loading
functionality used by the Flask API endpoints.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load
from typing import List

# Setup paths
CURRENT_DIR = Path(__file__).parent.resolve()
ML_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Machine-Learning/main-recommender-module')))
MODEL_DIR = ML_DIR / "models"

# Add content_based_filtering directory to sys.path
module_path = os.path.abspath(os.path.join(ML_DIR, 'content_based_filtering'))
sys.path.append(module_path)

# Load collaborative filtering model
cf_model = load(MODEL_DIR / "cf_model.pkl")

# Load ratings data
def load_ratings():
    """Load and optimize ratings dataset."""
    try:
        # Try to load from main-recommender-module path
        ratings_path = ML_DIR / "ml_data/ratings_1m.csv"
        df = pd.read_csv(ratings_path)
    except FileNotFoundError:
        # Fallback to preprocessed data
        ratings_path = ML_DIR / "../preprocessing/ratings.csv"
        df = pd.read_csv(ratings_path)
    
    df['UserID'] = df['UserID'].astype('int32')
    df['MovieID'] = df['MovieID'].astype('int32')
    df['Rating'] = df['Rating'].astype('float16')
    return df

# Load ratings data
try:
    ratings_df = load_ratings()
    print(f"Loaded ratings data with {len(ratings_df)} entries")
except Exception as e:
    print(f"Error loading ratings data: {e}")
    ratings_df = None

# Recommender functions

def get_cf_recommendations(user_id: int) -> List[int]:
    """
    Get recommendations using only the CF model.
    Always returns exactly 10 recommendations.
    
    Args:
        user_id: User ID (int)
        
    Returns:
        List of 10 recommended movie IDs
    """
    if ratings_df is None:
        return []
        
    try:
        # Get all unique movies
        all_movie_ids = ratings_df['MovieID'].unique()
        
        # Filter out movies the user has already rated
        user_rated = ratings_df[ratings_df['UserID'] == user_id]['MovieID'].values
        candidate_movies = np.setdiff1d(all_movie_ids, user_rated)
        
        # If there are too many candidates, sample them to improve performance
        if len(candidate_movies) > 1000:
            np.random.seed(42)
            candidate_movies = np.random.choice(candidate_movies, 1000, replace=False)
            
        # Get predictions for all candidate movies
        predictions = []
        for mid in candidate_movies:
            try:
                pred = cf_model.predict(user_id, mid)
                predictions.append((int(mid), pred.est))
            except Exception as e:
                print(f"Error predicting for movie {mid}: {e}")
                continue
                
        # Sort by predicted rating and get top 10
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [int(mid) for mid, _ in predictions[:10]]
    
    except Exception as e:
        print(f"Error getting CF recommendations: {e}")
        return []

def get_recommendations(user_id):
    return ["Item 1", "Item 2", "Item 3"]  # Example output

def get_test_list(gender, age, profession):

    return ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5"]  # Example output

def get_similar_movies(movie_id):
    return ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5", "Item 6", "Item 7"]  # Example output
