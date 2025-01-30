import sys
import time
import threading
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Constants (Easy modification)
MOVIES_FILE = "ml-latest-small/movies.csv"
RATINGS_FILE = "ml-latest-small/ratings.csv"
LIKE_THRESHOLD = 2.5  # Define "Like" vs "Dislike" threshold

# Genre Index Mapping
GENRE_NUM_INDEX = {
    "Action": 0, "Adventure": 1, "Animation": 2, "Children": 3,
    "Comedy": 4, "Crime": 5, "Documentary": 6, "Drama": 7,
    "Fantasy": 8, "Film-Noir": 9, "Horror": 10, "Musical": 11,
    "Mystery": 12, "Romance": 13, "Sci-Fi": 14, "Thriller": 15,
    "War": 16, "Western": 17, "Other (Genre)": 18
}

# Global flag for loading animation
done = False

def load_data():
    """Loads movies and ratings datasets from CSV files."""
    movies = pd.read_csv(MOVIES_FILE)
    ratings = pd.read_csv(RATINGS_FILE)
    return movies, ratings

def compute_user_stats(ratings):
    """Computes user-specific features: average rating given and number of ratings per user."""
    user_stats = ratings.groupby('userId').agg(
        user_avg_rating=('rating', 'mean'),
        user_num_ratings=('rating', 'count')
    ).reset_index()
    return user_stats

def encode_genres(movies):
    """One-hot encodes the genres column and returns updated movies DataFrame."""
    movies['genres'] = movies['genres'].str.split('|')

    encoded_genres_list = []
    
    for genres in movies["genres"]:
        genres_list = [0] * len(GENRE_NUM_INDEX)
        
        for genre in genres:
            if genre in GENRE_NUM_INDEX:
                genres_list[GENRE_NUM_INDEX[genre]] = 1
            else:
                genres_list[-1] = 1  # Mark as "Other (Genre)"

        encoded_genres_list.append(genres_list)

    # Convert the list to a DataFrame with separate genre columns
    genre_df = pd.DataFrame(encoded_genres_list, columns=GENRE_NUM_INDEX.keys())

    # Merge encoded genres with movies DataFrame
    movies = pd.concat([movies, genre_df], axis=1)

    # Drop the original 'genres' column
    movies.drop(columns=['genres'], inplace=True)

    return movies


def compute_movie_stats(ratings):
    """Computes average rating and number of ratings for each movie."""
    movie_stats = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()

    return movie_stats

def extract_year(title):
    """Extracts the release year from the movie title without using regex."""
    parts = title.strip().split(' ')
    if parts and parts[-1].startswith('(') and parts[-1].endswith(')'):
        year_str = parts[-1][1:-1]  # Remove parentheses
        if year_str.isdigit():  # Ensure it's a number
            return int(year_str)
    return 0  # Default if no year is found

def merge_movie_data(movies, movie_stats):
    """Merges computed rating statistics into the movies dataset and extracts movie release year."""
    movies['year'] = movies['title'].apply(extract_year)

    movies = movies.merge(movie_stats, on="movieId", how="left")

    # Fill NaN values (some movies may have no ratings)
    movies[['avg_rating', 'num_ratings']] = movies[['avg_rating', 'num_ratings']].fillna(0)

    return movies

def build_feature_matrix(movies):
    """Extracts the feature matrix X for training."""
    feature_columns = list(GENRE_NUM_INDEX.keys()) + ['avg_rating', 'num_ratings', 'year']
    X = movies[feature_columns].values  # Convert to NumPy array
    return X

def prepare_X_y(ratings, movies):
    """Aligns X (features) and y (ratings) so they have the same number of rows."""

    user_stats = compute_user_stats(ratings)

    ratings = ratings.merge(user_stats, on="userId", how="left")

    merged_data = ratings.merge(movies, on="movieId", how="inner")

    # Extract target variable (y) → User ratings
    y = merged_data['rating'].values  

    # Extract features (X) → Movie features for each rating
    feature_columns = list(GENRE_NUM_INDEX.keys()) + ['avg_rating', 'num_ratings', 'year', 'user_avg_rating', 'user_num_ratings']
    X = merged_data[feature_columns].values  # Now same length as y

    return X, y


def prepare_movie_data():
    """Runs the entire preprocessing pipeline and returns feature matrix X and target variable y."""
    # Step 1: Load data
    movies, ratings = load_data()

    # Step 2: Encode genres
    movies = encode_genres(movies)

    # Step 3: Compute avg_rating and num_ratings
    movie_stats = compute_movie_stats(ratings)

    # Step 4: Merge computed stats with movies
    movies = merge_movie_data(movies, movie_stats)

    # Step 5: Align X and y correctly
    X, y = prepare_X_y(ratings, movies)

    return X, y, movies, ratings

def predict_user_rating(user_id, movie_id, movies, ratings, model):
    """Predicts how a specific user would rate a specific movie."""
    # Get user stats
    user_stats = compute_user_stats(ratings)
    user_data = user_stats[user_stats['userId'] == user_id]
    user_avg_rating = user_data['user_avg_rating'].values[0] if not user_data.empty else ratings['rating'].mean()
    user_num_ratings = user_data['user_num_ratings'].values[0] if not user_data.empty else ratings['userId'].value_counts().median()

    # Get movie features
    if movie_id not in movies['movieId'].values:
        print(f"Movie {movie_id} not found in dataset. Cannot predict.")
        return None

    movie_features = movies.loc[movies['movieId'] == movie_id]
    
    # Extract relevant features
    feature_columns = list(GENRE_NUM_INDEX.keys()) + ['avg_rating', 'num_ratings', 'year']
    
    X_test = movie_features[feature_columns].values
    X_test = np.append(X_test, [[user_avg_rating, user_num_ratings]], axis=1)  # Add user features
    
    # Reshape for prediction
    X_test = X_test.reshape(1, -1)
    
    # Predict rating
    predicted_rating = model.predict(X_test)[0]
    
    return predicted_rating

def loading_animation():
    # Simple loading animation
    counter = 0
    while not done:
        sys.stdout.write('\rLoading' + '.' * (counter % 4))  
        sys.stdout.flush()
        time.sleep(0.5)
        counter += 1

    sys.stdout.write('\r' + ' ' * 15)  # preventing visual bug where output has characters from loading
    sys.stdout.write('\rDone Processing\n\n')
    sys.stdout.flush()

def process_and_output(user_id, movie_id, model_choice, X, y, movies, ratings):
    """Processes the input user ID, movie ID, and model choice to train and evaluate a movie rating prediction model."""
    global done

    title = movies.loc[movies['movieId'] == movie_id, 'title'].values[0]
    print(f"\nRunning prediction \nUser: {user_id}  \nMovie: {title} id: ({movie_id}) \nModel: {model_choice}\n")

    # Start the loading animation in a separate thread
    done = False
    loading_thread = threading.Thread(target=loading_animation)
    loading_thread.start()

    if model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)  

    y_pred = model.predict(X)

    # Stop the loading animation
    done = True
    loading_thread.join()

    # Compute Mean Squared Error (MSE) & R² Score
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred) # R² (R-Squared) is a metric that tells us how well our model explains the variability of the target variable (y).

    print('-' * 45, 'Results', '-' * 46)

    predicted_rating = predict_user_rating(user_id, movie_id, movies, ratings, model)

    print(f"User id: {user_id} \nMovie Title: {title}\nMovie id: {movie_id}\n\n")

    actual_likes = (y >= LIKE_THRESHOLD).astype(int) # Convert actual ratings to binary (1 = Like, 0 = Dislike)

    predicted_likes = (y_pred >= LIKE_THRESHOLD).astype(int) # Convert predicted ratings to binary (1 = Like, 0 = Dislike)

    like_accuracy = np.mean(actual_likes == predicted_likes) * 100 # Calculate Accuracy for Like Predictions

    print(f"Model: {model_choice}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}\n")  # Closer to 1 means better model performance
    print("Predicted Rating: {predicted_rating:.2f}")
    print(f"Prediction Accuracy (Ratings ≥ {LIKE_THRESHOLD}): {like_accuracy:.2f}%")
    print('-' * 100)

def get_valid_user_id(ratings):
    """Prompt user for a valid User ID until a correct one is entered."""
    while True:
        try:
            print("\nUser Ids can be found in ratings.csv. Valid Ids: 1, 2, 217, 509, 610")
            user_id = int(input("Enter User ID: ").strip())
            if user_id in ratings['userId'].values:
                return user_id
            else:
                print("❌ Invalid User ID. Please enter a valid one from the dataset. These can be found in ratings.csv")
        except ValueError:
            print("❌ Invalid input. Please enter a numeric User ID.")

def get_valid_movie_id(movies):
    """Prompt user for a valid Movie ID until a correct one is entered."""
    while True:
        try:
            print("\nMovie IDs be found in movies.csv. Valid Ids: 1, 2, 12, 48, 161918")
            movie_id = int(input("Enter Movie ID: ").strip())
            if movie_id in movies['movieId'].values:
                return movie_id
            else:
                print("❌ Invalid Movie ID. Please enter a valid one from the dataset. These can be found in ratings.csv")
        except ValueError:
            print("❌ Invalid input. Please enter a numeric Movie ID.")

def repl():
    """Main method which allows the user to interact and input with program and provide inputs/outputs."""
    X, y, movies, ratings = prepare_movie_data()

    print('-' * 100)
    while True:
        print("\nMovie Rating Prediction REPL")
        print("1: Use Linear Regression (More accurate but slower)")
        print("2: Use Random Forest Regressor (Less accurate but much faster)")
        print("Q: Quit")

        model_choice = input("Select a model (1/2) or 'Q' to quit: ").strip().lower()

        if model_choice == '1':
            model_choice = "Linear Regression"
        elif model_choice == '2':
            model_choice = "Random Forest Regressor"

        if model_choice == 'q':
            print("\nExiting program. Goodbye!")
            break
        elif model_choice not in ("Linear Regression", "Random Forest Regressor"):
            print("\nInvalid choice. Please enter 1, 2, or Q.")
            continue
        
        user_id = get_valid_user_id(ratings)
        movie_id = get_valid_movie_id(movies)
        
        process_and_output(user_id, movie_id, model_choice, X, y, movies, ratings)


repl()