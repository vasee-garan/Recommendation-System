import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------
# Step 1: Create a Larger Sample Dataset
# -----------------------------------
data = {
    'Inception':        [5, 4, 0, 0, 3, 4, 5],
    'Interstellar':     [4, 0, 0, 2, 4, 5, 3],
    'The Dark Knight':  [5, 5, 0, 4, 0, 3, 5],
    'Tenet':            [0, 4, 5, 3, 0, 2, 4],
    'Memento':          [3, 0, 4, 0, 5, 3, 0],
    'Dunkirk':          [0, 3, 4, 0, 4, 5, 3],
    'Prestige':         [4, 0, 5, 3, 0, 4, 5],
    'Batman Begins':    [5, 4, 0, 4, 3, 5, 4]
}

# Users
ratings_df = pd.DataFrame(data, index=['User1','User2','User3','User4','User5','User6','User7'])
print("🎬 Movie Ratings & Viewing History Dataset:\n")
print(ratings_df)

# -----------------------------------
# Step 2: Compute Cosine Similarity Between Users
# -----------------------------------
similarity_matrix = cosine_similarity(ratings_df)
similarity_df = pd.DataFrame(similarity_matrix, 
                             index=ratings_df.index, 
                             columns=ratings_df.index)

print("\n📊 Cosine Similarity Matrix Between Users:\n")
print(similarity_df.round(2))

# -----------------------------------
# Step 3: Recommend Movies Function
# -----------------------------------
def recommend_movies(target_user, ratings_df, similarity_df, top_n=5):
    # Get similarity scores for target user
    similar_users = similarity_df[target_user].drop(target_user)  # exclude self
    similar_users = similar_users.sort_values(ascending=False)

    print(f"\n🔎 Similarity Scores for {target_user}:")
    for user, score in similar_users.items():
        print(f"   {user}: {score:.2f}")

    # Take the most similar user
    most_similar_user = similar_users.index[0]
    similarity_score = similar_users.iloc[0]

    print(f"\n✅ Most similar user to {target_user}: {most_similar_user} (Cosine Similarity = {similarity_score:.2f})")

    # Find movies target user has not watched
    user_ratings = ratings_df.loc[target_user]
    unseen_movies = user_ratings[user_ratings == 0].index

    # Recommend top movies that similar user rated highly
    similar_user_ratings = ratings_df.loc[most_similar_user, unseen_movies]

    # Remove zero-rated movies from similar user
    similar_user_ratings = similar_user_ratings[similar_user_ratings > 0]

    recommended_movies = similar_user_ratings.sort_values(ascending=False).head(top_n)

    return recommended_movies

# Example: Recommend movies for User1
recommendations = recommend_movies("User1", ratings_df, similarity_df, top_n=5)

print("\n🎯 Recommended Movies for User1:\n")
print(recommendations.to_dict())
