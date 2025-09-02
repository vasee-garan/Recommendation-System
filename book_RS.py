import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample dataset (10 books)
data = {
    'title': [
        "The Hunger Games", "Catching Fire", "Mockingjay", 
        "Harry Potter and the Sorcerer's Stone", "Harry Potter and the Chamber of Secrets",
        "The Hobbit", "The Lord of the Rings", "1984", "Animal Farm", "Pride and Prejudice"
    ],
    'authors': [
        "Suzanne Collins", "Suzanne Collins", "Suzanne Collins",
        "J.K. Rowling", "J.K. Rowling",
        "J.R.R. Tolkien", "J.R.R. Tolkien", "George Orwell", "George Orwell", "Jane Austen"
    ],
    'genres': [
        "Dystopian Adventure", "Dystopian Adventure", "Dystopian Adventure",
        "Fantasy", "Fantasy",
        "Fantasy Adventure", "Fantasy Adventure", "Dystopian", "Satire", "Romance"
    ],
    'description': [
        "A girl fights to survive in a deadly competition.", 
        "The rebellion grows as Katniss becomes a symbol of hope.",
        "The final showdown against the Capitol.",
        "A young wizard discovers his magical powers.", 
        "The wizarding world faces new challenges.", 
        "A hobbit goes on an epic journey to reclaim a treasure.",
        "Frodo must destroy a powerful ring to save Middle-Earth.",
        "A totalitarian regime controls every aspect of life.",
        "Farm animals rebel against their human farmer.",
        "A story about love, society, and manners in England."
    ]
}

# Create DataFrame
books = pd.DataFrame(data)

# Combine features
books['combined_features'] = (
    books['title'] + " " + books['authors'] + " " + books['genres'] + " " + books['description']
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['combined_features'])

# Cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Mapping from book title to index
indices = pd.Series(books.index, index=books['title']).drop_duplicates()

# Recommendation function with similarity score
def recommend_books(title, num_recommendations=5):
    if title not in indices:
        return f"❌ Book '{title}' not found!"
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Skip itself
    sim_scores = sim_scores[1:num_recommendations+1]
    book_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    
    recommendations = books[['title','authors']].iloc[book_indices].copy()
    recommendations['cosine_similarity'] = scores
    return recommendations

# Example
print("Recommended books for 'The Hobbit':")
print(recommend_books("The Hobbit", 5))
