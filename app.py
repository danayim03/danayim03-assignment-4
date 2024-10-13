import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for MacOS

from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import base64
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load the dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Convert the documents into a term-document matrix
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Perform SVD (LSA) to reduce dimensionality
svd = TruncatedSVD(n_components=100)
X_reduced = svd.fit_transform(X)

# Function to transform the query
def query_transform(query, vectorizer, svd):
    query_tfidf = vectorizer.transform([query])  # Transform query using TF-IDF
    query_reduced = svd.transform(query_tfidf)   # Reduce dimensionality using SVD
    return query_reduced

# Function to get the top 5 most similar documents
def get_top_documents(query_reduced, X_reduced, top_n=5):
    similarities = cosine_similarity(query_reduced, X_reduced)[0]
    top_docs_idx = np.argsort(similarities)[-top_n:][::-1]
    return [(idx, similarities[idx]) for idx in top_docs_idx]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    query_reduced = query_transform(query, vectorizer, svd)
    top_docs = get_top_documents(query_reduced, X_reduced)
    
    # Extract document indices, cosine similarities, and preview content
    doc_indices = [doc[0] for doc in top_docs]
    doc_similarities = [doc[1] for doc in top_docs]
    doc_previews = [documents[idx][:500] for idx in doc_indices]  # Get the first 500 characters of each document
    
    # Plot the results
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(doc_indices)), doc_similarities, tick_label=doc_indices)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Document Index')
    ax.invert_yaxis()  # Reverse the y-axis
    
    # Convert plot to image and embed in HTML
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    # Pass the document previews and query along with the other data
    return render_template('results.html', plot_url=plot_url, top_docs=top_docs, doc_previews=doc_previews, query=query)


if __name__ == '__main__':
    app.run(debug=True)
