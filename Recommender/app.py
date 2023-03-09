from flask import Flask, request, render_template, send_from_directory
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the article data
articles = pd.read_csv('mdh-1.csv')

# Convert the article content into a matrix of TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
article_vectors = vectorizer.fit_transform(articles['Article'])


# Create a Flask app
app = Flask(__name__)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root.path, 'static'),'favicon.ico', mimetype='image/vnd.microsoft.icon')


# Define a route to recommend articles
@app.route('/', methods=['GET','POST'])
def recommend_articles():
    # Prompt the user to enter some words
    # Get the input words from the request body
    if request.method == 'POST':
        # Get the input words from the form
        input_words = request.form['input_words']
        # Convert the user input into a vector of TF-IDF features
        input_vector = vectorizer.transform([input_words])

        # Compute the cosine similarity between the user input vector and the article vectors
        similarities = cosine_similarity(input_vector, article_vectors)

        # Get the indices of the top 7 most similar articles
        most_similar_indices = similarities.argsort()[0][-7:][::-1]
        '''
        # Get the title and content of the most similar article
        recommended_titles = articles.loc[most_similar_indices, 'Article']
        recommended_content = articles.loc[most_similar_indices, 'Condition Identified']
        OR
        '''
        # Get the titles and content of the most similar articles
        recommended_articles = []
        for i in most_similar_indices:
            recommended_articles.append({
                'title': articles.loc[i, 'Article'],
                'content': articles.loc[i, 'Condition Identified']
            })

        '''
        # Display the recommended articles to the user
        for title, content in zip(recommended_titles, recommended_content):
            print("Recommended article: {} \n{}".format(title, content))


        # Return the recommended articles as JSON
        return jsonify(recommended_articles)
        '''
        # Render the recommended articles template with the recommended articles
        return render_template('Rec_aRt.html', recommended_articles=recommended_articles)
    
    else:
        # Render the input form template
        return render_template('index.html')


if __name__ == '__main__':
    app.run()