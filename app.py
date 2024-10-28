from flask import Flask, request, render_template
import requests
import re
import string
import time
from transformers import pipeline

app = Flask(__name__)

# YouTube Data API key
API_KEY = 'GETYOUROWNAPIKEY'

def get_video_id(url):
    # Extract video ID from the YouTube URL
    if 'v=' in url:
        return url.split('v=')[1].split('&')[0]
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[1].split('?')[0]
    return None

def get_comments(video_id, api_key):
    comments = []
    url = f'https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key={api_key}&maxResults=100'

    while True:
        response = requests.get(url)
        data = response.json()

        if 'items' not in data:
            break

        # Append the comments from the current batch
        for item in data['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        # Check if there are more comments to fetch
        if 'nextPageToken' in data:
            next_page_token = data['nextPageToken']
            url = f'https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key={api_key}&maxResults=100&pageToken={next_page_token}'
            
            # Add a small delay to avoid getting rate-limited
            time.sleep(1)
        else:
            break

    return comments

def preprocess_text(text):
    # Custom preprocessing function to remove punctuation
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation using regex
    # Limit the length to 200 characters to prevent oversized input for the model
    if len(text) > 200:
        text = text[:200]
    return text

# Load the multilingual sentiment analysis pipeline using BERT
sentiment_analyzer = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

@app.route('/', methods=['GET', 'POST'])
def index():
    positive_comments = []
    negative_comments = []
    neutral_comments = []
    unclassified_comments = []
    video_url = ""
    total_comments = 0

    if request.method == 'POST':
        video_url = request.form.get('video_url')
        video_id = get_video_id(video_url)

        if video_id:
            comments = get_comments(video_id, API_KEY)
            if comments:
                total_comments = len(comments)

                # Preprocess comments
                processed_comments = [preprocess_text(comment) for comment in comments]

                # Predict sentiment using Multilingual BERT
                predictions = sentiment_analyzer(processed_comments)

                # Categorize comments
                for comment, prediction in zip(comments, predictions):
                    sentiment = prediction['label'].lower()
                    if '5 stars' in sentiment or '4 stars' in sentiment:
                        positive_comments.append(comment)
                    elif '2 stars' in sentiment or '1 star' in sentiment:
                        negative_comments.append(comment)
                    elif '3 stars' in sentiment:
                        neutral_comments.append(comment)
                    else:
                        unclassified_comments.append(comment)

    return render_template(
        'index.html', 
        video_url=video_url,
        positive=positive_comments, 
        negative=negative_comments, 
        neutral=neutral_comments, 
        unclassified=unclassified_comments,
        total_comments=total_comments
    )

if __name__ == '__main__':
    app.run(debug=True)
