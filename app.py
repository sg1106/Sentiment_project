

import pandas as pd
import numpy as np
from textblob import TextBlob
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, request, render_template, jsonify
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK data: {e}")

app = Flask(__name__)

# Store results in memory for the session
results = []

# Preprocess text: clean and normalize, return tokens and clean text
def preprocess_text(text):
    try:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        return tokens, ' '.join(tokens)
    except Exception as e:
        logger.error(f"Error in preprocess_text: {e}")
        return [], text

# Perform sentiment analysis using TextBlob
def analyze_sentiment(text):
    try:
        tokens, clean_text = preprocess_text(text)
        blob = TextBlob(clean_text)
        polarity = blob.sentiment.polarity
        sentiment = 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'
        return {
            'original_text': text,
            'clean_text': clean_text,
            'tokens': tokens,
            'sentiment': sentiment,
            'polarity': polarity
        }
    except Exception as e:
        logger.error(f"Error in analyze_sentiment: {e}")
        return None

# Calculate detailed analysis
def calculate_analysis(df):
    try:
        positive_polarity = df[df['sentiment'] == 'Positive']['polarity'].mean() if not df[df['sentiment'] == 'Positive'].empty else 0
        negative_polarity = df[df['sentiment'] == 'Negative']['polarity'].mean() if not df[df['sentiment'] == 'Negative'].empty else 0
        combined_polarity = df['polarity'].mean() if not df.empty else 0
        
        # Extract positive and negative words
        positive_words = set()
        negative_words = set()
        for _, row in df.iterrows():
            if row['sentiment'] == 'Positive':
                positive_words.update(row['tokens'])
            elif row['sentiment'] == 'Negative':
                negative_words.update(row['tokens'])
        
        return {
            'positive_polarity': round(positive_polarity, 3),
            'negative_polarity': round(negative_polarity, 3),
            'combined_polarity': round(combined_polarity, 3),
            'positive_words': list(positive_words),
            'negative_words': list(negative_words)
        }
    except Exception as e:
        logger.error(f"Error calculating analysis: {e}")
        return {
            'positive_polarity': 0,
            'negative_polarity': 0,
            'combined_polarity': 0,
            'positive_words': [],
            'negative_words': []
        }

# Generate plots and convert to base64 for HTML rendering
def generate_plots(df):
    plots = {}
    try:
        if df.empty or 'sentiment' not in df or 'clean_text' not in df:
            logger.warning("Invalid or empty DataFrame, cannot generate plots")
            return plots

        # Sentiment distribution
        try:
            sentiment_counts = df['sentiment'].value_counts()
            if not sentiment_counts.empty:
                plt.figure(figsize=(6, 4))
                sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
                plt.title('Sentiment Distribution')
                plt.xlabel('Sentiment')
                plt.ylabel('Count')
                plt.xticks(rotation=0)
                img = io.BytesIO()
                plt.savefig(img, format='png', bbox_inches='tight')
                img.seek(0)
                plots['sentiment_distribution'] = base64.b64encode(img.getvalue()).decode()
                plt.close()
                logger.info("Sentiment distribution plot generated")
            else:
                logger.warning("No sentiment data for distribution plot")
        except Exception as e:
            logger.error(f"Error generating sentiment distribution: {e}\n{traceback.format_exc()}")

        # Positive word cloud
        try:
            positive_texts = ' '.join(df[df['sentiment'] == 'Positive']['clean_text'].dropna())
            if positive_texts.strip() and len(positive_texts.split()) > 1:
                wordcloud_pos = WordCloud(width=600, height=300, background_color='white').generate(positive_texts)
                plt.figure(figsize=(8, 4))
                plt.imshow(wordcloud_pos, interpolation='bilinear')
                plt.axis('off')
                plt.title('Word Cloud - Positive Sentiments')
                img = io.BytesIO()
                plt.savefig(img, format='png', bbox_inches='tight')
                img.seek(0)
                plots['positive_wordcloud'] = base64.b64encode(img.getvalue()).decode()
                plt.close()
                logger.info("Positive word cloud generated")
            else:
                logger.warning("Insufficient positive text for word cloud")
        except Exception as e:
            logger.error(f"Error generating positive word cloud: {e}\n{traceback.format_exc()}")

        # Negative word cloud
        try:
            negative_texts = ' '.join(df[df['sentiment'] == 'Negative']['clean_text'].dropna())
            if negative_texts.strip() and len(negative_texts.split()) > 1:
                wordcloud_neg = WordCloud(width=600, height=300, background_color='white').generate(negative_texts)
                plt.figure(figsize=(8, 4))
                plt.imshow(wordcloud_neg, interpolation='bilinear')
                plt.axis('off')
                plt.title('Word Cloud - Negative Sentiments')
                img = io.BytesIO()
                plt.savefig(img, format='png', bbox_inches='tight')
                img.seek(0)
                plots['negative_wordcloud'] = base64.b64encode(img.getvalue()).decode()
                plt.close()
                logger.info("Negative word cloud generated")
            else:
                logger.warning("Insufficient negative text for word cloud")
        except Exception as e:
            logger.error(f"Error generating negative word cloud: {e}\n{traceback.format_exc()}")

        return plots
    except Exception as e:
        logger.error(f"Error in generate_plots: {e}\n{traceback.format_exc()}")
        return {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    global results
    text = request.form.get('text')
    if text and text.strip():
        result = analyze_sentiment(text)
        if result:
            results.append(result)
            df = pd.DataFrame(results)
            analysis = calculate_analysis(df)
            return jsonify({
                'original_text': result['original_text'],
                'sentiment': result['sentiment'],
                'polarity': round(result['polarity'], 3),
                'positive_polarity': analysis['positive_polarity'],
                'negative_polarity': analysis['negative_polarity'],
                'combined_polarity': analysis['combined_polarity'],
                'positive_words': analysis['positive_words'],
                'negative_words': analysis['negative_words']
            })
        else:
            logger.error("Sentiment analysis failed")
            return jsonify({'error': 'Failed to analyze sentiment'}), 500
    return jsonify({'error': 'Please enter some text'}), 400

@app.route('/visualize', methods=['GET'])
def visualize():
    global results
    if not results:
        logger.warning("No results to visualize")
        return jsonify({'error': 'No data to visualize yet'}), 400
    try:
        df = pd.DataFrame(results)
        if df.empty or not all(col in df for col in ['original_text', 'clean_text', 'tokens', 'sentiment', 'polarity']):
            logger.error("Invalid DataFrame structure")
            return jsonify({'error': 'Invalid data structure for visualization'}), 400
        plots = generate_plots(df)
        return jsonify(plots)
    except Exception as e:
        logger.error(f"Visualization endpoint error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'An error occurred while generating visualizations. Check the server logs for details.'}), 500

@app.route('/clear', methods=['POST'])
def clear():
    global results
    results = []
    logger.info("Results cleared")
    return jsonify({'message': 'Results cleared'})

if __name__ == '__main__':
    app.run(debug=True)