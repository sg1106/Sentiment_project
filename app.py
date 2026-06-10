import pandas as pd
import numpy as np
from textblob import TextBlob
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import io
import base64
import json
import datetime
import collections
from flask import Flask, request, render_template, jsonify
import logging
import traceback
import os
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from werkzeug.utils import secure_filename

# Optional file parsing libraries
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    import docx as python_docx
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

try:
    import openpyxl
    XLSX_SUPPORT = True
except ImportError:
    XLSX_SUPPORT = False

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK data: {e}")

app = Flask(__name__)
CORS(app)

vader = SentimentIntensityAnalyzer()

# In-memory storage
results = []
sessions = {}  # Named sessions for project tracking
themes = {}    # Theme/topic tagging


# ─── NLP HELPERS ──────────────────────────────────────────────────────────────

BUSINESS_THEMES = {
    "Product Quality": ["quality", "feature", "bug", "performance", "reliable", "broken", "slow", "fast", "crash", "error", "works", "glitch", "update", "version"],
    "Customer Support": ["support", "help", "service", "response", "agent", "team", "customer", "issue", "resolve", "ticket", "wait", "reply", "contact"],
    "Pricing & Value": ["price", "cost", "expensive", "cheap", "value", "worth", "money", "affordable", "fee", "subscription", "billing", "refund", "overpriced"],
    "User Experience": ["easy", "difficult", "intuitive", "confusing", "simple", "complicated", "interface", "design", "ux", "ui", "navigate", "complex", "smooth"],
    "Delivery & Logistics": ["delivery", "shipping", "late", "fast", "arrived", "package", "tracking", "damaged", "delayed", "courier", "dispatch"],
    "Brand & Trust": ["trust", "brand", "reputation", "honest", "transparent", "reliable", "scam", "fake", "genuine", "authentic", "recommend"],
    "Competitor Mentions": ["competitor", "alternative", "versus", "better", "worse", "switch", "compare", "other", "prefer", "instead"],
}

EMOTION_LEXICON = {
    "joy": ["happy", "delighted", "wonderful", "amazing", "fantastic", "love", "excellent", "great", "brilliant", "superb", "joyful", "pleased", "thrilled"],
    "anger": ["angry", "furious", "outraged", "terrible", "horrible", "awful", "hate", "disgusting", "unacceptable", "worst", "disgrace", "infuriating"],
    "fear": ["worried", "scared", "concerned", "anxious", "nervous", "afraid", "uncertain", "doubt", "risk", "threat", "scary"],
    "sadness": ["disappointed", "sad", "unhappy", "unfortunate", "regret", "sorry", "miserable", "depressed", "let down", "poor"],
    "surprise": ["unexpected", "surprised", "shocked", "amazed", "astonished", "unbelievable", "incredible", "wow", "sudden"],
    "trust": ["trust", "reliable", "dependable", "consistent", "honest", "safe", "secure", "confident", "recommend", "faithful"],
}

URGENCY_KEYWORDS = ["urgent", "immediately", "asap", "critical", "broken", "crash", "down", "cannot", "unable", "fail", "emergency", "now", "fix", "serious", "severe"]
CHURN_RISK_KEYWORDS = ["cancel", "leave", "switching", "switch", "unsubscribe", "quit", "moving on", "goodbye", "refund", "disappointed", "never again", "last time", "competitor"]
OPPORTUNITY_KEYWORDS = ["wish", "would love", "if only", "suggest", "idea", "feature request", "please add", "need", "missing", "could you", "why not", "should have"]


def preprocess_text(text):
    try:
        text_clean = text.lower()
        text_clean = re.sub(r'http\S+|www\S+|https\S+', '', text_clean, flags=re.MULTILINE)
        text_clean = re.sub(r'\d+', '', text_clean)
        text_clean = re.sub(r'[^\w\s]', '', text_clean)
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text_clean)
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        return tokens, ' '.join(tokens)
    except Exception as e:
        logger.error(f"Error in preprocess_text: {e}")
        return [], text


def detect_emotions(text):
    text_lower = text.lower()
    detected = {}
    for emotion, keywords in EMOTION_LEXICON.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            detected[emotion] = count
    if not detected:
        return "neutral"
    return max(detected, key=detected.get)


def detect_themes(text):
    text_lower = text.lower()
    matched = []
    for theme, keywords in BUSINESS_THEMES.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(theme)
    return matched if matched else ["General Feedback"]


def detect_intent_flags(text):
    text_lower = text.lower()
    flags = []
    if any(kw in text_lower for kw in URGENCY_KEYWORDS):
        flags.append("🚨 High Urgency")
    if any(kw in text_lower for kw in CHURN_RISK_KEYWORDS):
        flags.append("⚠️ Churn Risk")
    if any(kw in text_lower for kw in OPPORTUNITY_KEYWORDS):
        flags.append("💡 Feature Request")
    return flags


def classify_nps_bucket(polarity):
    if polarity >= 0.3:
        return "Promoter"
    elif polarity >= -0.1:
        return "Passive"
    else:
        return "Detractor"


def analyze_sentiment(text, source="manual", session_id="default", custom_tags=None):
    try:
        tokens, clean_text = preprocess_text(text)

        # TextBlob
        blob = TextBlob(clean_text)
        tb_polarity = blob.sentiment.polarity
        tb_subjectivity = blob.sentiment.subjectivity

        # VADER
        vader_scores = vader.polarity_scores(text)
        vader_compound = vader_scores['compound']

        # Ensemble polarity (weighted average)
        ensemble_polarity = (tb_polarity * 0.4 + vader_compound * 0.6)
        sentiment = 'Positive' if ensemble_polarity > 0.05 else 'Negative' if ensemble_polarity < -0.05 else 'Neutral'

        # Confidence score
        confidence = min(abs(ensemble_polarity) * 2, 1.0)
        if confidence < 0.2:
            confidence_label = "Low"
        elif confidence < 0.6:
            confidence_label = "Medium"
        else:
            confidence_label = "High"

        # Advanced features
        emotion = detect_emotions(text)
        detected_themes = detect_themes(text)
        intent_flags = detect_intent_flags(text)
        nps_bucket = classify_nps_bucket(ensemble_polarity)
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text.strip()))
        avg_sentence_length = word_count / max(sentence_count, 1)

        return {
            'id': len(results) + 1,
            'original_text': text,
            'clean_text': clean_text,
            'tokens': tokens,
            'sentiment': sentiment,
            'polarity': round(ensemble_polarity, 4),
            'tb_polarity': round(tb_polarity, 4),
            'vader_compound': round(vader_compound, 4),
            'subjectivity': round(tb_subjectivity, 4),
            'confidence': round(confidence, 4),
            'confidence_label': confidence_label,
            'emotion': emotion,
            'themes': detected_themes,
            'intent_flags': intent_flags,
            'nps_bucket': nps_bucket,
            'source': source,
            'session_id': session_id,
            'custom_tags': custom_tags or [],
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': round(avg_sentence_length, 1),
            'vader_pos': round(vader_scores['pos'], 3),
            'vader_neg': round(vader_scores['neg'], 3),
            'vader_neu': round(vader_scores['neu'], 3),
            'timestamp': datetime.datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error in analyze_sentiment: {e}\n{traceback.format_exc()}")
        return None


def calculate_analysis(df):
    try:
        if df.empty:
            return {}
        pos_df = df[df['sentiment'] == 'Positive']
        neg_df = df[df['sentiment'] == 'Negative']
        neu_df = df[df['sentiment'] == 'Neutral']

        # NPS Score calculation
        total = len(df)
        promoters = len(df[df['nps_bucket'] == 'Promoter'])
        detractors = len(df[df['nps_bucket'] == 'Detractor'])
        nps_score = round(((promoters - detractors) / total) * 100, 1) if total > 0 else 0

        # Theme breakdown
        theme_counts = {}
        for _, row in df.iterrows():
            for theme in row.get('themes', []):
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
        top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Emotion breakdown
        emotion_counts = df['emotion'].value_counts().to_dict() if 'emotion' in df.columns else {}

        # Urgency flags
        all_flags = []
        for _, row in df.iterrows():
            all_flags.extend(row.get('intent_flags', []))
        flag_counts = {f: all_flags.count(f) for f in set(all_flags)}

        # Trend (polarity over time)
        trend_data = []
        if 'timestamp' in df.columns and len(df) > 1:
            df_sorted = df.sort_values('timestamp')
            for i, (_, row) in enumerate(df_sorted.iterrows()):
                trend_data.append({'x': i + 1, 'y': round(row['polarity'], 3), 'label': row['original_text'][:40]})

        # Source breakdown
        source_counts = df['source'].value_counts().to_dict() if 'source' in df.columns else {}

        # Top positive/negative words
        positive_words = []
        negative_words = []
        for _, row in df.iterrows():
            if row['sentiment'] == 'Positive':
                positive_words.extend(row.get('tokens', []))
            elif row['sentiment'] == 'Negative':
                negative_words.extend(row.get('tokens', []))

        pos_word_freq = collections.Counter(positive_words).most_common(20)
        neg_word_freq = collections.Counter(negative_words).most_common(20)

        # Subjectivity breakdown
        if 'subjectivity' in df.columns:
            avg_subjectivity = round(df['subjectivity'].mean(), 3)
        else:
            avg_subjectivity = 0

        return {
            'total': total,
            'positive_count': len(pos_df),
            'negative_count': len(neg_df),
            'neutral_count': len(neu_df),
            'positive_pct': round(len(pos_df) / total * 100, 1) if total > 0 else 0,
            'negative_pct': round(len(neg_df) / total * 100, 1) if total > 0 else 0,
            'neutral_pct': round(len(neu_df) / total * 100, 1) if total > 0 else 0,
            'positive_polarity': round(pos_df['polarity'].mean(), 3) if not pos_df.empty else 0,
            'negative_polarity': round(neg_df['polarity'].mean(), 3) if not neg_df.empty else 0,
            'combined_polarity': round(df['polarity'].mean(), 3),
            'avg_subjectivity': avg_subjectivity,
            'nps_score': nps_score,
            'promoters': promoters,
            'passives': len(df[df['nps_bucket'] == 'Passive']),
            'detractors': detractors,
            'top_themes': top_themes,
            'emotion_counts': emotion_counts,
            'flag_counts': flag_counts,
            'trend_data': trend_data,
            'source_counts': source_counts,
            'pos_word_freq': pos_word_freq,
            'neg_word_freq': neg_word_freq,
            'avg_confidence': round(df['confidence'].mean(), 3) if 'confidence' in df.columns else 0,
            'high_urgency_count': all_flags.count("🚨 High Urgency"),
            'churn_risk_count': all_flags.count("⚠️ Churn Risk"),
            'feature_request_count': all_flags.count("💡 Feature Request"),
        }
    except Exception as e:
        logger.error(f"Error calculating analysis: {e}\n{traceback.format_exc()}")
        return {}


def generate_plots(df):
    plots = {}
    plt.style.use('dark_background')
    COLORS = {
        'positive': '#00E5A0',
        'negative': '#FF5C7A',
        'neutral': '#7C83FD',
        'bg': '#0F1117',
        'card': '#1A1D2E',
        'accent': '#7C83FD',
        'text': '#E0E0E0',
    }

    try:
        # 1. Sentiment Distribution (Donut)
        try:
            sc = df['sentiment'].value_counts()
            labels = sc.index.tolist()
            sizes = sc.values.tolist()
            colors_map = {'Positive': COLORS['positive'], 'Negative': COLORS['negative'], 'Neutral': COLORS['neutral']}
            pie_colors = [colors_map.get(l, '#888') for l in labels]

            fig, ax = plt.subplots(figsize=(5, 5), facecolor=COLORS['bg'])
            ax.set_facecolor(COLORS['bg'])
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%',
                                               startangle=90, pctdistance=0.75,
                                               wedgeprops=dict(width=0.5, edgecolor=COLORS['bg'], linewidth=2))
            for t in texts:
                t.set_color(COLORS['text'])
                t.set_fontsize(11)
            for at in autotexts:
                at.set_color(COLORS['bg'])
                at.set_fontweight('bold')
                at.set_fontsize(10)
            ax.set_title('Sentiment Distribution', color=COLORS['text'], fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', facecolor=COLORS['bg'])
            img.seek(0)
            plots['sentiment_distribution'] = base64.b64encode(img.getvalue()).decode()
            plt.close()
        except Exception as e:
            logger.error(f"Donut chart error: {e}")

        # 2. Polarity Timeline
        try:
            if len(df) >= 2:
                fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=COLORS['bg'])
                ax.set_facecolor(COLORS['bg'])
                x = range(1, len(df) + 1)
                polarities = df['polarity'].tolist()
                ax.fill_between(x, polarities, 0, where=[p >= 0 for p in polarities], alpha=0.3, color=COLORS['positive'])
                ax.fill_between(x, polarities, 0, where=[p < 0 for p in polarities], alpha=0.3, color=COLORS['negative'])
                ax.plot(x, polarities, color=COLORS['accent'], linewidth=2.5, marker='o', markersize=5)
                ax.axhline(y=0, color='#555', linestyle='--', linewidth=1)
                ax.set_xlabel('Entry #', color=COLORS['text'], fontsize=10)
                ax.set_ylabel('Polarity Score', color=COLORS['text'], fontsize=10)
                ax.set_title('Sentiment Trend Over Time', color=COLORS['text'], fontsize=14, fontweight='bold')
                ax.tick_params(colors=COLORS['text'])
                for spine in ax.spines.values():
                    spine.set_color('#333')
                plt.tight_layout()
                img = io.BytesIO()
                plt.savefig(img, format='png', bbox_inches='tight', facecolor=COLORS['bg'])
                img.seek(0)
                plots['polarity_trend'] = base64.b64encode(img.getvalue()).decode()
                plt.close()
        except Exception as e:
            logger.error(f"Trend chart error: {e}")

        # 3. Emotion Wheel
        try:
            if 'emotion' in df.columns:
                ec = df['emotion'].value_counts()
                emotion_colors = {
                    'joy': '#FFD700', 'anger': '#FF4444', 'fear': '#9B59B6',
                    'sadness': '#3498DB', 'surprise': '#FF8C00', 'trust': '#2ECC71', 'neutral': '#95A5A6'
                }
                colors = [emotion_colors.get(e, '#888') for e in ec.index]
                fig, ax = plt.subplots(figsize=(6, 4), facecolor=COLORS['bg'])
                ax.set_facecolor(COLORS['bg'])
                bars = ax.barh(ec.index.tolist(), ec.values.tolist(), color=colors, height=0.6, edgecolor=COLORS['bg'])
                for bar, val in zip(bars, ec.values.tolist()):
                    ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2, str(val),
                            va='center', color=COLORS['text'], fontweight='bold')
                ax.set_title('Emotion Breakdown', color=COLORS['text'], fontsize=14, fontweight='bold')
                ax.tick_params(colors=COLORS['text'])
                for spine in ax.spines.values():
                    spine.set_color('#333')
                ax.set_xlabel('Count', color=COLORS['text'])
                plt.tight_layout()
                img = io.BytesIO()
                plt.savefig(img, format='png', bbox_inches='tight', facecolor=COLORS['bg'])
                img.seek(0)
                plots['emotion_breakdown'] = base64.b64encode(img.getvalue()).decode()
                plt.close()
        except Exception as e:
            logger.error(f"Emotion chart error: {e}")

        # 4. Theme Heatmap
        try:
            theme_sentiment = {}
            for _, row in df.iterrows():
                for theme in row.get('themes', []):
                    if theme not in theme_sentiment:
                        theme_sentiment[theme] = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
                    theme_sentiment[theme][row['sentiment']] += 1
            if theme_sentiment:
                themes_list = list(theme_sentiment.keys())[:8]
                sentiments = ['Positive', 'Negative', 'Neutral']
                data_matrix = np.array([[theme_sentiment[t].get(s, 0) for s in sentiments] for t in themes_list])
                fig, ax = plt.subplots(figsize=(7, max(3, len(themes_list) * 0.6 + 1)), facecolor=COLORS['bg'])
                ax.set_facecolor(COLORS['bg'])
                im = ax.imshow(data_matrix, aspect='auto', cmap='RdYlGn', vmin=0)
                ax.set_xticks(range(len(sentiments)))
                ax.set_xticklabels(sentiments, color=COLORS['text'], fontsize=11)
                ax.set_yticks(range(len(themes_list)))
                ax.set_yticklabels(themes_list, color=COLORS['text'], fontsize=9)
                for i in range(len(themes_list)):
                    for j in range(len(sentiments)):
                        ax.text(j, i, str(data_matrix[i, j]), ha='center', va='center',
                                color='black', fontweight='bold', fontsize=11)
                ax.set_title('Theme × Sentiment Matrix', color=COLORS['text'], fontsize=14, fontweight='bold', pad=15)
                plt.tight_layout()
                img = io.BytesIO()
                plt.savefig(img, format='png', bbox_inches='tight', facecolor=COLORS['bg'])
                img.seek(0)
                plots['theme_heatmap'] = base64.b64encode(img.getvalue()).decode()
                plt.close()
        except Exception as e:
            logger.error(f"Theme heatmap error: {e}")

        # 5. Positive Word Cloud
        try:
            positive_texts = ' '.join(df[df['sentiment'] == 'Positive']['clean_text'].dropna())
            if positive_texts.strip() and len(positive_texts.split()) > 2:
                wc = WordCloud(width=700, height=320, background_color='#0F1117',
                               colormap='Greens', max_words=60,
                               prefer_horizontal=0.9).generate(positive_texts)
                fig, ax = plt.subplots(figsize=(8, 4), facecolor=COLORS['bg'])
                ax.set_facecolor(COLORS['bg'])
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('✅ Positive Signals', color=COLORS['positive'], fontsize=14, fontweight='bold')
                plt.tight_layout()
                img = io.BytesIO()
                plt.savefig(img, format='png', bbox_inches='tight', facecolor=COLORS['bg'])
                img.seek(0)
                plots['positive_wordcloud'] = base64.b64encode(img.getvalue()).decode()
                plt.close()
        except Exception as e:
            logger.error(f"Positive word cloud error: {e}")

        # 6. Negative Word Cloud
        try:
            negative_texts = ' '.join(df[df['sentiment'] == 'Negative']['clean_text'].dropna())
            if negative_texts.strip() and len(negative_texts.split()) > 2:
                wc = WordCloud(width=700, height=320, background_color='#0F1117',
                               colormap='Reds', max_words=60,
                               prefer_horizontal=0.9).generate(negative_texts)
                fig, ax = plt.subplots(figsize=(8, 4), facecolor=COLORS['bg'])
                ax.set_facecolor(COLORS['bg'])
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('⚠️ Pain Points', color=COLORS['negative'], fontsize=14, fontweight='bold')
                plt.tight_layout()
                img = io.BytesIO()
                plt.savefig(img, format='png', bbox_inches='tight', facecolor=COLORS['bg'])
                img.seek(0)
                plots['negative_wordcloud'] = base64.b64encode(img.getvalue()).decode()
                plt.close()
        except Exception as e:
            logger.error(f"Negative word cloud error: {e}")

        # 7. NPS Distribution
        try:
            if 'nps_bucket' in df.columns:
                nps_counts = df['nps_bucket'].value_counts()
                nps_order = ['Promoter', 'Passive', 'Detractor']
                nps_colors = [COLORS['positive'], COLORS['neutral'], COLORS['negative']]
                vals = [nps_counts.get(b, 0) for b in nps_order]
                fig, ax = plt.subplots(figsize=(5, 3.5), facecolor=COLORS['bg'])
                ax.set_facecolor(COLORS['bg'])
                bars = ax.bar(nps_order, vals, color=nps_colors, width=0.5, edgecolor=COLORS['bg'])
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                            str(val), ha='center', va='bottom', color=COLORS['text'], fontweight='bold')
                ax.set_title('NPS Bucket Distribution', color=COLORS['text'], fontsize=14, fontweight='bold')
                ax.tick_params(colors=COLORS['text'])
                for spine in ax.spines.values():
                    spine.set_color('#333')
                ax.set_ylabel('Count', color=COLORS['text'])
                plt.tight_layout()
                img = io.BytesIO()
                plt.savefig(img, format='png', bbox_inches='tight', facecolor=COLORS['bg'])
                img.seek(0)
                plots['nps_distribution'] = base64.b64encode(img.getvalue()).decode()
                plt.close()
        except Exception as e:
            logger.error(f"NPS chart error: {e}")

        # 8. Subjectivity vs Polarity Scatter
        try:
            if len(df) >= 3 and 'subjectivity' in df.columns:
                color_map = {'Positive': COLORS['positive'], 'Negative': COLORS['negative'], 'Neutral': COLORS['neutral']}
                colors_scatter = [color_map.get(s, '#888') for s in df['sentiment']]
                fig, ax = plt.subplots(figsize=(6, 4), facecolor=COLORS['bg'])
                ax.set_facecolor(COLORS['bg'])
                ax.scatter(df['polarity'], df['subjectivity'], c=colors_scatter, alpha=0.8, s=80, edgecolors=COLORS['bg'])
                ax.axvline(x=0, color='#555', linestyle='--', linewidth=1)
                ax.axhline(y=0.5, color='#555', linestyle='--', linewidth=1)
                ax.set_xlabel('Polarity (Negative ← → Positive)', color=COLORS['text'])
                ax.set_ylabel('Subjectivity (Objective ← → Subjective)', color=COLORS['text'])
                ax.set_title('Polarity vs Subjectivity Map', color=COLORS['text'], fontsize=14, fontweight='bold')
                ax.tick_params(colors=COLORS['text'])
                for spine in ax.spines.values():
                    spine.set_color('#333')
                legend_elements = [mpatches.Patch(color=COLORS['positive'], label='Positive'),
                                   mpatches.Patch(color=COLORS['negative'], label='Negative'),
                                   mpatches.Patch(color=COLORS['neutral'], label='Neutral')]
                ax.legend(handles=legend_elements, facecolor=COLORS['card'], labelcolor=COLORS['text'])
                plt.tight_layout()
                img = io.BytesIO()
                plt.savefig(img, format='png', bbox_inches='tight', facecolor=COLORS['bg'])
                img.seek(0)
                plots['scatter_map'] = base64.b64encode(img.getvalue()).decode()
                plt.close()
        except Exception as e:
            logger.error(f"Scatter map error: {e}")

    except Exception as e:
        logger.error(f"Error in generate_plots: {e}\n{traceback.format_exc()}")

    return plots


def generate_pm_report(df):
    """Generate a structured PM/consulting report."""
    if df.empty:
        return {}
    analysis = calculate_analysis(df)

    # Priority matrix: urgency vs impact
    priority_items = []
    for _, row in df.iterrows():
        flags = row.get('intent_flags', [])
        urgency = 2 if "🚨 High Urgency" in flags else 1 if "⚠️ Churn Risk" in flags else 0
        impact_score = abs(row['polarity']) + (0.5 if row['confidence_label'] == 'High' else 0)
        priority_items.append({
            'text': row['original_text'][:100],
            'sentiment': row['sentiment'],
            'urgency': urgency,
            'impact': round(impact_score, 2),
            'flags': flags,
            'themes': row.get('themes', []),
            'nps_bucket': row.get('nps_bucket', ''),
        })
    priority_items.sort(key=lambda x: (x['urgency'], x['impact']), reverse=True)

    # Executive summary
    total = analysis.get('total', 0)
    nps = analysis.get('nps_score', 0)
    top_theme = analysis.get('top_themes', [('N/A', 0)])[0][0] if analysis.get('top_themes') else 'N/A'

    if nps >= 30:
        health = "Strong"
        health_color = "positive"
    elif nps >= 0:
        health = "Moderate"
        health_color = "neutral"
    else:
        health = "At Risk"
        health_color = "negative"

    # Key risks and opportunities
    risks = []
    opportunities = []
    for _, row in df.iterrows():
        flags = row.get('intent_flags', [])
        if "⚠️ Churn Risk" in flags or "🚨 High Urgency" in flags:
            risks.append({'text': row['original_text'][:120], 'themes': row.get('themes', [])})
        if "💡 Feature Request" in flags:
            opportunities.append({'text': row['original_text'][:120], 'themes': row.get('themes', [])})

    return {
        'executive_summary': {
            'total_analyzed': total,
            'overall_health': health,
            'health_color': health_color,
            'nps_score': nps,
            'top_theme': top_theme,
            'avg_polarity': analysis.get('combined_polarity', 0),
            'high_urgency': analysis.get('high_urgency_count', 0),
            'churn_risk': analysis.get('churn_risk_count', 0),
        },
        'priority_matrix': priority_items[:10],
        'risks': risks[:5],
        'opportunities': opportunities[:5],
        'theme_breakdown': analysis.get('top_themes', []),
        'emotion_breakdown': analysis.get('emotion_counts', {}),
        'recommended_actions': generate_recommended_actions(analysis),
    }


def generate_recommended_actions(analysis):
    actions = []
    if analysis.get('churn_risk_count', 0) > 0:
        actions.append({
            'priority': 'P0',
            'action': f"Immediate outreach to {analysis['churn_risk_count']} churn-risk customer(s)",
            'rationale': 'Prevent revenue loss from at-risk accounts',
            'owner': 'Customer Success',
        })
    if analysis.get('high_urgency_count', 0) > 0:
        actions.append({
            'priority': 'P0',
            'action': f"Triage {analysis['high_urgency_count']} high-urgency issue(s) within 24hrs",
            'rationale': 'Unresolved critical issues compound churn risk',
            'owner': 'Support / Eng',
        })
    if analysis.get('feature_request_count', 0) > 0:
        actions.append({
            'priority': 'P1',
            'action': f"Review {analysis['feature_request_count']} feature request(s) for roadmap consideration",
            'rationale': 'Unmet needs are competitive vulnerability',
            'owner': 'Product',
        })
    top_themes = analysis.get('top_themes', [])
    if top_themes:
        actions.append({
            'priority': 'P1',
            'action': f"Deep-dive on '{top_themes[0][0]}' — highest volume theme",
            'rationale': 'Most mentioned area likely has systemic issues or opportunities',
            'owner': 'PM / Research',
        })
    if analysis.get('negative_pct', 0) > 40:
        actions.append({
            'priority': 'P1',
            'action': 'Launch root-cause analysis sprint for negative sentiment drivers',
            'rationale': f"{analysis['negative_pct']}% negative rate exceeds acceptable threshold (>40%)",
            'owner': 'Product / Eng',
        })
    if analysis.get('nps_score', 0) < 0:
        actions.append({
            'priority': 'P2',
            'action': 'Initiate NPS recovery programme: identify detractor themes and address systematically',
            'rationale': f"NPS of {analysis.get('nps_score', 0)} is negative — brand risk",
            'owner': 'CPO / CX Lead',
        })
    if not actions:
        actions.append({
            'priority': 'P2',
            'action': 'Continue monitoring — sentiment healthy. Set up automated alerts for score drops.',
            'rationale': 'Proactive monitoring prevents reactive firefighting',
            'owner': 'PM',
        })
    return actions


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    global results
    data = request.get_json() or {}
    text = data.get('text') or request.form.get('text', '')
    source = data.get('source', 'manual')
    session_id = data.get('session_id', 'default')
    custom_tags = data.get('tags', [])

    if not text or not text.strip():
        return jsonify({'error': 'Please enter some text'}), 400

    result = analyze_sentiment(text, source=source, session_id=session_id, custom_tags=custom_tags)
    if not result:
        return jsonify({'error': 'Failed to analyze sentiment'}), 500

    results.append(result)
    df = pd.DataFrame(results)
    analysis = calculate_analysis(df)

    return jsonify({**result, **analysis})


@app.route('/bulk_analyze', methods=['POST'])
def bulk_analyze():
    """Analyze multiple texts at once (CSV/paste)."""
    global results
    data = request.get_json() or {}
    texts = data.get('texts', [])
    source = data.get('source', 'bulk')
    session_id = data.get('session_id', 'default')

    if not texts:
        return jsonify({'error': 'No texts provided'}), 400

    new_results = []
    for text in texts:
        if text and text.strip():
            result = analyze_sentiment(text.strip(), source=source, session_id=session_id)
            if result:
                results.append(result)
                new_results.append(result)

    if not new_results:
        return jsonify({'error': 'No valid texts to analyze'}), 400

    df = pd.DataFrame(results)
    analysis = calculate_analysis(df)
    return jsonify({'count': len(new_results), 'results': new_results, **analysis})


@app.route('/visualize', methods=['GET'])
def visualize():
    global results
    if not results:
        return jsonify({'error': 'No data to visualize yet'}), 400
    try:
        df = pd.DataFrame(results)
        plots = generate_plots(df)
        return jsonify(plots)
    except Exception as e:
        logger.error(f"Visualization error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/pm_report', methods=['GET'])
def pm_report():
    global results
    if not results:
        return jsonify({'error': 'No data for report'}), 400
    df = pd.DataFrame(results)
    report = generate_pm_report(df)
    analysis = calculate_analysis(df)
    return jsonify({**report, **analysis})


@app.route('/export', methods=['GET'])
def export():
    global results
    if not results:
        return jsonify({'error': 'No data to export'}), 400
    df = pd.DataFrame(results)
    # Return JSON-friendly export
    export_data = df[['id', 'original_text', 'sentiment', 'polarity', 'emotion',
                        'nps_bucket', 'confidence_label', 'themes', 'intent_flags',
                        'source', 'timestamp']].to_dict(orient='records')
    return jsonify({'data': export_data, 'count': len(export_data)})


@app.route('/clear', methods=['POST'])
def clear():
    global results
    results = []
    return jsonify({'message': 'Results cleared'})


@app.route('/history', methods=['GET'])
def history():
    global results
    return jsonify({'results': results, 'count': len(results)})


@app.route('/parse_file', methods=['POST'])
def parse_file():
    """
    Accept an uploaded file (.txt, .pdf, .docx, .xlsx / .xls / .csv)
    and return extracted lines ready for bulk analysis.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename).lower()
    lines = []

    try:
        # ── TXT ──────────────────────────────────────────────────────────────
        if filename.endswith('.txt'):
            content = file.read().decode('utf-8', errors='ignore')
            lines = [l.strip() for l in content.splitlines() if l.strip()]

        # ── PDF ──────────────────────────────────────────────────────────────
        elif filename.endswith('.pdf'):
            if not PDF_SUPPORT:
                return jsonify({'error': 'pdfplumber not installed. Run: pip install pdfplumber'}), 500
            file_bytes = io.BytesIO(file.read())
            with pdfplumber.open(file_bytes) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        for l in text.splitlines():
                            l = l.strip()
                            if l:
                                lines.append(l)

        # ── DOCX ─────────────────────────────────────────────────────────────
        elif filename.endswith('.docx'):
            if not DOCX_SUPPORT:
                return jsonify({'error': 'python-docx not installed. Run: pip install python-docx'}), 500
            file_bytes = io.BytesIO(file.read())
            doc = python_docx.Document(file_bytes)
            for para in doc.paragraphs:
                t = para.text.strip()
                if t:
                    lines.append(t)

        # ── XLSX / XLS ────────────────────────────────────────────────────────
        elif filename.endswith(('.xlsx', '.xls')):
            if not XLSX_SUPPORT:
                return jsonify({'error': 'openpyxl not installed. Run: pip install openpyxl'}), 500
            file_bytes = io.BytesIO(file.read())
            wb = openpyxl.load_workbook(file_bytes, read_only=True, data_only=True)
            ws = wb.active
            for row in ws.iter_rows(values_only=True):
                for cell in row:
                    if cell and str(cell).strip():
                        lines.append(str(cell).strip())
            wb.close()

        # ── CSV ───────────────────────────────────────────────────────────────
        elif filename.endswith('.csv'):
            content = file.read().decode('utf-8', errors='ignore')
            import csv, io as _io
            reader = csv.reader(_io.StringIO(content))
            for row in reader:
                for cell in row:
                    cell = cell.strip()
                    if cell:
                        lines.append(cell)

        else:
            return jsonify({'error': f'Unsupported file type. Supported: .txt, .pdf, .docx, .xlsx, .csv'}), 400

        # Filter out very short lines (headers, noise)
        lines = [l for l in lines if len(l) > 10]

        return jsonify({
            'lines': lines,
            'count': len(lines),
            'filename': filename
        })

    except Exception as e:
        logger.error(f"File parse error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Failed to parse file: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5001)), debug=True)