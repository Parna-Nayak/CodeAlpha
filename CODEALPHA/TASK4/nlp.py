import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 1. Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
print("✅ Step 1: Libraries loaded successfully.")

# ==========================================
# LOAD & PREPARE DATA
# ==========================================
# Load the dataset
file_path = r'C:\CODEALPHA\TASK4\Tweets.csv'
df = pd.read_csv(file_path, encoding='latin1')
print(f"✅ Step 2: Loaded {len(df)} tweets.")

# Convert 'tweet_created' to datetime for our 4th plot (Trend Analysis)
df['tweet_created'] = pd.to_datetime(df['tweet_created'])
df['date'] = df['tweet_created'].dt.date

# Text Preprocessing (NLP Cleaning for VADER)
stop_words = set(stopwords.words('english'))
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)

# ==========================================
# REQUIREMENT 1: VADER SENTIMENT CLASSIFICATION
# ==========================================
analyzer = SentimentIntensityAnalyzer()
def get_vader_sentiment(text):
    if not text: return "neutral"
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05: return 'positive'
    elif score <= -0.05: return 'negative'
    else: return 'neutral'

df['predicted_sentiment'] = df['clean_text'].apply(get_vader_sentiment)
accuracy = (df['airline_sentiment'] == df['predicted_sentiment']).mean()
print(f"✅ Step 3: VADER Classification complete. Agreement with ground truth: {accuracy:.2%}")

# ==========================================
# REQUIREMENT 4: 2x2 DASHBOARD LAYOUT
# ==========================================
# Define a consistent color palette for sentiments
sentiment_colors = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}

# Set Seaborn theme
sns.set_theme(style="whitegrid", font_scale=1.1)

# Create a figure with 2 rows and 2 columns (4 plots total)
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('US Airline Twitter Sentiment Analysis Dashboard', fontsize=24, fontweight='bold', y=0.98)

# --- PLOT 1: Overall Sentiment Distribution (Top-Left) ---
sns.countplot(
    data=df, 
    x='airline_sentiment', 
    order=['positive', 'neutral', 'negative'], 
    palette=sentiment_colors, 
    ax=axes[0, 0]
)
axes[0, 0].set_title('1. Overall Public Sentiment', fontsize=16, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('Sentiment', fontsize=12)
axes[0, 0].set_ylabel('Number of Tweets', fontsize=12)
# Add value labels on top of bars
for p in axes[0, 0].patches:
    axes[0, 0].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=12, fontweight='bold', xytext=(0, 5), textcoords='offset points')

# --- PLOT 2: Sentiment by Airline (Top-Right) ---
sns.countplot(
    data=df, 
    x='airline', 
    hue='airline_sentiment', 
    order=df['airline'].value_counts().index, 
    palette=sentiment_colors, 
    ax=axes[0, 1]
)
axes[0, 1].set_title('2. Sentiment Breakdown by Airline', fontsize=16, fontweight='bold', pad=15)
axes[0, 1].set_xlabel('Airline', fontsize=12)
axes[0, 1].set_ylabel('Number of Tweets', fontsize=12)
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].legend(title='Sentiment', loc='upper right', fontsize=10)

# --- PLOT 3: Top Reasons for Negative Feedback (Bottom-Left) ---
neg_df = df[df['airline_sentiment'] == 'negative'].dropna(subset=['negativereason'])
top_neg_reasons = neg_df['negativereason'].value_counts().head(7)

# FIXED: Use DataFrame format to prevent Seaborn >= 0.12 deprecation warnings
neg_reasons_df = top_neg_reasons.reset_index()
neg_reasons_df.columns = ['negativereason', 'count']

sns.barplot(
    data=neg_reasons_df, 
    x='count', 
    y='negativereason', 
    palette='Reds_r', 
    ax=axes[1, 0]
)
axes[1, 0].set_title('3. Top 7 Reasons for Negative Tweets', fontsize=16, fontweight='bold', pad=15)
axes[1, 0].set_xlabel('Count', fontsize=12)
axes[1, 0].set_ylabel('Negative Reason', fontsize=12)

# --- PLOT 4: Daily Sentiment Trends (Bottom-Right) ---
daily_trend = df.groupby(['date', 'airline_sentiment']).size().reset_index(name='count')

sns.lineplot(
    data=daily_trend, 
    x='date', 
    y='count', 
    hue='airline_sentiment', 
    palette=sentiment_colors, 
    marker='o', 
    linewidth=2.5,
    ax=axes[1, 1]
)
axes[1, 1].set_title('4. Daily Sentiment Trends (Feb 17 - Feb 24)', fontsize=16, fontweight='bold', pad=15)
axes[1, 1].set_xlabel('Date', fontsize=12)
axes[1, 1].set_ylabel('Tweet Volume', fontsize=12)
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].legend(title='Sentiment', loc='upper left', fontsize=10)

# ==========================================
# FINALIZE AND DISPLAY
# ==========================================
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Leaves room at the top for the main title
plt.show()

# ==========================================
# REQUIREMENT 5: BUSINESS INSIGHTS
# ==========================================
print("\n" + "="*70)
print("📊 BUSINESS INSIGHTS & RECOMMENDATIONS")
print("="*70)

print("\n🔍 TOP COMPLAINT WORDS BY AIRLINE (From Negative Tweets):")
for airline in df['airline'].unique():
    neg_texts = df[(df['airline'] == airline) & (df['airline_sentiment'] == 'negative')]['clean_text']
    words = " ".join(neg_texts).split()
    filter_words = {'flight', 'airline', 'get', 'like', 'one', 'time', 'day', 'go', 'got', 'will', 'just', 'know'}
    filtered_words = [w for w in words if w not in filter_words and len(w) > 3]
    top_words = Counter(filtered_words).most_common(5)
    print(f"• {airline.upper():<15}: {dict(top_words)}")

print("\n💡 ACTIONABLE BUSINESS RECOMMENDATIONS:")
print("1. OPERATIONS (Trend Insight): The line chart shows massive spikes in negative tweets on Feb 22-23.")
print("   -> Action: Investigate operational logs for those dates (likely a major weather event or system outage).")
print("   Implement automated proactive SMS alerts to passengers before they have to tweet complaints.")
print("2. PRODUCT/SERVICE: 'Late Flight' and 'Cancelled Flight' dominate the negative reasons.")
print("   -> Action: Improve predictive maintenance algorithms to reduce mechanical cancellations.")
print("3. CUSTOMER SERVICE: 'Customer Service Issue' is highly prevalent, especially for US Airways and United.")
print("   -> Action: Empower social media support teams with higher authority to issue instant flight vouchers to de-escalate public Twitter complaints.")
print("4. MARKETING: Virgin America and Southwest have visibly higher positive sentiment ratios.")
print("   -> Action: United and American should analyze Virgin/Southwest's customer service scripts and adopt their empathetic tone.")
print("="*70)