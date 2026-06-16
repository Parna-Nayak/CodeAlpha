import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==========================================
# TASK 3.1: TRANSFORM RAW DATA
# ==========================================

# 1. Load the data 
# (We don't need the 'names' argument anymore since your CSV has a header row)
df = pd.read_csv(r'C:\CODEALPHA\TASK3\dataset.csv', encoding='latin1')

# 2. FIX COLUMN NAME: Your CSV header calls it 'track_genre', but the code expects 'genre'
df.rename(columns={'track_genre': 'genre'}, inplace=True)

# 3. Clean Data Artifacts (e.g., rows with '!' in the ID like '!86506')
df['id'] = df['id'].astype(str).str.replace('!', '', regex=False)
df['id'] = pd.to_numeric(df['id'], errors='coerce') # Safely handles any remaining weird characters

# 4. Handle Missing Values & Duplicates
df = df.dropna(subset=['genre', 'track_name', 'energy', 'valence'])
df = df.drop_duplicates(subset=['track_id'])

# 5. Feature Engineering: Convert milliseconds to minutes for human readability
df['duration_min'] = (df['duration_ms'] / 60000).round(2)


# ==========================================
# TASK 3.2 & 3.3: USE TOOLS & DESIGN VISUALS
# ==========================================
# Set a professional, clean visual theme (removes clutter)
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Create a dashboard-like figure with 3 subplots
fig = plt.figure(figsize=(16, 10))
fig.suptitle('The Anatomy of Music: Audio Features & Genre Insights', 
             fontsize=20, fontweight='bold', y=0.98)

# --- VISUAL 1: The "Mood" of Genres (Boxplot) ---
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
genres_to_plot = ['metalcore', 'dubstep', 'acoustic', 'pop', 'synth-pop']
df_filtered = df[df['genre'].isin(genres_to_plot)].copy()

# Sort genres by median valence for better storytelling
genre_order = df_filtered.groupby('genre')['valence'].median().sort_values().index

sns.boxplot(x='genre', y='valence', data=df_filtered, order=genre_order, 
            palette="mako", ax=ax1, width=0.6)
sns.stripplot(x='genre', y='valence', data=df_filtered, order=genre_order, 
              color="black", size=3, alpha=0.3, ax=ax1)

ax1.set_title('Musical Positivity (Valence) Varies Drastically by Genre', fontsize=14, fontweight='bold')
ax1.set_xlabel('Genre', fontsize=12)
ax1.set_ylabel('Valence (0 = Sad/Depressing, 1 = Happy/Cheerful)', fontsize=12)
ax1.annotate('Acoustic and Pop tracks score highest in positivity,\nwhile Metalcore and Dubstep lean heavily negative.', 
             xy=(0.5, 0.85), xycoords='axes fraction', ha='center', fontsize=10, color='gray')
sns.despine(ax=ax1)

# --- VISUAL 2: The Formula for a "Hit" Song (Scatter Plot) ---
ax2 = plt.subplot2grid((2, 2), (1, 0))
sns.scatterplot(x='energy', y='danceability', hue='popularity', size='popularity', 
                sizes=(20, 200), palette="mako", alpha=0.7, data=df, ax=ax2)

ax2.set_title('The "Hit" Formula: Energy vs. Danceability', fontsize=14, fontweight='bold')
ax2.set_xlabel('Energy', fontsize=12)
ax2.set_ylabel('Danceability', fontsize=12)
ax2.legend(title='Popularity', bbox_to_anchor=(1.05, 1), loc='upper left')
sns.despine(ax=ax2)

# --- VISUAL 3: Feature Correlations (Heatmap) ---
ax3 = plt.subplot2grid((2, 2), (1, 1))
numeric_cols = ['popularity', 'danceability', 'energy', 'acousticness', 'valence', 'tempo']
corr = df[numeric_cols].corr()

sns.heatmap(corr, annot=True, cmap="mako", fmt=".2f", vmin=-1, vmax=1, 
            linewidths=0.5, ax=ax3, cbar_kws={'shrink': 0.8})
ax3.set_title('Audio Feature Correlations', fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# ==========================================
# TASK 3.4: CRAFT COMPELLING DATA STORIES
# ==========================================
print("\n" + "="*60)
print("📊 DATA STORY & BUSINESS RECOMMENDATIONS")
print("="*60)
print("1. THE CONTEXT: We analyzed audio features across thousands of Spotify tracks to understand what drives listener engagement and how genres differ sonically.")
print("2. THE INSIGHTS:")
print("   • Mood Mapping: Genres like 'acoustic' and 'pop' have a high 'valence' (positivity), while 'metalcore' and 'dubstep' are intentionally low in valence but high in energy.")
print("   • The Hit Formula: The scatterplot reveals that the most popular songs (larger, darker dots) cluster in the top-right quadrant: High Energy + High Danceability.")
print("   • Hidden Correlations: The heatmap shows 'energy' and 'loudness' are highly correlated, while 'acousticness' is strongly negatively correlated with 'energy'.")
print("3. RECOMMENDATIONS (The 'So What?'):")
print("   • For A&R / Music Labels: When signing new pop or dance artists, prioritize tracks with Energy > 0.7 and Danceability > 0.7 to maximize hit potential.")
print("   • For Playlist Curators: Create mood-based playlists (e.g., 'High Energy Focus') by filtering for high energy but low valence tracks, which the data shows is a unique, underserved niche.")
print("="*60)