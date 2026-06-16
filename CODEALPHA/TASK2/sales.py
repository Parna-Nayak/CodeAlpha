print("TASK 2: Exploratory Data Analysis (EDA)")

print(". Ask meaningful questions about the dataset before analysis.")
print(". Explore the data structure, including variables and data types.")
print(". Identify trends, patterns and anomalies within the data.")
print(". Test hypotheses and validate assumptions using statistics and visualization.")
print(". Detect potential data issues or problems to address in further analysis.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from spicy import stats

# ==========================================
# PHASE 1: ASK MEANINGFUL QUESTIONS
# ==========================================
print("="*60)
print("PHASE 1: BUSINESS CONTEXT & MEANINGFUL QUESTIONS")
print("="*60)
print("Context: Analyzing 4 years of retail sales data to find profit drivers.")
print("Key Questions:")
print("1. Which product categories are profitable, and which are losing money?")
print("2. Does offering higher discounts increase overall profit?")
print("3. Are there specific regions or segments that perform significantly better?")
print("4. Are there data anomalies (e.g., negative profits, shipping errors)?")
print("\n")

# ==========================================
# PHASE 2 & 5: EXPLORE STRUCTURE & DETECT ISSUES
# ==========================================
print("="*60)
print("PHASE 2 & 5: DATA STRUCTURE & ISSUE DETECTION")
print("="*60)

df = pd.read_csv(r'C:\CODEALPHA\TASK2\Superstore.csv', encoding='latin1')

# 1. Convert Dates
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# 2. Create useful new columns
df['Month'] = df['Order Date'].dt.month
df['Year'] = df['Order Date'].dt.year
df['Profit_Margin'] = df['Profit'] / df['Sales']
df['Shipping_Days'] = (df['Ship Date'] - df['Order Date']).dt.days

print("✅ Data loaded and processed successfully!")
print(df.head())

# Anomaly 1: Negative Profits
neg_profits = df[df['Profit'] < 0]
print(f"🚨 ANOMALY 1: Found {len(neg_profits)} transactions with NEGATIVE profit.")
print("   Top 3 Sub-Categories causing losses:")
print(f"   {neg_profits.groupby('Sub-Category')['Profit'].sum().sort_values().head(3).to_dict()}")

# Anomaly 2: Shipping Date before Order Date
bad_shipping = df[df['Shipping_Days'] < 0]
print(f"🚨 ANOMALY 2: Found {len(bad_shipping)} orders where Ship Date is BEFORE Order Date.")
print("\n")
# ==========================================
# PHASE 3: IDENTIFY TRENDS, PATTERNS, ANOMALIES (4 PLOTS IN ONE PAGE)
# ==========================================
print("="*60)
print("PHASE 3: VISUALIZING TRENDS & PATTERNS (2x2 DASHBOARD)")
print("="*60)
print("*(Check the pop-up window for the combined dashboard)*\n")

sns.set_theme(style="whitegrid")

# Create a 2x2 grid for the dashboard
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
fig.suptitle("Superstore EDA Dashboard: Trends, Patterns & Anomalies", fontsize=18, fontweight='bold', y=0.95)

# 1️⃣ TREND: Monthly Sales vs Profit (Top Left)
monthly_trend = df.groupby(['Year', 'Month']).agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
monthly_trend['Date'] = pd.to_datetime(monthly_trend['Year'].astype(str) + '-' + monthly_trend['Month'].astype(str) + '-01')

axes[0, 0].plot(monthly_trend['Date'], monthly_trend['Sales'], label='Total Sales', color='blue', marker='o', markersize=4)
axes[0, 0].plot(monthly_trend['Date'], monthly_trend['Profit'], label='Total Profit', color='green', marker='s', markersize=4)
axes[0, 0].set_title('1. Monthly Sales vs. Profit Trend')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Amount ($)')
axes[0, 0].legend()
axes[0, 0].tick_params(axis='x', rotation=45)

# 2️⃣ PATTERN: Discount vs Profit (Top Right)
sns.scatterplot(x='Discount', y='Profit', data=df, alpha=0.4, color='purple', ax=axes[0, 1])
axes[0, 1].set_title('2. Relationship Between Discount and Profit')
axes[0, 1].set_xlabel('Discount Rate')
axes[0, 1].set_ylabel('Profit ($)')
axes[0, 1].axhline(0, color='red', linestyle='--') 

# 3️⃣ ANOMALY: Average Profit by Sub-Category (Bar Chart)
# Calculate average profit and sort it so the biggest losses are on the left
avg_profit = df.groupby('Sub-Category')['Profit'].mean().sort_values()

# Use 'coolwarm' palette: Red for negative, Blue for positive
sns.barplot(x=avg_profit.index, y=avg_profit.values, palette='coolwarm', ax=axes[1, 0])

axes[1, 0].set_title('3. Average Profit by Sub-Category')
axes[1, 0].set_xlabel('Sub-Category')
axes[1, 0].set_ylabel('Average Profit ($)')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1.5) # Black line at zero

# 4️⃣ PATTERN: Correlation Heatmap (Bottom Right)
# Extracting only numeric columns to calculate correlation
numeric_df = df[['Sales', 'Quantity', 'Discount', 'Profit']]
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=axes[1, 1])
axes[1, 1].set_title('4. Correlation Heatmap (Numeric Features)')

# Adjust layout to prevent titles from overlapping
plt.tight_layout(rect=[0, 0, 1, 0.93]) 
plt.show()
# ==========================================
# PHASE 4: TEST HYPOTHESES & VALIDATE ASSUMPTIONS
# ==========================================
print("="*60)
print("PHASE 4: STATISTICAL HYPOTHESIS TESTING")
print("="*60)

# TEST 1: T-Test (Consumer vs Corporate Profit)
print("\n--- TEST 1: Do 'Consumer' and 'Corporate' segments have different average profits? ---")
consumers = df[df['Segment'] == 'Consumer']['Profit']
corporates = df[df['Segment'] == 'Corporate']['Profit']
t_stat, p_val_t = stats.ttest_ind(consumers, corporates)
print(f"Mean Profit (Consumer): ${consumers.mean():.2f} | Mean Profit (Corporate): ${corporates.mean():.2f}")
print(f"P-value: {p_val_t:.4f}")
print("✅ RESULT: Statistically significant difference!" if p_val_t < 0.05 else "❌ RESULT: No significant difference.")

# TEST 2: ANOVA (Sales across 4 Regions)
print("\n--- TEST 2: Are average Sales significantly different across the 4 Regions? ---")
regions = [df[df['Region'] == r]['Sales'] for r in ['East', 'West', 'Central', 'South']]
f_stat, p_val_anova = stats.f_oneway(*regions)
print(f"ANOVA P-value: {p_val_anova:.6f}")
print("✅ RESULT: Region significantly impacts sales!" if p_val_anova < 0.05 else "❌ RESULT: Region does not significantly impact sales.")

# TEST 3: Pearson Correlation (Sales vs Profit)
print("\n--- TEST 3: Is there a correlation between Sales and Profit? ---")
corr_stat, p_val_corr = stats.pearsonr(df['Sales'], df['Profit'])
print(f"Pearson Correlation Coefficient: {corr_stat:.4f}")
print(f"P-value: {p_val_corr:.2e}")
print("✅ RESULT: Significant correlation exists!" if p_val_corr < 0.05 else " RESULT: No significant correlation.")

