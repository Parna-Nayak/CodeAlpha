import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import os 

url = "https://en.wikipedia.org/wiki/Web_scraping"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

data = []

for item in soup.find_all("a"):
    title = item.get_text(strip=True)
    link = item.get("href")

    if title and link and link.startswith("/wiki/"):
        full_link = urljoin(url, link)
        data.append({
            "Title": title,
            "Link": full_link
        })

df = pd.DataFrame(data)

# ==========================================
# UPDATED SAVE SECTION
# ==========================================
# 1. Define the exact path to your TASK1 folder
save_path = r'C:\CODEALPHA\TASK1\scraped_data.csv'

# 2. PRO TIP: This automatically creates the TASK1 folder if you forgot to make it!
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# 3. Save the file. We use encoding='utf-8' so Wikipedia's special symbols don't break it.
df.to_csv(save_path, index=False, encoding='utf-8')

print(f"\n✅ SUCCESS! Your file is saved at:")
print(save_path)

print(f"\nTotal Wikipedia articles collected: {len(df)}")