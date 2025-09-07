import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud

csv_path = r"C:\Users\MSI KATANA\IdeaProjects\taller-ii-boilerplate\data\preprocessed\reviews_clean.csv"
df = pd.read_csv(csv_path)

df['tokens'] = df['Review_text'].apply(word_tokenize)
df[['Review_text', 'tokens']]
stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish'))
df['tokens_clean'] = df['tokens'].apply(lambda tokens: [w.lower() for w in tokens if w.lower() not in stop_words and w.isalpha()])
df[['Review_text', 'tokens_clean']]
all_tokens = [token for tokens in df['tokens_clean'] for token in tokens]
word_freq = Counter(all_tokens)
word_freq.most_common(10)
print(word_freq.most_common(10))
freq_df = pd.DataFrame(word_freq.most_common(10), columns=['Palabra', 'Frecuencia'])
print(freq_df)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_tokens))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()