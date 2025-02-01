# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Importing essential libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
import numpy as np
from tqdm import tqdm
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# Load dataset
data_path = '/content/drive/MyDrive/Flipkart Data.csv'
data = pd.read_csv(data_path)

# Display first few rows
print("Dataset Preview:")
print(data.head())

# Unique ratings
print("\nUnique Ratings:", pd.unique(data['rating']))

# Plot rating distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='rating', order=data['rating'].value_counts().index, palette='coolwarm')
plt.title("Review Ratings Distribution")
plt.show()

# Assign labels (1 = Positive, 0 = Negative)
data['label'] = np.where(data['rating'] >= 5, 1, 0)

# Function to preprocess reviews
def preprocess_text(texts):
    stop_words = set(stopwords.words('english'))
    cleaned_texts = []
    
    for sentence in tqdm(texts, desc="Preprocessing Text"):
        sentence = re.sub(r'[^\w\s]', '', str(sentence))  # Remove punctuations
        words = nltk.word_tokenize(sentence.lower())  # Tokenize & lowercase
        filtered_sentence = ' '.join(word for word in words if word not in stop_words)  
        cleaned_texts.append(filtered_sentence)
    
    return cleaned_texts

# Apply preprocessing
data['cleaned_review'] = preprocess_text(data['review'].astype(str))

# Show processed data
print("\nPreprocessed Data Preview:")
print(data[['review', 'cleaned_review', 'label']].head())

# Visualizing Positive Reviews with WordCloud
positive_text = ' '.join(data['cleaned_review'][data['label'] == 1])
wordcloud = WordCloud(width=1600, height=800, random_state=21, max_font_size=110).generate(positive_text)

plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud for Positive Reviews")
plt.show()

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=2500)
X = vectorizer.fit_transform(data['cleaned_review']).toarray()
y = data['label']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)

# Train Decision Tree Model
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)

# Evaluate model performance
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)

print("\nModel Training Accuracy:", round(train_accuracy * 100, 2), "%")

# Confusion Matrix Visualization
cm = confusion_matrix(y_train, train_predictions)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
cm_display.plot()
plt.title("Confusion Matrix - Training Data")
plt.show()

