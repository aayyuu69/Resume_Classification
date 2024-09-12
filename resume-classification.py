#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data Loading and Exploration
import pandas as pd
df = pd.read_csv("D:/UpdatedResumeDataSet.csv")


# In[2]:


print(df.head())


# In[3]:


print(df.isnull().sum())

# If there are missing values, handle them appropriately
# For example, dropping rows with missing values:
df = df.dropna()


# In[5]:


print(df.columns)


# In[10]:


print("Columns in the DataFrame:")
for i, column in enumerate(df.columns):
    print(f"{i}: {column}")


# In[7]:


print(df.info())
print("\nColumn names:")
print(df.columns)
print("\nFirst few rows:")
print(df.head())


# In[12]:


import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Use the 'Resume' column for cleaning
df['cleaned_text'] = df['Resume'].apply(clean_text)

print("Cleaning complete. First few rows of cleaned text:")
print(df['cleaned_text'].head())


# In[13]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

# Tokenization
df['tokens'] = df['cleaned_text'].apply(word_tokenize)

# Lemmatization
lemmatizer = WordNetLemmatizer()
df['lemmatized'] = df['tokens'].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

print("Tokenization and Lemmatization complete. First few rows:")
print(df[['tokens', 'lemmatized']].head())


# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Join the lemmatized tokens back into strings
df['lemmatized_text'] = df['lemmatized'].apply(' '.join)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['lemmatized_text'])
y = df['Category']  # Assuming 'Category' is your label column

print("Feature extraction complete.")
print("Shape of feature matrix:", X.shape)
print("Sample of feature matrix:")
print(X[:5].toarray())


# In[8]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
# Replace 'your_data_file.csv' with the actual name of your CSV file
df = pd.read_csv("D:/UpdatedResumeDataSet.csv")

# Text preprocessing
def clean_text(text):
    # Add your text cleaning logic here if needed
    # For now, we'll just return the text as is
    return str(text)

df['cleaned_text'] = df['Resume'].apply(clean_text)

# Feature extraction
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_text'])
y = df['Category']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# # Feature Importance
# feature_importance = pd.DataFrame({'feature': tfidf.get_feature_names_out(), 'importance': abs(model.coef_[0])})
# feature_importance = feature_importance.sort_values('importance', ascending=False)
# print("\nTop 10 most important features:")
# print(feature_importance.head(10))


# In[6]:


# Feature Importance
try:
    feature_names = tfidf.get_feature_names_out()
except AttributeError:
    feature_names = tfidf.get_feature_names()

feature_importance = pd.DataFrame({'feature': feature_names, 'importance': abs(model.coef_[0])})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 10 most important features:")
print(feature_importance.head(10))


# In[7]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Binarize the output
y_test_bin = label_binarize(y_test, classes=np.unique(y))
y_score = model.decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test_bin.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




