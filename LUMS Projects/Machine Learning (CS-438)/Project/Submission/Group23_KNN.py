# %% [markdown]
# # 1st Model Implementation (K Nearest Neighbors)

# %% [markdown]
# ## Imports

# %%
import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# # Splitting Data:
# 

# %% [markdown]
# The data has been split using sklearn into Training and Testing data in a 70-30 split.

# %%
data = pd.read_csv('merged_file.csv')
texts = data['Cleaned_content']
labels = data['Gold Label']
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42)

# %% [markdown]
# ### We use this Function to build a vocabulary list of words from the texts that are given to us.`

# %%
def build_vocabulary(texts):
    vocabulary = set()
    for text in texts:
        words = text.split()
        vocabulary.update(words)
    return sorted(list(vocabulary))
vocabulary = build_vocabulary(X_train_texts)
word_to_index = {word: idx for idx, word in enumerate(vocabulary)}

# %% [markdown]
# ### This code calculates the Inverse Document Frequency (IDF) for each word in a given vocabulary based on a set of texts. IDF is a key component of the TF-IDF (Term Frequency-Inverse Document Frequency) approach.

# %%
def compute_idf(texts, vocabulary):
    N = len(texts)
    idf = np.zeros(len(vocabulary))
    for idx, word in enumerate(vocabulary):
        df = sum(1 for text in texts if word in text.split())
        print()
        idf[idx] = math.log((N + 1) / (df + 1)) + 1  
    return idf
idf = compute_idf(X_train_texts, vocabulary)


# %% [markdown]
# Next we convert text data into numerical representations using TF-IDF vectors, which can be directly used as input to our KNN Model.

# %%
def text_to_vector(text, vocabulary, idf):
    tf = np.zeros(len(vocabulary))
    words = text.split()
    word_counts = Counter(words)
    for word in words:
        idx = word_to_index.get(word)
        if idx is not None:
            tf[idx] = word_counts[word]
    tfidf = tf * idf
    norm = np.linalg.norm(tfidf)
    if norm != 0:
        tfidf = tfidf / norm
    return tfidf

X_train_vectors = np.array([text_to_vector(text, vocabulary, idf) for text in X_train_texts])
X_test_vectors = np.array([text_to_vector(text, vocabulary, idf) for text in X_test_texts])

# %% [markdown]
# Next we define a function to compute the cosine similarity between two vectors which is a commonly used metric in text processing and machine learning, especially in models like K-Nearest Neighbors, where distance or similarity between feature vectors matters.

# %%
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

# %% [markdown]
# ## Implementing our Model

# %%
def knn_predict(test_vector, train_vectors, train_labels, k):
    similarities = []
    for idx, train_vector in enumerate(train_vectors):
        similarity = cosine_similarity(test_vector, train_vector)
        similarities.append((similarity, train_labels.iloc[idx]))
    
    similarities.sort(reverse=True, key=lambda x: x[0])    
    top_k = similarities[:k]
    top_k_labels = [label for _, label in top_k]
    label_counts = Counter(top_k_labels)
    predicted_label = label_counts.most_common(1)[0][0]
    
    return predicted_label

# %% [markdown]
# ## Running our model

# %%
k = 5
predictions = []

for test_vector in X_test_vectors:
    predicted_label = knn_predict(test_vector, X_train_vectors, y_train, k)
    predictions.append(predicted_label)

print(classification_report(y_test, predictions))

# %% [markdown]
# ## Evaluation:

# %%
cm = confusion_matrix(y_test, predictions, labels=data['Gold Label'].unique())
sns.heatmap(cm, annot=True, fmt='d', xticklabels=data['Gold Label'].unique(), yticklabels=data['Gold Label'].unique())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


