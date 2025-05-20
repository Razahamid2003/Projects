# %% [markdown]
# # PA1: Naive Bayes
# 
# ### Introduction
# 
# In this notebook, you will be implementing two types of Naive Bayes model based on the dataset features and task requirements.
# 
# For reference and additional details, please go through [Chapter 4](https://web.stanford.edu/~jurafsky/slp3/) of the SLP3 book.
# 
# In this assignment, you are provided with two datasets. One is suitable for **Multinomial Naive Bayes**, while the other is appropriate for **Bernoulli Naive Bayes**. Your task is to:
# 1. Analyze both datasets and determine which Naive Bayes model to apply based on the dataset’s characteristics.
# 2. Implement both **Multinomial** and **Bernoulli Naive Bayes** from scratch, adhering to the guidelines below regarding allowed libraries.
# 3. Finally, apply the corresponding models using the `sklearn` library and compare the results with your own implementation.
# 
# ### Guidelines:
# - Use only **numpy** and **pandas** for the manual implementation of Naive Bayes classifiers. No other libraries should be used for this part.
# - For the final part of the assignment, you will use **sklearn** to compare your implementation results.
# 
# 
# ### Instructions
# 
# - Follow along with the notebook, filling out the necessary code where instructed.
# 
# - <span style="color: red;">Read the Submission Instructions, Plagiarism Policy, and Late Days Policy in the attached PDF.</span>
# 
# - <span style="color: red;">Make sure to run all cells for credit.</span>
# 
# - <span style="color: red;">Do not remove any pre-written code.</span>
# 
# - <span style="color: red;">You must attempt all parts.</span>

# %% [markdown]
# All necessary libraries for this assignment have already been added. You are not allowed to add any additional imports.

# %%
!pip install datasets
!pip install nltk

# %%
# Standard library imports
import numpy as np
import regex as re

# Third-party library imports
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from datasets import load_dataset

# NLTK-specific download
nltk.download("punkt")

# %% [markdown]
# ## 1. Loading the Datasets
# 
# In this assignment, you are provided with two datasets:
# 
# - **Dataset 1**: Golf Dataset (available in CSV format in the given folder)
# - **Dataset 2**: Tweet Evaluation Dataset (to be loaded from Hugging Face)
# 
# ### Instructions:
# 
# 1. **Golf Dataset**: You can find the CSV file of the Golf Dataset in the resources provided with this assignment. This dataset aims to explore factors that influence the decision to play golf, which could be valuable for predictive modeling tasks. ​​
# 
# 2. **Tweet Evaluation Dataset**: Instead of downloading the dataset manually, we will be using the [`datasets`](https://huggingface.co/docs/datasets) library from Hugging Face to automatically download and manage the Tweet Eval dataset. This library is part of the Hugging Face ecosystem, widely used for Natural Language Processing (NLP) tasks. The `datasets` library not only downloads the dataset but also offers a standardized interface for accessing and handling the data, making it compatible with other popular libraries like Pandas and PyTorch. Format each split of the dataset into a Pandas DataFrame. The columns should be `text` and `label`, where `text` is the sentence and `label` is the emotion label. The goal is to classify tweets into various emotional categories (e.g., joy, sadness, anger) by analyzing their content.
# 
#    You can explore the extensive list of datasets available on Hugging Face [here](https://huggingface.co/datasets).
# 
# ### Why Use Hugging Face?
# 
# Familiarizing yourself with Hugging Face tools now will be beneficial for future assignments and projects, where we will be relying on this platform for various NLP-related tasks. It simplifies data handling and ensures smooth integration with machine learning workflows.
# 
# ### Task:
# 
# - Explore both datasets and identify their key features. This will help you determine which dataset is best suited for **Multinomial Naive Bayes** and which is better suited for **Bernoulli Naive Bayes**. You can read more about Bernoulli Naive Bayes [here](https://medium.com/@gridflowai/part-2-dive-into-bernoulli-naive-bayes-d0cbcbabb775).
# 

# %%
# code here
golf_data = pd.read_csv('D:\\LUMS Files\\Fall24-25\\ML\\Codes\\PA1.1\\golf_data.csv')

# %%
# code here
tweet_data = load_dataset('tweet_eval', 'emotion', cache_dir="datasets")

# %% [markdown]
# ##### Before proceeding with further tasks, ensure you have determined which type of Naive Bayes is most suitable for each dataset.

# %% [markdown]
# ## 2. Data Preprocessing

# %% [markdown]
# ### 2.1 Preprocessing the Golf Dataset

# %% [markdown]
# In this task, you will apply one-hot encoding to the categorical columns of the Golf dataset and split the data into training and test sets. You can use `sklearn's` `train_test_split` which has been imported for you above. Ensure that the `test_size` parameter is set to 0.3.

# %%
X = golf_data.drop('Play', axis=1)
y = golf_data['Play']

X_encoded = pd.get_dummies(X, drop_first=True)
X_encoded = X_encoded.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# %% [markdown]
# ### 2.2 Preprocessing the Tweet Eval Dataset

# %% [markdown]
# At this stage, you need to pre-process your data to ensure it's in a clean format for further analysis. The following steps should be performed:
# 
# - Remove any URL.
# - Remove punctuation and non-alphanumeric characters.
# - Convert all text to lowercase.
# - Remove any extra whitespace.
# - Eliminate common stopwords.
# 
# In the cell below, implement a function that carries out these tasks. You can utilize the `re` library for cleaning text and the `nltk` library for removing stopwords.
# 
# Once the function is complete, apply it to the `text` column of your dataset to obtain the preprocessed text.
# 

# %%
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

tweet_data = load_dataset('tweet_eval', 'emotion', cache_dir="datasets")

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.lower()
    text = text.replace('user', '')
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    text = text.lower()
    return text

tweet_data = tweet_data.map(lambda x: {'text': preprocess_text(x['text'])})
tetx = tweet_data['train']['text']

# %% [markdown]
# ## 3. Implementing Naive Bayes from Scratch

# %% [markdown]
# ## 3.1 Bernoulli Naive Bayes
# 
# ### From Scratch
# 
# Recall that the Bernoulli Naive Bayes model is based on **Bayes' Theorem**:
# 
# $$
# P(y \mid x) = \frac{P(x \mid y)P(y)}{P(x)}
# $$
# 
# What we really want is to find the class \(c\) that maximizes \(P(c \mid x)\), so we can use the following equation:
# 
# $$
# \hat{c} = \underset{c}{\text{argmax}} \ P(c \mid x) = \underset{c}{\text{argmax}} \ P(x \mid c)P(c)
# $$
# 
# In the case of **Bernoulli Naive Bayes**, we assume that each word \(x_i\) in a sentence follows a **Bernoulli distribution**, meaning that the word either appears (1) or does not appear (0) in the document. We can simplify the formula using this assumption:
# 
# $$
# \hat{c} = \underset{c}{\text{argmax}} \ P(c) \prod_{i=1}^{n} P(x_i = 1 \mid c)^{x_i} P(x_i = 0 \mid c)^{1 - x_i}
# $$
# 
# Where:
# 
# - $x_i = 1$ if the $i^{\text{th}}$ word is present in the document.
# - $x_i = 0$ if the $i^{\text{th}}$ word is not present in the document.
# 
# 
# We can estimate $P(c)$ by counting the number of times each class appears in our training data, and dividing by the total number of training examples. We can estimate $P(x_i = 1 \mid c)$ by counting the number of documents in class $c$ that contain the word $x_i$, and dividing by the total number of documents in class $c$.
# 
# ### **Important: Laplace Smoothing**
# 
# When calculating $P(x_i = 1 \mid c)$ and $P(x_i = 0 \mid c)$, we apply **Laplace smoothing** to avoid zero probabilities. This is essential because, without it, any word that has not appeared in a document of class $c$ will have a probability of zero, which would make the overall product zero, leading to incorrect classification.
# 
# **Reason**: Laplace smoothing ensures that we don't encounter zero probabilities by adding a small constant (typically 1) to both the numerator and the denominator. This is particularly useful when a word has never appeared in the training data for a specific class.
# 
# The smoothed probability formula is:
# 
# $$
# P(x_i = 1 \mid c) = \frac{\text{count of documents in class } c \text{ where } x_i = 1 + 1}{\text{total documents in class } c + 2}
# $$
# 
# This ensures no word has a zero probability, even if it was unseen in the training data.
# 
# ### Avoiding Underflow with Logarithms:
# 
# To avoid underflow errors due to multiplying small probabilities, we apply logarithms, which convert the product into a sum:
# 
# $$
# \hat{c} = \underset{c}{\text{argmax}} \ \log P(c) + \sum_{i=1}^{n} \left[ x_i \log P(x_i = 1 \mid c) + (1 - x_i) \log P(x_i = 0 \mid c) \right]
# $$
# 
# You will now implement this algorithm.
# 
# <span style="color: red;"> For this part, the only external library you will need is `numpy`. You are not allowed to use anything else.</span>
# 

# %%
class BernoulliNaiveBayes:
    def __init__(self):
        self.class_probabilities = None
        self.feature_probabilities = None
        self.classes = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.class_probabilities = np.zeros(n_classes)
        self.feature_probabilities = np.zeros((n_classes, n_features, 2))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_probabilities[idx] = X_c.shape[0] / n_samples
            self.feature_probabilities[idx, :, 1] = (np.sum(X_c, axis=0) + 1) / (X_c.shape[0] + 2)
            self.feature_probabilities[idx, :, 0] = 1 - self.feature_probabilities[idx, :, 1]

    def predict(self, X):
        n_samples, n_features = X.shape
        n_classes = len(self.classes)
        
        log_probabilities = np.zeros((n_samples, n_classes))

        for idx, c in enumerate(self.classes):
            log_prior = np.log(self.class_probabilities[idx])
            log_likelihood = X * np.log(self.feature_probabilities[idx, :, 1]) + \
                             (1 - X) * np.log(self.feature_probabilities[idx, :, 0])
            
            log_probabilities[:, idx] = log_prior + np.sum(log_likelihood, axis=1)

        return self.classes[np.argmax(log_probabilities, axis=1)]

# %% [markdown]
# Now use your implementation to train a Naive Bayes model on the training data, and generate predictions for the Validation Set.
# 
# Report the Accuracy, Precision, Recall, and F1 score of your model on the validation data. Also display the Confusion Matrix. You are allowed to use `sklearn.metrics` for this.
# 
# We wiill be discussing these metrics in detail in the upcoming lectures.

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

model = BernoulliNaiveBayes()
model.fit(X_train, y_train)

#-----------------------------------------------------------------------------------#

y_pred_val = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred_val)
precision = precision_score(y_val, y_pred_val, average='weighted')
recall = recall_score(y_val, y_pred_val, average='weighted')
f1 = f1_score(y_val, y_pred_val, average='weighted')

print("Bernoulli Naive Bayes Metrics (Validation Data):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
conf_matrix = confusion_matrix(y_val, y_pred_val)
print("Confusion Matrix:")
print(conf_matrix)

#-----------------------------------------------------------------------------------#

y_pred_test = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test, average='weighted')
recall = recall_score(y_test, y_pred_test, average='weighted')
f1 = f1_score(y_test, y_pred_test, average='weighted')

print("Bernoulli Naive Bayes Metrics (Test Data):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
conf_matrix = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix:")
print(conf_matrix)

# %% [markdown]
# ## 3.2 Multinomial Naive Bayes (Manual Implementation)

# %% [markdown]
# ### Vectorizing sentences with Bag of Words
# 
# Now that we have loaded in our data, we will need to vectorize our sentences - this is necessary to be able to numericalize our inputs before feeding them into our model. 
# 
# We will be using a Bag of Words approach to vectorize our sentences. This is a simple approach that counts the number of times each word appears in a sentence. 
# 
# The element at index $\text{i}$ of the vector will be the number of times the $\text{i}^{\text{th}}$ word in our vocabulary appears in the sentence. So, for example, if our vocabulary is `["the", "cat", "sat", "on", "mat"]`, and our sentence is `"the cat sat on the mat"`, then our vector will be `[2, 1, 1, 1, 1]`.
# 
# You will now create a `BagOfWords` class to vectorize our sentences. This will involve creating
# 
# 1. A vocabulary from our corpus
# 
# 2. A mapping from words to indices in our vocabulary
# 
# 3. A function to vectorize a sentence in the fashion described above
# 
# It may help you to define something along the lines of a `fit` and a `vectorize` method.

# %%
class BagOfWords:
    def __init__(self):
        self.vocab = {}
    
    def fit(self, corpus):
        vocab_set = set()
        
        for sentence in corpus:
            words = sentence.split()
            vocab_set.update(words)
        
        self.vocab = {word: idx for idx, word in enumerate(sorted(vocab_set))}

    def vectorize(self, sentence):
        vector = np.zeros(len(self.vocab))
        words = sentence.split()
        
        for word in words:
            if word in self.vocab:
                vector[self.vocab[word]] += 1
                
        return vector

    def transform(self, corpus):
        return np.array([self.vectorize(sentence) for sentence in corpus])

# %% [markdown]
# For a sanity check, you can manually set the vocabulary of your `BagOfWords` object to the vocabulary of the example above, and check that the vectorization of the sentence is correct.
# 
# Once you have implemented the `BagOfWords` class, fit it to the training data, and vectorize the training, validation, and test data.

# %%
corpus_train = tweet_data['train']['text']
corpus_val = tweet_data['validation']['text']
corpus_test = tweet_data['test']['text']

bow = BagOfWords()
bow.fit(corpus_train)

X_train_bow = bow.transform(corpus_train)
X_val_bow = bow.transform(corpus_val)
X_test_bow = bow.transform(corpus_test)

# %% [markdown]
# 
# 
# ### From Scratch
# 
# Now that we have vectorized our sentences, we can implement our Naive Bayes model. Recall that the Naive Bayes model is based off of the Bayes Theorem:
# 
# $$
# P(y \mid x) = \frac{P(x \mid y)P(y)}{P(x)}
# $$
# 
# What we really want is to find the class $c$ that maximizes $P(c \mid x)$, so we can use the following equation:
# 
# $$
# \hat{c} = \underset{c}{\text{argmax}} \ P(c \mid x) = \underset{c}{\text{argmax}} \ P(x \mid c)P(c)
# $$
# 
# We can then use the Naive Bayes assumption to simplify this:
# 
# $$
# \hat{c} = \underset{c}{\text{argmax}} \ P(c \mid x) = \underset{c}{\text{argmax}} \ P(c) \prod_{i=1}^{n} P(x_i \mid c)
# $$
# 
# Where $x_i$ is the $i^{\text{th}}$ word in our sentence.
# 
# All of these probabilities can be estimated from our training data. We can estimate $P(c)$ by counting the number of times each class appears in our training data, and dividing by the total number of training examples. We can estimate $P(x_i \mid c)$ by counting the number of times the $i^{\text{th}}$ word in our vocabulary appears in sentences of class $c$, and dividing by the total number of words in sentences of class $c$.
# 
# It would help to apply logarithms to the above equation so that we translate the product into a sum, and avoid underflow errors. This will give us the following equation:
# 
# $$
# \hat{c} = \underset{c}{\text{argmax}} \ \log P(c) + \sum_{i=1}^{n} \log P(x_i \mid c)
# $$
# 
# You will now implement this algorithm. It would help to go through [this chapter from SLP3](https://web.stanford.edu/~jurafsky/slp3/4.pdf) to get a better understanding of the model - **it is recommended base your implementation off the pseudocode that has been provided on Page 6**. You can either make a `NaiveBayes` class, or just implement the algorithm across two functions.
# 
# <span style="color: red;"> For this part, the only external library you will need is `numpy`. You are not allowed to use anything else.</span>

# %%
class MultinomialNaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.conditional_probs = {}
        self.vocab = None
        self.class_counts = {}
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.vocab = np.arange(n_features)

        self.class_counts = {}
        for label in y:
            if label in self.class_counts:
                self.class_counts[label] += 1
            else:
                self.class_counts[label] = 1
        
        total_samples = len(y)
        self.class_priors = {label: count / total_samples for label, count in self.class_counts.items()}
        
        self.conditional_probs = {}
        for label in self.class_counts:
            self.conditional_probs[label] = np.zeros(n_features)
        
        for idx, label in enumerate(y):
            self.conditional_probs[label] += X[idx]
        
        for label in self.conditional_probs:
            total_words = self.conditional_probs[label].sum()
            self.conditional_probs[label] = (self.conditional_probs[label] + 1) / (total_words + n_features)

    def predict(self, X):
        predictions = []
        for x in X:
            log_probs = {}
            for label in self.class_priors:
                log_prob = np.log(self.class_priors[label])
                log_prob += np.sum(np.log(self.conditional_probs[label]) * x)
                log_probs[label] = log_prob
            
            predicted_class = max(log_probs, key=log_probs.get)
            predictions.append(predicted_class)
        
        return predictions

# %% [markdown]
# Now use your implementation to train a Naive Bayes model on the training data, and generate predictions for the Validation Set.
# 
# Report the Accuracy, Precision, Recall, and F1 score of your model on the validation data. Also display the Confusion Matrix. You are allowed to use `sklearn.metrics` for this.

# %%
mnb = MultinomialNaiveBayes()
mnb.fit(X_train_bow, tweet_data['train']['label'])

#-----------------------------------------------------------------------------------#

y_val_pred = mnb.predict(X_val_bow)

accuracy = accuracy_score(tweet_data['validation']['label'], y_val_pred)
precision = precision_score(tweet_data['validation']['label'], y_val_pred, average='weighted')
recall = recall_score(tweet_data['validation']['label'], y_val_pred, average='weighted')
f1 = f1_score(tweet_data['validation']['label'], y_val_pred, average='weighted')
conf_matrix = confusion_matrix(tweet_data['validation']['label'], y_val_pred)

print("Multinomial Naive Bayes Metrics (Validation Data):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

#-----------------------------------------------------------------------------------#

y_test_pred = mnb.predict(X_test_bow)

accuracy = accuracy_score(tweet_data['test']['label'], y_test_pred)
precision = precision_score(tweet_data['test']['label'], y_test_pred, average='weighted')
recall = recall_score(tweet_data['test']['label'], y_test_pred, average='weighted')
f1 = f1_score(tweet_data['test']['label'], y_test_pred, average='weighted')
conf_matrix = confusion_matrix(tweet_data['test']['label'], y_test_pred)

print("Bernoulli Naive Bayes Metrics (Test Data):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# %% [markdown]
# ## 4. Implementing Naive Bayes using sklearn
# 
# In this section, you will compare your manual implementations with `sklearn`'s implementations of both of the Naive Bayes models we have covered above.

# %%
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

bnb = BernoulliNB()
bnb.fit(X_train, y_train)

y_predval_sk_bnb = bnb.predict(X_val)

accuracy_bnb = accuracy_score(y_val, y_predval_sk_bnb)
precision_bnb = precision_score(y_val, y_predval_sk_bnb, average='weighted')
recall_bnb = recall_score(y_val, y_predval_sk_bnb, average='weighted')
f1_bnb = f1_score(y_val, y_predval_sk_bnb, average='weighted')
conf_matrix_bnb = confusion_matrix(y_val, y_predval_sk_bnb)

print("Sklearn Bernoulli Naive Bayes Metrics (Validation):")
print(f"Accuracy: {accuracy_bnb:.4f}")
print(f"Precision: {precision_bnb:.4f}")
print(f"Recall: {recall_bnb:.4f}")
print(f"F1 Score: {f1_bnb:.4f}")
print("Confusion Matrix:")
print(conf_matrix_bnb)

y_predtest_sk_bnb = bnb.predict(X_test)

accuracy_bnb = accuracy_score(y_test, y_predtest_sk_bnb)
precision_bnb = precision_score(y_test, y_predtest_sk_bnb, average='weighted')
recall_bnb = recall_score(y_test, y_predtest_sk_bnb, average='weighted')
f1_bnb = f1_score(y_test, y_predtest_sk_bnb, average='weighted')
conf_matrix_bnb = confusion_matrix(y_test, y_predtest_sk_bnb)

print("Sklearn Bernoulli Naive Bayes Metrics (Test):")
print(f"Accuracy: {accuracy_bnb:.4f}")
print(f"Precision: {precision_bnb:.4f}")
print(f"Recall: {recall_bnb:.4f}")
print(f"F1 Score: {f1_bnb:.4f}")
print("Confusion Matrix:")
print(conf_matrix_bnb)

#------------------------------------------------------------------------------------------------------------#

mnb_sklearn = MultinomialNB()
mnb_sklearn.fit(X_train_bow, tweet_data['train']['label'])

y_predval_sk_mnb = mnb_sklearn.predict(X_val_bow)

accuracy_mnb = accuracy_score(tweet_data['validation']['label'], y_predval_sk_mnb)
precision_mnb = precision_score(tweet_data['validation']['label'], y_predval_sk_mnb, average='weighted')
recall_mnb = recall_score(tweet_data['validation']['label'], y_predval_sk_mnb, average='weighted')
f1_mnb = f1_score(tweet_data['validation']['label'], y_predval_sk_mnb, average='weighted')
conf_matrix_mnb = confusion_matrix(tweet_data['validation']['label'], y_predval_sk_mnb)

print("Sklearn Multinomial Naive Bayes Metrics (Validation):")
print(f"Accuracy: {accuracy_mnb:.4f}")
print(f"Precision: {precision_mnb:.4f}")
print(f"Recall: {recall_mnb:.4f}")
print(f"F1 Score: {f1_mnb:.4f}")
print("Confusion Matrix:")
print(conf_matrix_mnb)

y_predtest_sk_mnb = mnb_sklearn.predict(X_test_bow)

accuracy_mnb = accuracy_score(tweet_data['test']['label'], y_predtest_sk_mnb)
precision_mnb = precision_score(tweet_data['test']['label'], y_predtest_sk_mnb, average='weighted')
recall_mnb = recall_score(tweet_data['test']['label'], y_predtest_sk_mnb, average='weighted')
f1_mnb = f1_score(tweet_data['test']['label'], y_predtest_sk_mnb, average='weighted')
conf_matrix_mnb = confusion_matrix(tweet_data['test']['label'], y_predtest_sk_mnb)

print("Sklearn Multinomial Naive Bayes Metrics:")
print(f"Accuracy: {accuracy_mnb:.4f}")
print(f"Precision: {precision_mnb:.4f}")
print(f"Recall: {recall_mnb:.4f}")
print(f"F1 Score: {f1_mnb:.4f}")
print("Confusion Matrix:")
print(conf_matrix_mnb)


# %% [markdown]
# ## 5. Conclusion
# 
# 1. Explain the key factors you considered when determining which dataset is more suitable for **Multinomial Naive Bayes** and which is better suited for **Bernoulli Naive Bayes**.

# %% [markdown]
# ***Nature of the Data:***
# Multinomial is best suited for datasets where the features represent counts or frequencies of events. Whereas Binomial is ideal for datasets where the features are binary.
# 
# ***Data Distribution:***
# Multinomial is suited for data that has features that can take many non-zero integer values. However Binomial checks for only the absence or presence of a value.
# 
# ***Pattern Finding:***
# Multinomial assumes independence between features but allows for variability in feature count, which can capture more nuanced patterns in the data. However, Binomial only assumes independence of features based solely on presence/absence, which may be simpler but might miss out on patterns related to frequency.
# 
# Thus the choice between Binomial and Multinomial for a dataset comes depends on wether the frequency of a feature plays a big role in determening th epredicted output. If it's presence or absence alone is enough of a determening factor, then Binomial Naive Bayes is preferred. Thus why it was used to predict outputs for the Golf Dataset.


