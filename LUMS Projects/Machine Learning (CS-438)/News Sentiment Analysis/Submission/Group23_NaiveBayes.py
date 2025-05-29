# %% [markdown]
# # 1st Model Implementation (Multinomial Naive Bayes)

# %% [markdown]
# ## Imports

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# %% [markdown]
# ### The cell below is only for when we have the four seperate .csv files that we need to merge and sort.

# %%
file1 = 'cleaned_articles_geo.csv'
file2 = 'cleaned_articles_express.csv'
file3 = 'cleaned_articles_dawn.csv'
file4 = 'cleaned_articles_jang.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)
df4 = pd.read_csv(file4)

merged_df = pd.concat([df1, df2, df3, df4])

merged_df['Index'] = range(len(merged_df))
merged_df.set_index('Index', inplace=True)

output_file = 'merged_file.csv'
merged_df.to_csv(output_file, index=False)

print(f"Merged file saved as {output_file}")


# %% [markdown]
# # Splitting Data:

# %% [markdown]
# The data has been split using sklearn into Training and Testing data in a 70-30 split. Validation data was not included in this model, since this is only the first model and it's implementation. In a Naive Bayes model, having validation data doesn't make much of a difference upon training of the model, thus we will only be using training and testing.

# %%
X = merged_df.drop('Gold Label', axis=1)
Y = merged_df['Gold Label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

print("Shape of Data:", X.shape)
print("Shape of Labels:", Y.shape)
print("Shape of Training Data:", X_train.shape)
print("Shape of Testing Data:", X_test.shape)
print("Shape of Training Labels:", Y_train.shape)
print("Shape of Testing Labels:", Y_test.shape)

# %% [markdown]
# # Multinomial Naive Bayes Implementation:
# ## Bag of Words:
# 
# Since we are implementing a *Multinomial* Naive Bayes model, we will need to create a Bag of Words Function.
# 1. Fit: Creates an empty set 'vocab_set', and for each sentance in the dataset, splits the words and adds every unique (all words only once) into the set.
# 
# 2. Vectorize: This converts every sentance in the dataset into a vector representation, based on the number of repetiotions of each word in that sentance, by creating a zero vector and incrementing the word counter of the vector based on the word's index.
# 
# 3. Transform:  This merely converts an entire corpus into a matrix of vectors

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
# We now create a testing and training corpus based on the test and train dataset. Once done, we initialize the BagofWords function and fit it to the training Corpus. The reason behind this is to maintain consistency when comparing outputs. The corpus of training and testing is then transformed into Bag of Words variables which will be used in the training and testing of our Naive Bayes Model.

# %%
corpus_train = X_train['Cleaned_content']
corpus_test = X_test['Cleaned_content']

bow = BagOfWords()
bow.fit(corpus_train)

X_train_bow = bow.transform(corpus_train)
X_test_bow = bow.transform(corpus_test)

# %% [markdown]
# ## Model Creation from scratch:

# %% [markdown]
# To create the Multinomial model from scratch, we implemented the following formula: 
# $$
# P(y \mid x) = \frac{P(x \mid y)P(y)}{P(x)}
# $$
# We also want to find the class $c$ that maximizes $P(c \mid x)$, so we can use the following equation:
# 
# $$
# \hat{c} = \underset{c}{\text{argmax}} \ P(c) \prod_{i=1}^{n} P(x_i \mid c)
# $$
# 
# 1. Fit: To estimate probability $P(c)$, we count the number of occurences of each class in our training data. To find $P(x_i \mid c)$, we count the number of times the $i^{\text{th}}$ word in our vocabulary appears in sentences of a given class, and dividing by the total number of words in sentences of that class.
# 
# 2. Predict: We then predict the class of a given vector/sentance, based on the formula: $\hat{c} = \underset{c}{\text{argmax}} \ \log P(c) + \sum_{i=1}^{n} \log P(x_i \mid c)$. The reason we apply logarithms to the is to $\hat{c}$ is to translate the produnct into a sum and avoid underflow errors.

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
# ## Testing the Model:

# %% [markdown]
# Finally the model is trained using our training Bag of Words variable and labels, we then use our testing Bag of Words to generate predicted labels for each vector in the testing dataset. These predicted values are then compared with the actual labels of the testing dataset, using sklearn metrics, to ensure that our model sucessfully ran without any errors, or biases of any sort.

# %%
mnb = MultinomialNaiveBayes()
mnb.fit(X_train_bow, Y_train)

y_test_pred = mnb.predict(X_test_bow)

accuracy = accuracy_score(Y_test, y_test_pred)
precision = precision_score(Y_test, y_test_pred, average='weighted')
recall = recall_score(Y_test, y_test_pred, average='weighted')
f1 = f1_score(Y_test, y_test_pred, average='weighted')
conf_matrix = confusion_matrix(Y_test, y_test_pred)

print("Bernoulli Naive Bayes Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# %% [markdown]
# Based on the output, we can see that we have achieved a very high accuracy of 95.6%. Along with this we observe a 95.63% and 95.61% precision and recall values, respectively, which in turn leads to a high F1 Score of 95.6%, this means that the model generalizes well across different classes and doesn't show bias. This coupled with the high Accuracy means that we can safely say that the Multinomial Naive Bayes model was implemented sucessfully.


