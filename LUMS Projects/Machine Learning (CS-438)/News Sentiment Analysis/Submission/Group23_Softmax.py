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
# The data has been split using sklearn into Training and Testing data in a 70-30 split. Validation data was not included in this model, since this is only the first model and it's implementation. In a Softmax model, having validation data doesn't make much of a difference upon training of the model, thus we will only be using training and testing.

# %%
merged_df = pd.read_csv('merged_file.csv')
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
# # Softmax text Classifier Implementation:
# ## Bag of Words:
# 
# Since we are implementing a softmax model, we will need to create a Bag of Words Function.
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
# We now create a testing and training corpus based on the test and train dataset. Once done, we initialize the BagofWords function and fit it to the training Corpus. The reason behind this is to maintain consistency when comparing outputs. The corpus of training and testing is then transformed into Bag of Words variables which will be used in the training and testing of our Model.

# %%
corpus_train = X_train['Cleaned_content']
corpus_test = X_test['Cleaned_content']

bow = BagOfWords()
bow.fit(corpus_train)

X_train_bow = bow.transform(corpus_train)
X_test_bow = bow.transform(corpus_test)

# %% [markdown]
# 
# ### SoftmaxTextClassifier
# 
# 1. The SoftmaxTextClassifier is a neural network model designed for text classification tasks. It uses a softmax activation function to predict the class probabilities of input text data. The model is trained using the gradient descent optimization algorithm, where the gradients are used to update the weights and biases of the model in the opposite direction of the gradient to minimize the loss function.
# 
# 
# 2. **Softmax Function**:
#    $$
#    \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
#    $$
# 
# 3. **Cross-Entropy Loss**:
#    $$
#    \mathcal{L} = -\frac{1}{m} \sum_{i=1}^m \log(y_{\text{pred}}[i, y_{\text{true}}[i]])
#    $$
# 
# 4. **Forward Propagation**:
#    $$
#    z = X \cdot \mathbf{W} + \mathbf{b}, \quad y_{\text{pred}} = \text{softmax}(z)
#    $$
# 
# 5. **Backward Propagation**:
#    Gradients for \(z\), \(\mathbf{W}\), and \(\mathbf{b}\):
#    $$
#    \frac{\partial \mathcal{L}}{\partial z} = \frac{y_{\text{pred}} - \mathbf{y}_{\text{true}}}{m}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{W}} = X^T \cdot \frac{\partial \mathcal{L}}{\partial z}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \sum \frac{\partial \mathcal{L}}{\partial z}
#    $$
# 
# 6. **Gradient Descent Update**:
#    $$
#    \mathbf{W} \leftarrow \mathbf{W} - \eta \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{W}}, \quad \mathbf{b} \leftarrow \mathbf{b} - \eta \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{b}}
#    $$
# 
# 7. **Prediction**:
#    $$
#    z = X \cdot \mathbf{W} + \mathbf{b}, \quad y_{\text{pred}} = \text{softmax}(z), \quad \text{label} = \text{argmax}(y_{\text{pred}})
#    $$
# 

# %%
class SoftmaxTextClassifier:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_classes = None
        self.label_to_id = None
        self.id_to_label = None
        self.W = None
        self.b = None

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true])
        return np.sum(log_likelihood) / m

    def fit(self, X, Y):
        unique_labels = list(set(Y))
        self.num_classes = len(unique_labels)
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        y = np.array([self.label_to_id[label] for label in Y])

        num_features = X.shape[1]
        self.W = np.random.randn(num_features, self.num_classes) * 0.01
        self.b = np.zeros((1, self.num_classes))

        for epoch in range(self.epochs):
            z = np.dot(X, self.W) + self.b
            y_pred = self.softmax(z)
            loss = self.cross_entropy_loss(y_pred, y)
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss}")
            m = X.shape[0]
            grad_z = y_pred
            grad_z[range(m), y] -= 1
            grad_z /= m

            dW = np.dot(X.T, grad_z)
            db = np.sum(grad_z, axis=0, keepdims=True)

            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

    def predict(self, X):
        z = np.dot(X, self.W) + self.b
        y_pred = self.softmax(z)
        y_pred_labels = np.argmax(y_pred, axis=1)
        return [self.id_to_label[label] for label in y_pred_labels]

    def evaluate(self, X, Y):
        y_test_pred = self.predict(X)
        accuracy = accuracy_score(Y_test, y_test_pred)
        precision = precision_score(Y_test, y_test_pred, average='weighted')
        recall = recall_score(Y_test, y_test_pred, average='weighted')
        f1 = f1_score(Y_test, y_test_pred, average='weighted')
        conf_matrix = confusion_matrix(Y_test, y_test_pred)

        print("Softmax Text Classifier Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)

# %% [markdown]
# ## Testing the model:

# %%
classifier = SoftmaxTextClassifier(learning_rate=0.1, epochs=100)
classifier.fit(X_train_bow, Y_train)
classifier.evaluate(X_test_bow, Y_test)

# %% [markdown]
# The model demonstrated strong performance, achieving an accuracy of 94.57%, indicating that the predictions were correct in the majority of cases. It achieved a precision of 94.64%, reflecting a high proportion of true positives among all positive predictions. The recall was 94.57%, showing the model's ability to correctly identify most of the actual positive instances. Finally, the F1 score, which balances precision and recall, was 94.59%, highlighting the model's overall effectiveness in handling the classification task


