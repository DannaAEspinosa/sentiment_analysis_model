###Imports
##Collecting data
import pandas as pd

##Text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
##Regular functions
import re

##DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

#RNN
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


###Gather the dataset for sentiment analysis from the UCI Machine Learning Repository
## Read text files

file1 = pd.read_csv("sentiment_labelled_sentences/amazon_cells_labelled.txt", delimiter = '\t', quoting = 3, header=None, names=["Phrase", "tag"])
file2 = pd.read_csv("sentiment_labelled_sentences/imdb_labelled.txt", delimiter = '\t', quoting = 3, header=None, names=["Phrase", "tag"])
file3 = pd.read_csv("sentiment_labelled_sentences/yelp_labelled.txt", delimiter = '\t', quoting = 3, header=None, names=["Phrase", "tag"])

#Check the size of each read file (incomplete in file2, database source error)
print("Number of rows in file1:", len(file1))
print("Number of rows in file2:", len(file2))
print(file2.tail())

print("Number of rows in file3:", len(file3))

df = file2
print(df)

# Concatenate the three files
combined_df = pd.concat([file1, file2, file3], ignore_index=True)

print(combined_df.columns)
print(combined_df.shape)
print(combined_df['tag'].value_counts() / combined_df['tag'].shape[0])


#Print initial combined_df
print(combined_df)

###Preprocess the text data, including tokenization, lowercasing, and removing stopwords.
# Function to clean text (regular expressions)
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters and punctuation
    return text

# Apply cleaning and tokenization for English words
combined_df['Cleaned_Phrase'] = combined_df['Phrase'].apply(clean_text)
combined_df['Tokenized_Phrase'] = combined_df['Cleaned_Phrase'].apply(word_tokenize)
combined_df['Tokenized_Phrase'] = combined_df['Tokenized_Phrase'].apply(lambda x: [word.lower() for word in x])

# Download and remove English stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
combined_df['Tokenized_Phrase'] = combined_df['Tokenized_Phrase'].apply(lambda x: [word for word in x if word not in stop_words])

# Print the dataframe with original phrases and preprocessed English words
print(combined_df[['Phrase', 'Tokenized_Phrase']])

##Implement a DummyClassifier.

# Assume 'X' are the preprocessed features and 'y' are the labels
X = combined_df['Tokenized_Phrase']
y = combined_df['tag']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##Create a DummyClassifier
# Strategy: predict the most frequent class
dummy_clf = DummyClassifier(strategy="stratified")  

#Train the model
dummy_clf.fit(X_train, y_train)

#Make predictions on the test set
y_pred = dummy_clf.predict(X_test)

##Calculate evaluation metrics (Test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)

#Show results of test set metrics
print('\n'+'Metrics for test set'+'\n')
print(f"Accuracy Test: {accuracy}")
print(f"Precision Test: {precision}")
print(f"Recall Test: {recall}")
print(f"F1 Score Test: {f1}")
print(f"Kappa Test: {kappa}")

##Calculate evaluation metrics (Train)
y_pred_train = dummy_clf.predict(X_train)

#Calculate evaluation metrics (Train)
accuracy = accuracy_score(y_train, y_pred_train)
precision = precision_score(y_train, y_pred_train)
recall = recall_score(y_train, y_pred_train)
f1 = f1_score(y_train, y_pred_train)
kappa = cohen_kappa_score(y_train, y_pred_train)

#Show results of train set metrics
print('\n'+'Metrics for train set'+'\n')
print(f"Accuracy Train: {accuracy}")
print(f"Precision Train: {precision}")
print(f"Recall Train: {recall}")
print(f"F1 Score Train: {f1}")
print(f"Kappa Train: {kappa}")

###Implement a vanilla RNN sentiment analysis model. 

##Tokenize text
# Maximum number of words to consider
max_words = 10000

# Create a Tokenizer with a specified maximum number of words
tokenizer = Tokenizer(num_words=max_words)

# Fit the Tokenizer on the training text data
tokenizer.fit_on_texts(X_train)

# Convert the training and test text data into sequences of indices
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Adjust the length of the sequences to a fixed size
maxlen = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

##Define the architecture of the model:
def create_rnn_model(units=50):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=50, input_length=maxlen))
    model.add(SimpleRNN(units, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

rnn_model = KerasClassifier(build_fn=create_rnn_model, epochs=10, batch_size=32, verbose=0)

param_grid = {'units': [50, 100, 150], 'batch_size': [32, 64, 128]}
grid_search = GridSearchCV(estimator=rnn_model, param_grid=param_grid, scoring='accuracy', cv=3)

grid_result = grid_search.fit(X_train_pad, y_train)