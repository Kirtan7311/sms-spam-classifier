# SMS Spam Classifier

This project is an SMS spam classifier that uses machine learning techniques to detect spam messages. The classifier is trained on a dataset of SMS messages and can predict whether an SMS is spam or not. The web application is built using Streamlit, allowing users to input a message and get a prediction in real-time.

## Features
- Uses Multinomial Naive Bayes for classification.
- Preprocessing steps include tokenization, removing stopwords and punctuation, and stemming.
- Achieves high accuracy, precision, and recall on the test dataset.

## Installation
To run this project, you need to have Python and the following libraries installed:
```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/spam-classifier.git
    ```

2. Navigate to the project directory:
    ```bash
    cd spam-classifier
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:
    ```bash
    streamlit run app.py
    ```

5. Open your web browser and go to:
    ```
    http://localhost:8501
    ```

6. Enter a message into the input box to classify it as "Spam" or "Not Spam".

## Files

- **app.py**: Main application script built with Streamlit. This script handles the web interface and prediction logic.
- **model.pkl**: Trained machine learning model saved as a pickle file. This model is used to make predictions.
- **vectorizer.pkl**: Vectorizer used for transforming text data into numerical format, saved as a pickle file.
- **spam.csv**: Dataset used for training and testing the model. This CSV file contains labeled SMS messages.
- **spam-classifier.ipynb**: Jupyter Notebook with the project development, including data preprocessing, exploratory data analysis (EDA), model training, and evaluation.
- **requirements.txt**: List of dependencies required for the project. This file ensures that all necessary libraries are installed.
- **.gitignore**: File specifying which files and directories to ignore in version control, such as virtual environments and cache files.

## Dataset

The dataset used for training and testing the SMS spam classifier is included in the repository as `spam.csv`. This dataset contains labeled SMS messages, which are classified as either "ham" (not spam) or "spam".

## Model Training

### 1. Reading Data
The dataset is read from the `spam.csv` file and the relevant columns are renamed for clarity.

```python
import pandas as pd

# Read the dataset
df = pd.read_csv("spam.csv", usecols=['v1', 'v2'])
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
```
### 2. Data Cleaning
The data cleaning process involves encoding the target labels and removing duplicate entries from the dataset.

1. **Encoding Target Labels**:
   - The target labels ("ham" and "spam") are encoded as numerical values using `LabelEncoder`.

2. **Removing Duplicates**:
   - Duplicate entries in the dataset are identified and removed to ensure the quality of the data.

```python
from sklearn.preprocessing import LabelEncoder

# Encode the target labels
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Remove duplicate entries
df.drop_duplicates(inplace=True)
```
### 3. Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA) is performed to understand the distribution of the data and key characteristics of the dataset. The following steps are taken during EDA:

1. **Distribution of Target Labels**:
   - A pie chart is created to visualize the distribution of "ham" and "spam" messages in the dataset.

2. **Feature Extraction**:
   - Additional features are extracted from the text data, including the number of characters, words, and sentences in each message.

3. **Visualizing Feature Distributions**:
   - Histograms are created to visualize the distributions of the number of characters, words, and sentences for "ham" and "spam" messages.
   - A heatmap is generated to show the correlation between the target label and the extracted features.

```python
import matplotlib.pyplot as plt
import nltk

# Pie chart of target labels
plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct='%0.2f')

# Extract additional features
df['num_characters'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sent'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

# Visualize feature distributions
import seaborn as sns
sns.histplot(df[df['target'] == 0]['num_characters'], kde=True, label='Ham')
sns.histplot(df[df['target'] == 1]['num_characters'], kde=True, color='red', label='Spam')
plt.legend()

# Heatmap of feature correlations
sns.heatmap(df[['target', 'num_characters', 'num_words', 'num_sent']].corr(), annot=True)
```
### 4. Data Preprocessing
The data preprocessing steps involve transforming the text data to make it suitable for machine learning. The following steps are taken:

1. **Lowercasing**:
   - Convert all characters in the text to lowercase to ensure uniformity.

2. **Tokenization**:
   - Split the text into individual words (tokens) using NLTK's `word_tokenize` function.

3. **Removing Non-Alphanumeric Characters**:
   - Remove any non-alphanumeric characters from the text to focus on meaningful words.

4. **Removing Stopwords and Punctuation**:
   - Remove common stopwords (e.g., "the", "is") and punctuation marks using NLTK's stopwords and Python's `string.punctuation`.

5. **Stemming**:
   - Reduce words to their root forms using NLTK's `PorterStemmer` to reduce the vocabulary size and handle different forms of the same word.

```python
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

# Initialize the PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text
    text = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]
    
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    
    # Apply stemming
    text = [ps.stem(word) for word in text]
    
    # Join the processed words back into a single string
    return " ".join(text)

# Apply the transformation to the text column in the DataFrame
df['transformed_text'] = df['text'].apply(transform_text)
```
### 5. Model Building
The model building process involves vectorizing the text data and training a machine learning model to classify SMS messages as either "ham" or "spam". The following steps are taken:

1. **Text Vectorization**:
   - The text data is transformed into numerical format using the TF-IDF vectorizer, which converts the text into a matrix of TF-IDF features.

2. **Splitting the Data**:
   - The dataset is split into training and testing sets to evaluate the model's performance.

3. **Training the Model**:
   - A Multinomial Naive Bayes classifier is trained on the training data.

4. **Evaluating the Model**:
   - The trained model is evaluated on the test set to determine its accuracy and precision.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split

# Vectorize the text data using TF-IDF
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train the Multinomial Naive Bayes model
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mnb.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
```
### 6. Saving the Model
After training the machine learning model, we save both the trained model and the TF-IDF vectorizer as pickle files. This allows us to easily load and use them later in our Streamlit application for making predictions.

1. **Save the TF-IDF Vectorizer**:
   - The `TfidfVectorizer` object is saved as a pickle file named `vectorizer.pkl`.

2. **Save the Trained Model**:
   - The trained Multinomial Naive Bayes model is saved as a pickle file named `model.pkl`.

```python
import pickle

# Save the TF-IDF vectorizer
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))

# Save the trained model
pickle.dump(mnb, open('model.pkl', 'wb'))
```
### Results
The trained SMS spam classifier model was evaluated on a test dataset to determine its performance. The following metrics were used to assess the model's effectiveness:

1. **Accuracy**:
   - The accuracy of the model indicates the percentage of correctly classified messages out of the total messages in the test dataset.

2. **Precision**:
   - Precision measures the proportion of true positive predictions (correctly identified spam messages) out of all positive predictions made by the model.

The model achieved the following results on the test dataset:

- **Accuracy**: 97.1%
- **Precision**: 100%

These results demonstrate that the SMS spam classifier is capable of accurately identifying spam messages, with a high level of precision. The model's performance can be further improved by tuning hyperparameters, using more advanced machine learning algorithms, or incorporating additional features into the dataset.

## Contributing
Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

Please make sure your code follows the project's coding style and includes relevant tests. We appreciate your contributions and will review pull requests promptly.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments
- The dataset used for this project is publicly available from [source/link].
- This project uses several open-source libraries and tools, including:
  - [pandas](https://pandas.pydata.org/)
  - [scikit-learn](https://scikit-learn.org/)
  - [nltk](https://www.nltk.org/)
  - [streamlit](https://streamlit.io/)

We would like to thank the contributors and maintainers of these libraries for their valuable work.

