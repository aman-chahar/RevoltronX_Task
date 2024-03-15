# RevoltronX_Task
Similarity Search and classification of Global News Dataset
A Comprehensive Collection of more than 1 Million News Articles

1. **Data Acquisition**: You uploaded a dataset from your local system to Google Colab using the `files.upload()` function.

2. **Preparing Kaggle Credentials**: You created a directory `.kaggle` and copied your Kaggle API key to it for accessing datasets from Kaggle.

3. **Downloading Dataset**: Utilizing the Kaggle API, you downloaded the "Global News Dataset" from Kaggle using the `!kaggle datasets download` command.

4. **Extracting Dataset**: You extracted the compressed dataset using Python's `ZipFile` module.

5. **Importing Libraries**: You imported necessary libraries including pandas, numpy, re, spacy, string, nltk, tqdm, and sklearn for data manipulation, natural language processing, and machine learning tasks.

6. **Data Loading and Initial Exploration**: You loaded the dataset into a Pandas DataFrame and inspected the first few rows using `.head()` method. Additionally, you checked the information about the dataset using `.info()` method.

7. **Data Visualization**: You visualized the distribution of article sources using Plotly Express's `bar()` function.

8. **Data Cleaning**: You dropped non-useful columns and duplicate titles from the dataset.

9. **Balancing Dataset**: You balanced the dataset by selecting a subset of rows for each sentiment category to avoid class imbalance.

10. **Semantic Search using Transformers**: You installed and used `faiss-cpu` and `sentence-transformers` to perform semantic search on the dataset using pre-trained Transformer models.

11. **Text Preprocessing**: You tokenized, lemmatized, removed stop words and punctuations from the title text using spaCy and NLTK libraries.

12. **Sentiment Label Mapping**: You mapped sentiment labels (Negative, Neutral, Positive) to numerical values for classification.

13. **Text Representation**: You transformed text data into numerical representations using TF-IDF vectorization.

14. **Feature Selection**: You divided the data into features (x) and labels (y) and split it into training and testing sets.

15. **Model Training & Testing**: You trained a Logistic Regression model on the training data and evaluated its performance on the testing data.

16. **Prediction**: You made predictions on new text data and printed the predicted sentiment.

**Conclusion:**
In this project, I demonstrated end-to-end text data processing and sentiment analysis using Python. Starting from acquiring the dataset to model training and testing. The dataset was cleaned, visualized, and balanced to ensure reliable analysis. Semantic search using Transformers and TF-IDF vectorization aided in understanding the textual data better. The trained Logistic Regression model achieved satisfactory accuracy of 76% in predicting sentiment labels for news titles. This project showcases the power of Natural Language Processing (NLP) and Machine Learning (ML) techniques in analyzing textual data, offering valuable insights for decision-making in various domains.
