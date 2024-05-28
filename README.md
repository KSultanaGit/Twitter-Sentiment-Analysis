# Twitter Sentiment Analysis
It seems like I can’t do more advanced data analysis right now. Please try again later. However, you can manually download and extract the code snippets from your notebook to include in the README file. 

You can follow the steps below to create the README file based on the content of your notebook:

# Twitter Sentiment Analysis

## Introduction
This project performs sentiment analysis on tweets to understand the overall sentiment (positive, negative, or neutral) of Twitter users. The goal is to analyze the sentiment trends and provide insights into public opinion on various topics.

## Tech Stack
- **Python**: Main programming language
- **Jupyter Notebook**: Interactive development and analysis environment
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **NLTK**: Natural Language Toolkit for text processing
- **Scikit-learn**: Machine learning library for predictive analysis

## Files Included
- [Twitter_Sentiment_Analysis.ipynb](https://github.com/KSultanaGit/Twitter-Sentiment-Analysis/blob/main/Twitter_Sentiment_Analysis.ipynb): Jupyter Notebook containing the sentiment analysis.
- [tweets.csv](https://github.com/KSultanaGit/Twitter-Sentiment-Analysis/blob/main/tweets.csv): Dataset used for the analysis.
- [plots](https://github.com/KSultanaGit/Twitter-Sentiment-Analysis/tree/main/plots): Directory containing all the generated plots.

## Overview
The project involves analyzing tweet data to determine the sentiment expressed in each tweet. The analysis includes data preprocessing, sentiment classification using machine learning models, and visualization of the sentiment distribution and trends.

## Important Code Snippets, Plots, and Their Inferences
1. **Loading Data**:
    ```python
    import pandas as pd
    data = pd.read_csv('tweets.csv')
    data.head()
    ```

2. **Data Preprocessing**:
   There are no null values.
   The positive and negative tweets are equally distributed.
   Replacing 4 with 1 in the target Column.
    ```python
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk.download('stopwords')
    nltk.download('punkt')

    stop_words = set(stopwords.words('english'))

   def stemming(content) :
    stemmed_content=re.sub('[^a-zA-Z]',' ', content)#removing non-alphabets
    stemmed_content=stemmed_content.lower()#convert to lower case
    stemmed_content=stemmed_content.split()#splitting and storing in a list
    stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content=' '.join(stemmed_content)
    
    return stemmed_content
    ```

4. **Sentiment Analysis**:
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import classification_report, accuracy_score

    splitting data to training data and test data
    X_train,X_test,Y_train,Y_test=train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    #converting textual data to numerical data
    vectorizer=TfidfVectorizer()
    X_train=vectorizer.fit_transform(X_train)
    X_test=vectorizer.transform(X_test)

    #training the model:logistic regression-classification model ie negative or positive tweet

    model=LogisticRegression(max_iter=1000)
    model.fit(X_train,Y_train)

    predictions = model.predict(X_test_vec)
    print(classification_report(y_test, predictions))
    print(f'Accuracy: {accuracy_score(y_test, predictions)}')
    #model accuracy=77.8%
    ```

### Plots
- **Model Performance**:
    ![Model Performance](https://github.com/KSultanaGit/Twitter-Sentiment-Analysis/blob/main/plots/model_performance.png)
    - **Inference**: This plot illustrates the performance metrics (precision, recall, F1-score) of the sentiment classification model.

## Summary
The sentiment analysis on Twitter data provides valuable insights into public opinion on various topics. The project demonstrates the application of natural language processing and machine learning techniques to classify sentiments in tweets. The visualizations help in understanding the sentiment distribution and the effectiveness of the classification model.

## Contributor
- **KSultanaGit**: [GitHub Profile](https://github.com/KSultanaGit)
 
