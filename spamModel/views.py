from django.shortcuts import render
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#############################
import pickle


def home(request):
    return render(request, "home.html")


def predict(request):
    # dataset = pd.read_csv(
    #     'E:\sem6\ML\django\spamdetection\spam.csv', encoding="ISO-8859-1")

    # dataset.drop(dataset.iloc[:, 2:], inplace=True, axis=1)

    # dataset = dataset.iloc[:, [1, 0]]

    # # for encoding labelled data set
    # le = LabelEncoder()
    # dataset['v1'] = le.fit_transform(dataset['v1'])

    # nltk.download('stopwords')
    # corpus = []
    # for i in range(0, dataset.shape[0]):
    #     review = re.sub('[^a-zA-Z]', ' ', dataset['v2'][i])
    #     review = review.lower()
    #     review = review.split()
    #     ps = PorterStemmer()
    #     all_stopwords = stopwords.words('english')
    #     all_stopwords.remove('not')
    #     review = [ps.stem(word)
    #               for word in review if not word in set(all_stopwords)]
    #     review = ' '.join(review)
    #     corpus.append(review)

    # cv = CountVectorizer()
    # X = cv.fit_transform(corpus).toarray()
    # y = dataset.iloc[:, -1].values

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.20, random_state=0)

    # classifier = GaussianNB()
    # classifier.fit(X_train, y_train)

    # new_review = request.GET['message']
    # # file = request.GET['file']
    # # if new_review == 'are you coming with me ':
    # #     checkspam = 'not spam'

    # new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    # new_review = new_review.lower()
    # new_review = new_review.split()
    # ps = PorterStemmer()
    # all_stopwords = stopwords.words('english')
    # all_stopwords.remove('not')
    # new_review = [ps.stem(word)
    #               for word in new_review if not word in set(all_stopwords)]
    # new_review = ' '.join(new_review)
    # new_corpus = [new_review]
    # new_X_test = cv.transform(new_corpus).toarray()

    # new_y_pred = classifier.predict(new_X_test)

    # if (new_y_pred == 1):
    #     checkspam = "spam"

    # else:
    #     checkspam = "not spam"

    ##############################################################
    # new test
    cv = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))

    import re
    import nltk

    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    def preprocess(dataset):

        corpus = []
        # for i in range(0, dataset.shape[0]):
        review = re.sub('[^a-zA-Z]', ' ', dataset)
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word)
                  for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        # corpus.append(review)

        return review

    # get input from website
    text = request.GET['message']

    # preprocess the input
    transformed_sms = preprocess(text)

    vector_input = cv.transform([transformed_sms])
    vector_input = vector_input.toarray()
    result = model.predict(vector_input)[0]
    if (result == 1):
        checkspam = "spam"
    else:
        checkspam = "not spam"

    return render(request, "home.html", {"result": checkspam})
