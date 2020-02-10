from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

class NbModel(object):
    def __init__(self, tweets_arr, tweets_class, my_tokenizer):
        self.model = MultinomialNB(alpha=0.01)
        self.count_vectorizer = {}
        self.tweets_class = tweets_class
        self.tweets_arr = tweets_arr
        self.my_tokenizer = my_tokenizer


    def train_model(self):
        self.count_vectorizer = CountVectorizer(tokenizer=self.my_tokenizer)
        x_train, x_test, y_train, y_test = train_test_split(self.tweets_arr, self.tweets_class, test_size=0.2)

        self.count_vectorizer.fit(x_train)
        x_train = self.count_vectorizer.transform(x_train)
        x_test = self.count_vectorizer.transform(x_test)

        self.model.fit(x_train, y_train)

        predictions = self.model.predict(x_test)

        print(accuracy_score(y_test, predictions))
        print(classification_report(y_test, predictions))

    def predict_for_string(self, str):
        arr = [str]
        text = self.count_vectorizer.transform(arr)
        result = self.model.predict(text)
        if(result[0]==0):
            print("the result for the tweet '%s' is 0 (non-hate) message" %str)
        else:
            print("the result for the tweet '%s' is 1 (hate) message" %str)

        print(self.model.predict_proba(text))
        print()
