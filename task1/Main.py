import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from task1.Analyze import DataAnalyzer
from task1.NbModel import NbModel

TRAIN_MODEL = False
ANALYZE_DATA = True

USE_LEMMATIZATION = True
USE_STOP_WORDS = True
CUSTOM_STOP_WORDS = ["user", "@user", "new", "much"]

TWEETS_COL = "tweet"
LABELS_COL = "label"
CSV_FILE_PATH = "../data/tweets.csv"


def main():
    data_frame = pd.read_csv(CSV_FILE_PATH, ",")
    tweets_arr = data_frame[TWEETS_COL].values
    tweet_class = data_frame[LABELS_COL].values

    if(TRAIN_MODEL):
        nb_model = NbModel(tweets_arr, tweet_class, my_tokenizer)
        nb_model.train_model()
        print()
        nb_model.predict_for_string("soccer players have low stamina")
        custom_tokenizer("soccer players have low stamina", True, True, is_debug=True)

        print()
        nb_model.predict_for_string("Obama is a terrible player")
        nb_model.predict_for_string("Emanuel is a terrible player")

    if(ANALYZE_DATA):
        data_analyzer = DataAnalyzer(tweets_arr, tweet_class, my_tokenizer)
        data_analyzer.display_data_for_class(0)
        data_analyzer.display_data_for_class(1)

def my_tokenizer(text):
    return custom_tokenizer(text, USE_STOP_WORDS, USE_LEMMATIZATION, is_debug=False)

def custom_tokenizer(text, use_stop_words, use_lemma, **kwargs):
    tw_tokenizer = TweetTokenizer()
    tokens = tw_tokenizer.tokenize(text)
    is_debug = kwargs.get('is_debug', False)
    if (is_debug):
        print("initial tokens: ", tokens)

    if (use_stop_words):
        # nltk.download('stopwords')
        stop_words = stopwords.words('english')
        stop_words.extend(CUSTOM_STOP_WORDS)
        tokens = [token for token in tokens if token not in stop_words]
        if (is_debug):
            print("without stop words: ", tokens)

    if (use_lemma):
        # nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = []
        for token in tokens:
            lemma = lemmatizer.lemmatize(token)
            lemmatized_tokens.append(lemma)
        tokens = lemmatized_tokens
        if (is_debug):
            print("after lemmatization: ", tokens)

    return tokens



if __name__== "__main__":
  main()
