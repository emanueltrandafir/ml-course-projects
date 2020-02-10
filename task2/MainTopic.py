import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np

class PostsAnalyzer:

    def __init__(self):

        self.STOP_WORDS = ["lt", "gt", "li", "ul", "ol", "div", "pre", "code", "td", "in"]  # to remove the html tags from the posts
        self.TOPIC_THRESHOLD_PERCENT = 50
        self.NGRAM_RANGE = (1, 3)

        self.CSV_FILE_PATH = "../data/stackoverflow_big.csv"
        self.TITLE_COL = "title"
        self.POST_COL = "body"
        self.TAGS_COL = "tags"

        self.title_arr = []
        self.posts_arr = []
        self.tags_arr = []

        self.tfidf_matrix = {}


    def startJob(self):
        self.readData()

        docs = []
        for (post, title, tags) in zip(self.posts_arr, self.title_arr, self.tags_arr):
            doc = str(title) + "\n" + str(post) + "\n" + str(tags)
            docs.append(doc)

        tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words=self.STOP_WORDS, ngram_range=self.NGRAM_RANGE)
        tfidf_vectorizer.fit(docs)

        self.tfidf_matrix = tfidf_vectorizer.transform(docs).toarray()
        self.features = tfidf_vectorizer.get_feature_names()


    def get_topic_for_post_by_index(self, post_index):

        post_scores = self.tfidf_matrix[post_index]

        topic_score = np.amax(post_scores)


        if topic_score*100 > self.TOPIC_THRESHOLD_PERCENT:
            topic_index = (np.where(post_scores == topic_score))[0][0]
            topic_label = self.features[topic_index]

            title = self.title_arr[post_index]
            tags = self.tags_arr[post_index]
            post = self.posts_arr[post_index]
            print("index: %s; \ntopic: %s ; \nscore: %s; \npost: %s\ntitle: %s\ntags: %s \n\n * * * * * * * * * * * \n"
                  % (post_index, topic_label, str(topic_score), post, title, tags))
        else:
            # print("index: %s; \nno topic above thershold \n\n * * * * * * * * * * * \n" % post_index)
            pass
    def readData(self):
        data_frame = pd.read_csv(self.CSV_FILE_PATH, ",")
        self.title_arr = data_frame[self.TITLE_COL].values
        self.posts_arr = data_frame[self.POST_COL].values
        self.tags_arr = data_frame[self.TAGS_COL].values

if __name__ == "__main__":

    model = PostsAnalyzer()
    model.startJob()

    for i in range(1, 4000):
        model.get_topic_for_post_by_index(i)