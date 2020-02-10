from wordcloud import WordCloud
import matplotlib.pyplot as plt

class DataAnalyzer(object):

    def __init__(self, tweets_arr, tweet_class, my_tokenizer):
        self.tweets_arr = tweets_arr
        self.tweets_class = tweet_class
        self.my_tokenizer = my_tokenizer

    def display_data_for_class(self, class_nr):
        props = self.filter_comments(self.tweets_arr, self.tweets_class, class_nr)
        all_words = ""

        for prop in props:
            words_in_this_prop = " ".join(self.my_tokenizer(prop))
            all_words += " " + words_in_this_prop

        self.display_wordcloud_image(all_words)

    def filter_comments(self, tweets_arr, tweets_classes, needed_class):
        result = []
        for text, cls in zip(tweets_arr, tweets_classes):
            if (cls != needed_class):
                continue
            result.append(text)
        return result

    def display_wordcloud_image(self, words):
        cloud = WordCloud(width=1200, height=900).generate(words)
        plt.figure(figsize=(20, 20))
        plt.imshow(cloud)
        plt.axis('off')
        plt.show()