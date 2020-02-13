from nltk import TweetTokenizer

from task1.Analyze import DataAnalyzer
from task2.MainTopic import PostsAnalyzer

if __name__ == "__main__":

    stackOverflowPostsModel = PostsAnalyzer()
    posts = stackOverflowPostsModel.posts_arr
    simulatedClass = [0] * len(posts)
    tokenizer = TweetTokenizer()

    analyzer = DataAnalyzer(posts, simulatedClass, tokenizer)
    analyzer.display_data_for_class(posts)



