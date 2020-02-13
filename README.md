# Task 1 - Twitter sentiment alanysis

For the first task we had to predict if a certain post from Twitter was harmful based on the previously reported tweets.

In the Main File there is the main funcion that can either tran the model and do some simple predictions or analyze the data and display the most commonly used words for each of the classes *(reported and not-reported  tweets)*.

To switch between the 2 functionalties, I'm using the two boolean flags TRAIN_MODEL and ANALYZE_DATA.

There are also some other parameters that can be set to False for a faster training: USE_LEMMATIZATION or USE_STOP_WORDS.

As tokenizer I have use my own implementation which based on TweeterTokenizer, with the addition of some extra features.

For the classification I have used a simple NaiveBayes Classifier and for the data analysis I have craeted a WordCloud representation of the main tokens contained for each of the classes.

# Task 2 - StackOverflow questions

In this project I had to train a model that could determine the main topic of a question posted on StackOverflow.

I have used an TfidfVectorizer from sklearn to find the topic of each document.
As document I have used the string that resulted after concatenating the title of the question, the body and the tags.

At first, the topics generated were variable names from code snippets (from the questions) and html tags that were mentioned 4-5 times in that document (post) and they were never mentioned in any other post, and the TfidfVectorizer was returing a big value for them.

After a few more tests, I have added the *optional* min_df parameter that removes the tokens with a small number of appearances. When this parameter was graeter than 50, my model started returning topics such as "MySQL" instad of "myObj" :)

Because the text analyzer from the first task was generically enough, I was able to use it again for the second task (with some small adaptations) to generate the wordcloud representation of the most used tokens present in the StackOverflow questions and also to represent the most frecquently found topics. 

This is working for now, but it's somthing that needs to be improved.



