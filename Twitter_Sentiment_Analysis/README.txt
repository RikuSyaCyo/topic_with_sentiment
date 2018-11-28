Twitter Sentiment Analysis

*Dataset:

code/dataset/train.csv:20000 Tweets with sentiment polarity label, 10000 positive, 10000 negative
code/dataset/*.csv:Tweets from 10 American states.
data/positive-words.txt: List of positive words.(UCI)
data/negative-words.txt: List of negative words.(UCI)

*Requirements:

Python 3.7.0
numpy 1.15.2
pandas 0.23.4
scikit-learn

*Anaconda and spider inside are friendly suggested.

*How to run:
1. Run "preprocess.py" to process training and testing dataset. In our work, the all the data in the folder have been processed. You can skip this step.
2. Run "stat.py" to get the statistic information of dataset.
3. Run "classcifier.py":
    (1) You can set the defined constant bool variable "TRAIN" as True to train training dataset and then as False to test testing dataset.
    (2) You can set the defined constant bool variable "USE_BIGRAM" as True or False to determine whether use the bigram feature in the algorithm.