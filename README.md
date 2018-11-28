# Spatial-Temporal Topic Recommendation of Social Network Data

The project is focusing on the sentiment analysis of hot topic on twitter and represent on the map.

# Dataset Processing

Raw data can be found in the folder "dataset", .py are used to remove some noises and outliers.


# Sentiment Analysis

 - preprocess.py is used to process training and testing dataset. 
 - stat.py is used to get the statistic information of dataset.
 -  Run: NB, logisticR, DT, SVM, NeuralNet, Ramdom ,Forest classifiers
 - [ ] Set the defined constant bool variable "TRAIN" as True to train training dataset and then as False to test testing dataset.
 - [ ] Set the defined constant bool variable "USE_BIGRAM" as True or False to determine whether use the bigram feature in the algorithm.


# Topic Extraction from Tweets

Run twitter_topic_extraction.py. 
Input files. TXcon.csv. 
Outfile. TX.csv.
 
**Notes** : 

 - Fastfluster need to be installed first.
 - CMUTweetTagger.py is given, please put these files together.


# Map Visualization

- package: python Flask, javascript D3.js

- run app.py to activate a server on your computer

- input http://127.0.0.1:5000 in web browser to see the visualization map

file:

1. templates/map.html
2. static/js: basic js tools and main function draw_map.js 
3. app.py python file of server
