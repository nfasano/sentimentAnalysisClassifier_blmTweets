# Sentiment Classifier of BLM Movement Tweets

### Summary of results
Eight binary classifier models were built to predict whether the sentiment of a tweet was positive or negative toward the Black Lives Matter (BLM) movement. It was found that no model performed significantly better than naively assigning all tweets a positive label (which would yield an accuracy of 81%). Logistic regression achieved the best performance on the held out test set with an accuracy of 83% and AUC of 0.79, but it was not substantially better than SVM (accuracy = 83%, auc = 0.75) or KNN (accuracy = 83%, AUC = 0.74). Naive Bayes had a low accuracy of 73%, but the highest precision at 92%, suggesting that an ensemble learning model may improve prediction accuracy. Note that chatGPT had an accuracy of 76% when the testing tweets were fed into a chatGPT prompt without any preprocessing.

### Introduction
The summer of 2020 saw the rise of support for #BlackLivesMatter after the murder of George Floyd, with in-person protests supporting the #BLM movement being held during the coronavirus epidemic in every major US city. In addition to in-prson protest, social media sites — including Twitter — were also awash with opinions about the movement. When Twitter users read tweets about the #BLM movement, and the contemporaneous #BlueLivesMatter movement in support of policing efforts, they might be interested in categorizing tweets based on their attitudes toward the #BLM and #BlueLivesMatter movements. In this work, we use a dataset of 10,000+ labelled tweets pertaining to the #BLM movement to build a sentiment classifier that can predict whether a tweet was positive or negative toward the #BLM. After pre-processing of the dataset and exploratory data analysis which yielded insights on how to prune vocabulary (see details below), eight classifier models were built and comapared against one another (see comparison criteria below), including KNN, Multinomial Naive Bayes, Log. Reg., SVM (Lin), SVM (rbf), Complement Naive Bayes, LDA, and Random Forest. 5-fold cross validation was used for all hyperparameter selection.

### Dataset Overview 
|   DataSet   | Num. Negative | Num. Positive | Num. Neither | Totals   | 
|-------------|---------------|---------------|--------------|----------|
|  Train.csv  |    1347       |    5891       |   871        |   8109   |  
|  Test.csv   |    337        |    1474       |   218        |   2029   |  
|   Totals    |    1684       |    7365       |   1089       |   10138  |  

#### Train/Test csv column headers
  - created_at: the date of the tweet
  - hashtags: the hashtags used in the tweet
  - text: text of the tweet
  - BLM: label for the tweet (positive, negative, or neither toward the BLM movement)
     - Refer to the 'tweetLabelCriteria.pdf' document for details on how labels were constructed
     
### Data cleaning and feature extraction
#### Preprocessing of training and testing data
The following preprocessing steps were taken for each tweet in the training and testing dataset (see preprocessing_text(tweet) function in classifier.ipynb notebook)
  - Change emojis to words (import emoji)
  - Remove 'RT' from tweet
  - Remove capitalization
  - Remove urls & user mentions from tweet
  - Remove punctuation
  - Remove stopwords (from nltk.corpus import stopwords)
  - Perfom lemmatization on tokens (from nltk.stem import WordNetLemmatizer)
  - Removed Tweets that were labelled as neither

### Exploratory Data Analysis
The following figure presents word clouds for negatively and positively labelled tweets. The size of the word indicates the total amount of times the word appeared in the entire dataset with that label. Prior to making these word clouds, the dataset was cleaned of common stop words and the phrase blacklivesmatter, which was the most frequent word in both datasets.

In this figure, we observe that some words are frequently used in both positive and negative tweets such as the word "black". Other words are only commonly found in one of the two word clouds, such as the words "EricGarner" and "ICantBreath" which are commonly found in positively labelled tweets but not in negatively labelled tweets. Similarly, the word "BlueLivesMatter" is commonly found in negatively labelled tweets but not positively labelled tweets.

<p align="center">
<picture>
<img src="https://github.com/nfasano/sentimentClassifier_blmTweets/blob/main/images/wordCloud.jpg" alt="drawing" width="600"/> 
</picture>
</p>
Figure: Word clouds for negatively and positively labelled tweets.

Building on the observation that some words are more suggesstive of a positive or negative label, we constructed histograms for all words in the data set and compared them against the . The main idea here is that if a single word's histogram significantly defers from the training data sets histogram, then it may be a strong predictor of the tweet label. The following figure plots a few of these histograms against the histogram of the entire dataset. Here we see that some words, such as "make" which occurred in 183 tweets, has an identical histogram to the training dataset and so will not likely be a strong predicitve word. Other words, such as "alllivesmatter" and "justice" which occurred in 611 and 190 tweets, respectively, have histograms which strongly deviate from that of the training datasets histogram, indicating that they are strong predictors of a tweet being negative or positive.

<p align="center">
<picture>
<img src="https://github.com/nfasano/sentimentClassifier_blmTweets/blob/main/images/histograms.jpg" alt="drawing" width="850"/> 
</picture>
</p>
Figure: Histograms of (a) the fraction of tweets labeled as positive or negative towards the BLM movement and (b-d) the fraction of tweets the words "make", "alllivesmatter", and "justice" appeared in a negative or positive tweet compared only to the tweets in which that word appeared. The dashed red lines indicate the expected value if the histogram for that word followed the training dataset histogram. The title contains the word used in the histogram and the number of tweets that that word appeared in.

### Forming the dataset for classification - extracting features and pruning the vocabulary
The training dataset of tweets was transformed using a bag of words representation, where the features (columns in the dataframe) are all the unique words in the training dataset and each instance (rows of the dataframe) is one of the tweets from the dataset. Each entry of the dataframe then contains the number of times a particular word occurred in a particular tweet. 

In addition to the bag of words features, we also calculated the number of words and characters in each tweet and added them as features. This is helpful for removing very short tweets.

To prune the vocabulary, we used the insights obtained from the exploratory data analysis to remove any words that did not strongly deviate from the training dataset's histogram. This analysis is provided in section IV of the classifier.ipynb Python notebook.

Formaly, we performed a hypothesis test comparing the histogram of each word in the vocabulary against the histogram of entire training dataset. The null hypothesis is that the histogram of the word is identical to the histogram of the training dataset, i.e. the word appears in 81.4% of positive tweets and 18.6% negative tweets. To compare the word's histogram against the training dataset histogram, we used a binomial test to compute a p-value and rejected the null hypothesis if the p-vlaue was less than 0.05. Note that a standard z-test or t-test would not be appropriate here, especially for infrequent words and words with histograms that did not substantially deviate from the null hypothesis.

Finally, there is an option in the code to remove tweets with too few words/characters (e.g. one could drop any tweets with less than 3 words or 10 characters). In what follows, I only removed a tweet from the dataset if it had 0 words after pruning the vocabulary. 

### Comparison of classifiers
Eight models were trained on the training dataset: KNN, Multinomial Naive Bayes, Log. Reg., SVM (Lin), SVM (rbf), Complement Naive Bayes, LDA, and Random Forest.
5-Fold cross validation was used to determine hyperparameters at train time. These models were then used to predict labels for tweets from the held out testing set. The following table summarizes the accuracy, precision, recall, F1-score, confusion matrix, and area under curve (AUC) of all eight classifiers. 

|   Classifier   | Accuracy | Precision | Recall | F1-score |   TN   |   FP   |   FN   |   TP   |   AUC  |
|----------------|----------|-----------|--------|----------|--------|--------|--------|--------|--------|
|       KNN      |    0.83  |    0.85   |   0.97 |   0.90   |    76  |   261  |    44  |  1430  |  0.74  |
| M. Naive Bayes |    0.82  |    0.84   |   0.97 |   0.90   |    59  |   278  |    47  |  1427  |  0.78  |
|    Log. Reg.   |    0.83  |    0.85   |   0.96 |   0.90   |    90  |   247  |    66  |  1408  |  0.79  |
|    SVM (Lin)   |    0.82  |    0.83   |   0.98 |   0.90   |    45  |   292  |    26  |  1448  |  0.77  |
|    SVM (rbf)   |    0.83  |    0.85   |   0.96 |   0.90   |    84  |   253  |    58  |  1416  |  0.75  |
| C. Naive Bayes |    0.73  |    0.92   |   0.73 |   0.82   |   243  |    94  |   395  |  1079  |  0.78  | 
|       LDA      |    0.82  |    0.86   |   0.94 |   0.90   |   109  |   228  |    94  |  1380  |  0.78  |
|  Random Forest |    0.80  |    0.85   |   0.91 |   0.88   |   103  |   234  |   127  |  1347  |  0.72  |

Here we see that no model performed significantly better than naively assigning all tweets a positive label (which would yield an accuracy of 81%). Logistic regression achieved the best performance on the held out test set with an accuracy of 83% and AUC of 0.79, but it was not substantially better than SVM (accuracy = 83%, auc = 0.75) or KNN (accuracy = 83%, AUC = 0.74). Naive Bayes had a low accuracy of 73%, but the highest precision at 92%, suggesting that an ensemble learning model may improve prediction accuracy.

<p align="center">
<picture>
<img src="https://github.com/nfasano/sentimentClassifier_blmTweets/blob/main/images/roc_curve.jpg" alt="drawing" width="750"/> 
</picture>
</p>
Figure: Precision-recall curve for all eight trained classifiers.

Note that chatGPT had an accuracy of 76% when the testing tweets were fed into a chatGPT prompt without any preprocessing.

### Future Work
Build ensemble classifiers, bi-gram, and tf-idf instead of bag of words.
