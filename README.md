# Sentiment Classifier of BLM Movement Tweets

### Summary of results
Eight binary classifier models were built on a dataset of tweets to predict whethere a tweet was positive or negative toward the Black Lives Matter (BLM) movement. It was found that no model performed significantly better than a naively assigning all tweets a positive label (which would yield an accuracy of 80%). Logistic regression achieved the best performance on the held out testing with an accuracy of 83% and AUC of 0.79, but it was not substantially better than SVM (accuracy = 83%, auc = 0.75) or KNN (accuracy = 83%, AUC = 0.74). Naive Bayes had a low accuracy of 73%, but the highest precision of 92%, suggesting that an ensemble learning model may improve predcition accuracy.

### Background
The summer of 2020 saw the rise of support for #BlackLivesMatter and social justice movements after the murder of George Floyd. While in-person protests supporting the #BLM movement were held during the coronavirus epidemic in every major US city, social media sites — including Twitter — were also awash with opinions about the movement. When Twitter users read tweets about the #BLM movement, and the contemporaneous #BlueLivesMatter movement in support of policing efforts, they might be interested in categorizing tweets based on their attitudes toward the #BLM and #BlueLivesMatter movements.

### Dataset Overview 

#### Train/Test csv Column Headers
  - created_at: the date of the tweet
  - hashtags: the hashtags used in the tweet
  - text: text of the tweet
  - BLM: label for the tweet (positive, negative, or neither toward the BLM movement)
     - Refer to the 'tweetLabelCriteria.pdf' document for details on how labels were constructed
     
|   DataSet   | Num. Negative | Num. Positive | Num. Neither | Totals   | 
|-------------|----------     |-----------    |--------      |----------|
|     Train   |    1347       |    5891       |   871        |   8109   |  
|     Test    |    337        |    1474       |   218        |   2029   |  
|     Totals  |    1684       |    7365       |   1089       |   10138  |  


  
### Data cleaning and feature extraction
#### Preprocessing of training and testing data
  - Change emojis to words (import emoji)
  - Remove 'RT' from tweet
  - Remove capitalization
  - Remove urls & user mentions from tweet
  - Remove punctuation
  - Remove stopwords (from nltk.corpus import stopwords)
  - Perfom lemmatization on tokens (from nltk.stem import WordNetLemmatizer)
  - Removed Tweets that were labelled as neither

### Exploratory Data Analysis
<p align="center">
<picture>
<img src="https://github.com/nfasano/sentimentClassifier_blmTweets/blob/main/images/wordCloud.jpg" alt="drawing" width="600"/> 
</picture>
</p>

<p align="center">
<picture>
<img src="https://github.com/nfasano/sentimentClassifier_blmTweets/blob/main/images/histograms.jpg" alt="drawing" width="850"/> 
</picture>
</p>

### Form training and testing matrices

### Building models
Eight models were trained on the training dataset: KNN, M. Naive Bayes, Log. Reg., SVM (Lin), SVM (rbf), C. Naive Bayes, LDA, and Random Forest.
5-Fold cross validation was used to determine hyperparameters at train time

### Comparison of Classifiers

|   Classifier   | Accuracy | Precision | Recall | F1-score |   TN   |   FP   |   FN   |   TP   |
|----------------|----------|-----------|--------|----------|--------|--------|--------|--------|
|       KNN      |    0.83  |    0.85   |   0.97 |   0.90   |    76  |   261  |    44  |  1430  |
| M. Naive Bayes |    0.82  |    0.84   |   0.97 |   0.90   |    59  |   278  |    47  |  1427  |
|    Log. Reg.   |    0.83  |    0.85   |   0.96 |   0.90   |    90  |   247  |    66  |  1408  |
|    SVM (Lin)   |    0.82  |    0.83   |   0.98 |   0.90   |    45  |   292  |    26  |  1448  |
|    SVM (rbf)   |    0.83  |    0.85   |   0.96 |   0.90   |    84  |   253  |    58  |  1416  |
| C. Naive Bayes |    0.73  |    0.92   |   0.73 |   0.82   |   243  |    94  |   395  |  1079  |
|       LDA      |    0.82  |    0.86   |   0.94 |   0.90   |   109  |   228  |    94  |  1380  |
|  Random Forest |    0.80  |    0.85   |   0.91 |   0.88   |   103  |   234  |   127  |  1347  |


<p align="center">
<picture>
<img src="https://github.com/nfasano/sentimentClassifier_blmTweets/blob/main/images/roc_curve.jpg" alt="drawing" width="750"/> 
</picture>
</p>

### Future Work
Build ensemble classifiers, bi-gram, and tf-idf instead of bag of words.
