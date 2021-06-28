#Importing required libraries
import numpy as np
import tweepy 
from numpy.core.fromnumeric import shape 
import pandas as pd 
import re
import joblib
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Loading the dataset
fileName = "twitterNews.csv"
tweets = pd.read_csv(fileName)

#Visualizing the dataset
print(tweets.head())

#Initializing features and labels
features = tweets.iloc[:, 10].values
labels = tweets.iloc[:, 1].values

processed_features = []

#Processing the features
for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)

#Applying feature extration on processed data
cv=CountVectorizer() 
word_count_vector=cv.fit_transform(processed_features)

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
tfidf_transformer.fit(word_count_vector)

count_vector=cv.transform(processed_features) 
processed_features=tfidf_transformer.transform(count_vector)

#Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

#Training the model
text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)

#Testing the model on test data
predictions = text_classifier.predict(X_test)

print(accuracy_score(y_test, predictions))

#Saving the model to disk
joblib.dump(text_classifier, "random_forest.joblib")
print("Saved model to disk")

#Drawing confusion matrix
from sklearn import metrics
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig('ConfusionMatrix.png')

    
cm = metrics.confusion_matrix(y_test, predictions, labels=['negative', 'neutral', 'positive'])
plot_confusion_matrix(cm, classes=['negative', 'neutral', 'positive'])

#Implementing Sentiment Analysis on new tweets
consumer_key= 'UKTW4ujt7SaZcDu1weHrum7ZD'
consumer_secret= '2wZ0rPE9UbpzPcn23SdF6n6bYcfoBsRK2S7ts3SqRMsxdCwfoo'
access_key = "893754073-TUDQXMfD0orTk5NZok61FtkRlZnoIkGMucDLT6xc"
access_secret = "7UCICvzYIQ3v2iBCxhlyvyd64mXtfvBuPj7jEZuPpZfjg"
  
# Function to extract tweets 
def get_tweets(username): 
          
        # Authorization to consumer key and consumer secret 
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
  
        # Access to user's access key and access secret 
        auth.set_access_token(access_key, access_secret) 
  
        # Calling api 
        api = tweepy.API(auth) 
  
        # 10 tweets to be extracted 
        tweets = api.user_timeline(screen_name=username,count=10)
  
        # Empty Array 
        tmp=[]  
  
        # create array of tweet information: username,  
        # tweet id, date/time, text 
        tweets_for_csv = [tweet.text for tweet in tweets] # CSV file created  
        for j in tweets_for_csv: 
  
            # Appending tweets to the empty array tmp 
            tmp.append(j)  
  
        # Printing the tweets 
        tmp=np.array(tmp)
        print(tmp)
        return tmp
ch='Y'

while(1):
    #Getting the user handle
    print("\nEnter the user handle: ")
    handle=str(input()) 
    features=get_tweets(handle)

    processed_features = []

    for sentence in range(0, len(features)):
        # Remove all the special characters
        processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

        # remove all single characters
        processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

        # Remove single characters from the start
        processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

        # Substituting multiple spaces with single space
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

        # Removing prefixed 'b'
        processed_feature = re.sub(r'^b\s+', '', processed_feature)

        # Converting to Lowercase
        processed_feature = processed_feature.lower()

        processed_features.append(processed_feature)

    #Performing features extraction of the new tweets
    count_vector=cv.transform(processed_features) 
    processed_features=tfidf_transformer.transform(count_vector)

    #Loading the dataset
    model=joblib.load("random_forest.joblib")
    print("Loaded model from disk")

    #Predicting the sentiment
    X_test=processed_features
    predictions = model.predict(X_test)
    print("Predicted values: ")
    print(predictions)

    print("\nWant to continue?(Y/N) ")
    ch=str(input())
    if(ch=='N'):
        break;