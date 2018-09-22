import tweepy,twitter_text
auth = tweepy.OAuthHandler('k6BFDMNQ7Ze2zmfq7RXGlrW0T','2Sqax0hGpfXUVOwzUFGccq4x3evJ6hzWoUXeWxNFtYzxJA1itm')
auth.set_access_token('821745978122338310-xzXQc56UyBZm6G8nsL56MFqZazWFCUL','hAhPfPZbiCer6AnyQ1Fl43XepTLSneCPUFbJV4kOMlowq')
api = tweepy.API(auth)
print(api)

query = input('Enter your Query to be analysed:')
no_queries = int(input('Enter number of queries to be analysed:'))

searched = tweepy.Cursor(api.search, q = query, lang='en').items(no_queries)

import preprocessor as p
list1 = []
for tweet in searched:
    list1.append(p.clean(tweet.text))
    
def percent(part,whole):
    return(100* float(part)/float(whole))


positive = 0
negative = 0
neutral = 0

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
for i in list1:
    ss = sid.polarity_scores(i)
    #print(ss)
    negative +=ss['neg']
    neutral +=ss['neu']
    positive +=ss['pos']
    

    
    
positive = round(percent(positive,no_queries),2)
negative = round(percent(negative,no_queries),2)
neutral = round(percent(neutral,no_queries),2)



print(positive,'%'+' positive')
print(negative,'%'+' negative')
print(neutral,'%'+' neutral')

from nltk.corpus import twitter_samples
twitter_samples.fileids()

negtweets = twitter_samples.strings('negative_tweets.json')
postweets = twitter_samples.strings('positive_tweets.json')

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import preprocessor as p

def word_feats(words):
    return dict([(word,True) for word in words])

negfeats = [(word_feats(nltk.word_tokenize(p.clean(i))),'neg') for i in negtweets]
posfeats = [(word_feats(nltk.word_tokenize(p.clean(i))),'pos') for i in postweets]
#print(negfeats)


#neg_t = word_feats(negfeats,neg_words)
#pos_t = word_feats(posfeats,pos_words)
negcutoff = int(len(negfeats)*1/2)
poscutoff = int(len(posfeats)*1/2)
trainfeats = negfeats[:negcutoff]+posfeats[:poscutoff]
testfeats = posfeats[negcutoff:]+posfeats[poscutoff:]

print("Train on %d instances, test on %d instances"%(len(trainfeats),len(testfeats)))
classifier = NaiveBayesClassifier.train(trainfeats)

print("Accuracy:",nltk.classify.util.accuracy(classifier,testfeats))

for i in list1:
    test_tweet = word_feats(i)
    print(i+":"+classifier.classify(test_tweet))
    
classifier.show_most_informative_features()
