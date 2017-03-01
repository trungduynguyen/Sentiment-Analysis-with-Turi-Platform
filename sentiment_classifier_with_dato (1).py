# -*- coding: utf-8 -*-
# coding: utf-8

import graphlab as gl
import cPickle as pickle
from graphlab.toolkits.feature_engineering import TFIDF


def tokenize(sf=[], text_cols = ['review']):
	tokenizer = gl.feature_engineering.Tokenizer(features = text_cols)
	tokenizer.fit(sf)
	tokenized_sf = tokenizer.transform(sf)

	return tokenized_sf

def construct_tfidf(tokenized_sf = [], token_cols = ['review']):
	tfidf = gl.feature_engineering.TFIDF(token_cols)
	fit_tfidf = tfidf.fit(tokenized_sf)
	tfidf_sf = fit_tfidf.transform(tokenized_sf)

	return tfidf_sf

def extract_feature_from_a_tweet(tweet_list = ['']):
	tweet_sf = gl.SFrame(tweet_list)
	tweet_sf['1grams features'] = gl.text_analytics.count_ngrams(tweet_sf['X1'],1)
	tweet_sf['2grams features'] = gl.text_analytics.count_ngrams(tweet_sf['X1'],2)
	tweet_sf.head()

	return tweet_sf

def print_statistics(result):
    print "*" * 30
    print "Accuracy        : ", result["accuracy"]
    print "Recall          : ", result["recall"]
    print "Precision       : ", result["precision"]
    print "F1 Score        : ", result["f1_score"]
    print "Confusion Matrix: \n", result["confusion_matrix"]
    # print "ROC Curve       : \n", result["roc_curve"]
 

### Dataset Path
#data_path = "cutoff_data_train.csv"
data_path = 'preprocessed_train_data.csv'
# traindata_path = "prep_train.csv"
# testdata_path = "prep_test.csv"
data_test_path = 'preprocessed_test_data.csv'
#data_test_path = "cutoff_data_test.csv"

### Loading Data
sf_data = gl.SFrame.read_csv(data_path,header=True)
sf_data.head()
#train_data, test_data = sf_data.random_split(.7, seed=5)
train_data,test_data = sf_data.random_split(.1,seed = 7)
print(len(train_data), len(test_data))

### Constructing Bag-of-Words Classifier 
# train_data = gl.SFrame.read_csv(traindata_path,header=True)
train_data['1grams features'] = gl.text_analytics.count_ngrams(train_data['review'],1)
train_data['2grams features'] = gl.text_analytics.count_ngrams(train_data['review'],2)
train_data.head()

#creating the test dataset
# test_data = gl.SFrame.read_csv(testdata_path,header=True)
test_data['1grams features'] = gl.text_analytics.count_ngrams(test_data['review'],1)
test_data['2grams features'] = gl.text_analytics.count_ngrams(test_data['review'],2)
test_data.head()

# submission data
sm_data = gl.SFrame.read_csv(data_test_path,header=True)
sm_data['1grams features'] = gl.text_analytics.count_ngrams(sm_data['review'],1)
sm_data['2grams features'] = gl.text_analytics.count_ngrams(sm_data['review'],2)
sm_data.head()

#features_list=['1grams features','2grams features']
features_list=['1grams features']
######################
### without TF-IDF ###
######################
cls_run_name = "dato_gl_model_without_tfidf"
cls1 = gl.classifier.create(train_data, target='sentiment', features=features_list)
cls1.save("./model/" + cls_run_name)
# predicting the sentiment of each tweet in the test dataset
test_data['prediction'] = cls1.classify(test_data)['class'].astype(int)
# saving the prediction to a CSV for submission
test_data[['review','sentiment','prediction']].save("./result/" + cls_run_name + ".csv", format="csv")
test_data.head()
result1 = cls1.evaluate(test_data)
print_statistics(result1)

#####################################
### predict for unlabel test data ###
#####################################
cls_run_name = "kaggle_submission_1_and_2_grams"
sm_data['prediction'] = cls1.classify(sm_data)['class'].astype(int)
# saving the prediction to a CSV for submission
sm_data[['id','prediction']].save("./result/" + cls_run_name + ".csv", format="csv")
sm_data.head()



####################
### Using TF-IDF ###
####################

### Tokenize
text_column = ['review']
train_tokenized_sf = tokenize(train_data, text_column)
train_tokenized_sf.head()
test_tokenized_sf = tokenize(test_data, text_column)
test_tokenized_sf.head()
### TF-IDF transform
train_tfidf_sf = construct_tfidf(train_tokenized_sf, features_list)
train_tfidf_sf.head()
test_tfidf_sf = construct_tfidf(test_tokenized_sf, features_list)
test_tfidf_sf.head()
### Train model with TF-IDF
cls_run_name = "dato_gl_model_with_tfidf"
cls_with_tfidf = gl.classifier.create(train_tfidf_sf, target='sentiment', features=features_list)
cls_with_tfidf.save("./model/" + cls_run_name)
#predicting the sentiment of each tweet in the test dataset
test_tfidf_sf['prediction'] = cls_with_tfidf.classify(test_tfidf_sf)['class'].astype(int)

#saving the prediction to a CSV for submission
test_tfidf_sf[['review','sentiment','prediction']].save("./result/" + cls_run_name + ".csv", format="csv")
test_tfidf_sf.head()
result2 = cls_with_tfidf.evaluate(test_tfidf_sf)
print_statistics(result2)

#####################
### random forest ###
#####################
# cls3 = gl.random_forest_classifier.create(train_data, target='sentiment', features=['1grams features','2grams features'], max_iterations = 100)

# cls3.save("dato_gl_model_randomforest")

# #predicting the sentiment of each tweet in the test dataset
# test_data['prediction'] = cls3.classify(test_data)['class'].astype(int)

# #saving the prediction to a CSV for submission
# test_data[['tweet','sentiment','prediction']].save("/home/tanthm/Documents/sentiment-on-dato/result/predictions_randomforest.csv", format="csv")
# test_data.head()
# result3 = cls3.evaluate(test_data)
# print_statistics(result3)

### test data from brandwatch2
testdata_bw_path2 = "/home/tanthm/Documents/sentiment-on-dato/prep_bw_test2.csv"
 
bw_test_data2 = gl.SFrame.read_csv(testdata_bw_path2,header=True)
bw_test_data2['1grams features'] = gl.text_analytics.count_ngrams(bw_test_data2['tweet'],1)
bw_test_data2['2grams features'] = gl.text_analytics.count_ngrams(bw_test_data2['tweet'],2)
bw_test_data2.head()

bw_test_data2['prediction'] = cls2.classify(bw_test_data2)['class'].astype(int)
bw_test_data2[['tweet','sentiment','prediction']].save("/home/tanthm/Documents/sentiment-on-dato/result/predictions_without_tfidf_on_brandwatch_data.csv", format="csv")
bw_test_data2.head()
result4 = cls2.evaluate(bw_test_data2)
print_statistics(result4)

###########################

model = gl.random_forest_classifier.create(train_data,target='sentiment',features= features_list,validation_set= None, verbose= True)
prediction = model.classify(test_data)
prediction
result = model.evaluate(test_data)
result



