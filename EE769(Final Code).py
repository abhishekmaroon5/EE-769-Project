
# All the libraries
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import warnings
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold 
from sklearn import cross_validation
from collections import Counter, defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import math
from sklearn.metrics import normalized_mutual_info_score
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")
import nltk

from mlxtend.classifier import StackingClassifier

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression


def main():
# Lets import our dataset  
	data_variants = pd.read_csv("training_variants.csv")
	data_text =pd.read_csv("training_text.csv",sep="\|\|",engine="python",names=["ID","Text"],skiprows=1)

	Univariant_analysis(data_variants,data_text)

	# Some Preprocessing on Text Features
	nltk.download('stopwords')
	stop=set(stopwords.words('english'))
	# Lets print all the 
	print(stop)
	# loading stop words from nltk library
	stop_words = set(stopwords.words('english'))
	# Remove all the stop words
	for index, row in data_text.iterrows():
		nlp_preprocessing(data_text,row['Text'], index, 'Text',stop_words)
	# Comparision before and after preprocessing 
	data_text2 =pd.read_csv("training_text.csv",sep="\|\|",engine="python",names=["ID","Text"],skiprows=1)
	print("Earlier the count of the ",len(data_text2['Text'][1]))
	print("Now the count of the ",len(data_text['Text'][1]))
		# Merge both the tables
	result = pd.merge(data_variants, data_text,on='ID', how='left')


	# Preprocessing on Gene and Variation features
	y=result["Class"].values
	result.Gene      = result.Gene.str.replace('\s+', '_')
	result.Variation = result.Variation.str.replace('\s+', '_')

	X_tr, X_cv, y_tr, y_cv = cross_validation.train_test_split(result, y, test_size=0.3)
	X_tr, X_test, y_tr, y_test = cross_validation.train_test_split(X_tr, y_tr, test_size=0.3)
	print("Shape of training data %d ,test data %d ,cross-validate data %d"%(len(X_tr),len(X_test),len(X_cv)))

	#Base Line model 1
	Baseline_model(X_tr, X_cv, y_tr, y_cv,X_tr, X_test, y_tr, y_test)
	# Gene Feature Important(Model on Feature 1)
	Gene_only_model(X_tr, X_cv, y_tr, y_cv,X_tr, X_test, y_tr, y_test)
	# Model on Feature 2
	variation_only_model(X_tr, X_cv, y_tr, y_cv,X_tr, X_test, y_tr, y_test)
	# Model on Feature 3
	text_only_model(result,y)

	# Combime different features
	train,test,cv=Combine_features(X_tr, X_cv, y_tr, y_cv,X_tr, X_test, y_tr, y_test,result)
	# Model 1
	logreg(train,test,cv)
	# Model 2
	

	# Model 3

	# Stacked Model



##########################################
def nlp_preprocessing(data_text,total_text, index, column,stop_words):
	if type(total_text) is not int:
		string = ""
	# replace every special char with space
		total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))
	# replace multiple spaces with single space
		total_text = re.sub('\s+',' ', str(total_text))
	# converting all the chars into lower-case.
		total_text = total_text.lower()

	for word in total_text.split():
	# if the word is a not a stop word then retain that word from the data
		if not word in stop_words:
			string += word + " "
		data_text[column][index] = string
###########################################
def Baseline_model(X_tr, X_cv, y_tr, y_cv, X_test, y_test):
	# We need some a list of size 9 which will sum to 1
	test_data_len = X_test.shape[0]
	cv_data_len = X_cv.shape[0]

	# we create a output array that has exactly same size as the CV data
	cv_predicted = np.zeros((cv_data_len,9))
	j=1
	for i in range(cv_data_len):
		rand_probs = np.random.rand(1,9)# array of size 9 sum to 1.
		cv_predicted[i]=((rand_probs/sum(sum(rand_probs))))[0]# Lie between 0 to 1
	print("Log Loss at cross-validation step is %d",log_loss(y_cv,cv_predicted, eps=1e-15))    
	# we create a output array that has exactly same size as the CV data
	test_predicted = np.zeros((test_data_len,9))
	j=1
	for i in range(test_data_len):
		rand_probs = np.random.rand(1,9)# array of size 9 sum to 1.
		test_predicted[i]=((rand_probs/sum(sum(rand_probs))))[0]# Lie between 0 to 1
	print("Log Loss at test step is %d",log_loss(y_test,test_predicted, eps=1e-15))

##########################################
def Univariant_analysis(data_variants,data_text):
	print("")
##########################################
def Gene_only_model(X_tr, X_cv, y_tr, y_cv, X_test, y_test):
	gene_vectorizer = CountVectorizer()
	train_gene_feature_onehotCoding = gene_vectorizer.fit_transform(X_tr['Gene'])
	test_gene_feature_onehotCoding = gene_vectorizer.transform(X_test['Gene'])
	cv_gene_feature_onehotCoding = gene_vectorizer.transform(X_cv['Gene'])
	tunes_para=[10 ** x for x in range(-5, 1)]
	cv_array_loss=[]
	# want to tune for alpha in these code
	for i in tunes_para:
		print("for alpha =", i)
		clf = SGDClassifier(class_weight='balanced',alpha=i,penalty='l2',loss='log',random_state=42)
		clf.fit(train_gene_feature_onehotCoding,y_tr)
		clf2 = CalibratedClassifierCV(clf, method="sigmoid")
		clf2.fit(train_gene_feature_onehotCoding,y_tr)
		clf2_probs = clf2.predict_proba(cv_gene_feature_onehotCoding)
		cv_array_loss.append(log_loss(y_cv, clf2_probs, labels=clf.classes_, eps=1e-15))
		# to avoid rounding error while multiplying probabilites we use log-probability estimates
		print("Log Loss :",log_loss(y_cv, clf2_probs)) 
		# We got a minimum loss at 0.001 try it again
	clf = SGDClassifier(class_weight='balanced',alpha=0.001,penalty='l2',loss='log',random_state=42)
	clf.fit(train_gene_feature_onehotCoding,y_tr)
	clf2 = CalibratedClassifierCV(clf, method="sigmoid")
	clf2.fit(train_gene_feature_onehotCoding,y_tr)
	clf2_probs = clf2.predict_proba(cv_gene_feature_onehotCoding)
	cv_array_loss.append(log_loss(y_cv, clf2_probs, labels=clf.classes_, eps=1e-15))
	# to avoid rounding error while multiplying probabilites we use log-probability estimates
	print("Log Loss :",log_loss(y_cv, clf2_probs))
################################################################

def variation_only_model(X_tr, X_cv, y_tr, y_cv, X_test, y_test):
	#Let us do one hot convert of this(for test, train, cv)
	gene_vectorizer = CountVectorizer()
	train_Variation_feature_onehotCoding = gene_vectorizer.fit_transform(X_tr['Variation'])
	test_Variation_feature_onehotCoding = gene_vectorizer.transform(X_test['Variation'])
	cv_Variation_feature_onehotCoding = gene_vectorizer.transform(X_cv['Variation'])
	tunes_para=[10 ** x for x in range(-5, 1)]
	cv_array_loss=[]
	# want to tune for alpha in these code
	for i in tunes_para:
		print("for alpha =", i)
		clf = SGDClassifier(class_weight='balanced',alpha=i,penalty='l2',loss='log',random_state=42)
		clf.fit(train_Variation_feature_onehotCoding,y_tr)
		clf2 = CalibratedClassifierCV(clf, method="sigmoid")
		clf2.fit(train_Variation_feature_onehotCoding,y_tr)
		clf2_probs = clf2.predict_proba(cv_Variation_feature_onehotCoding)
		cv_array_loss.append(log_loss(y_cv, clf2_probs, labels=clf.classes_, eps=1e-15))
		# to avoid rounding error while multiplying probabilites we use log-probability estimates
		print("Log Loss :",log_loss(y_cv, clf2_probs)) 
	clf = SGDClassifier(class_weight='balanced',alpha=0.001,penalty='l2',loss='log',random_state=42)
	clf.fit(train_Variation_feature_onehotCoding,y_tr)
	clf2 = CalibratedClassifierCV(clf, method="sigmoid")
	clf2.fit(train_Variation_feature_onehotCoding,y_tr)
	clf2_probs = clf2.predict_proba(cv_Variation_feature_onehotCoding)
	cv_array_loss.append(log_loss(y_cv, clf2_probs, labels=clf.classes_, eps=1e-15))
	# to avoid rounding error while multiplying probabilites we use log-probability estimates
	print("Log Loss :",log_loss(y_cv, clf2_probs))
################################################################
def text_only_model(result,y):
	ext_vectorizer = CountVectorizer(min_df=4)# In Feature we choose only those words with greater then 3 times occurence
	x=text_vectorizer.fit_transform(Result['Text'])
	X_tr, X_cv, y_tr, y_cv = cross_validation.train_test_split(Dataset, y, test_size=0.3)
	X_tr, X_test, y_tr, y_test = cross_validation.train_test_split(X_tr, y_tr, test_size=0.3)
	Dataset=normalize(x,axis=0)
	tunes_para=[10 ** x for x in range(-5, 1)]
	cv_array_loss=[]
# want to tune for alpha in these code
	for i in tunes_para:
		print("for alpha =", i)
		clf = SGDClassifier(class_weight='balanced',alpha=i,penalty='l2',loss='log',random_state=42)
		clf.fit(X_tr,y_tr)
		clf2 = CalibratedClassifierCV(clf, method="sigmoid")
		clf2.fit(X_tr,y_tr)
		clf2_probs = clf2.predict_proba(X_test)
		cv_array_loss.append(log_loss(y_test, clf2_probs, labels=clf.classes_, eps=1e-15))
		# to avoid rounding error while multiplying probabilites we use log-probability estimates
		print("Log Loss :",log_loss(y_test, clf2_probs)) 
################################################################	
def  Combine_features(X_tr, X_cv, y_tr, y_cv, X_test, y_test,result):
	#Let us do one hot convert of this(for test, train, cv)
	gene_vectorizer = CountVectorizer()
	train_gene_feature_onehotCoding = gene_vectorizer.fit_transform(X_tr['Gene'])
	test_gene_feature_onehotCoding = gene_vectorizer.transform(X_test['Gene'])
	cv_gene_feature_onehotCoding = gene_vectorizer.transform(X_cv['Gene'])
	#Let us do one hot convert of this(for test, train, cv)
	gene_vectorizer = CountVectorizer()
	train_Variation_feature_onehotCoding = gene_vectorizer.fit_transform(X_tr['Variation'])
	test_Variation_feature_onehotCoding = gene_vectorizer.transform(X_test['Variation'])
	cv_Variation_feature_onehotCoding = gene_vectorizer.transform(X_cv['Variation'])



	text_vectorizer = CountVectorizer(min_df=4)# In Feature we choose only those words with greater then 3 times occurence
	x=text_vectorizer.fit_transform(Result['Text'])


	Dataset=normalize(x,axis=0)
	X_tr, X_cv, y_tr, y_cv = cross_validation.train_test_split(Dataset, y, test_size=0.3)
	X_tr, X_test, y_tr, y_test = cross_validation.train_test_split(X_tr, y_tr, test_size=0.3)
	# Lets print the shape of all the three Features
	print(train_gene_feature_onehotCoding.shape)
	print(test_gene_feature_onehotCoding.shape)
	print(cv_gene_feature_onehotCoding.shape)
	# Lets print the shape of all the three Features
	print(train_Variation_feature_onehotCoding.shape)
	print(test_Variation_feature_onehotCoding.shape)
	print(cv_Variation_feature_onehotCoding.shape)

	print(X_tr.shape)
	print(X_test.shape)
	print(X_cv.shape)

	X_tr=pd.DataFrame(X_tr.todense())
	X_test=pd.DataFrame(X_test.todense())
	X_cv=pd.DataFrame(X_cv.todense())
	train_Variation_feature_onehotCoding=pd.DataFrame(train_Variation_feature_onehotCoding.todense())
	test_Variation_feature_onehotCoding=pd.DataFrame(test_Variation_feature_onehotCoding.todense())
	cv_Variation_feature_onehotCoding=pd.DataFrame(cv_Variation_feature_onehotCoding.todense())
	train_gene_feature_onehotCoding=pd.DataFrame(train_gene_feature_onehotCoding.todense())
	test_gene_feature_onehotCoding=pd.DataFrame(test_gene_feature_onehotCoding.todense())
	cv_gene_feature_onehotCoding=pd.DataFrame(cv_gene_feature_onehotCoding.todense())
	train = X_tr.join(train_gene_feature_onehotCoding,lsuffix="_X_tr",rsuffix="_train_gene_feature_onehotCoding")
	train = train.join(train_Variation_feature_onehotCoding,lsuffix="_train",rsuffix="_train_Variation_feature_onehotCoding")
	print(train.shape)
	test = X_test.join(test_gene_feature_onehotCoding,lsuffix="_X_test",rsuffix="_test_gene_feature_onehotCoding")
	test = test.join(test_Variation_feature_onehotCoding,lsuffix="_test",rsuffix="_test_Variation_feature_onehotCoding")
	print(test.shape)
	cv = X_cv.join(test_gene_feature_onehotCoding,lsuffix="_X_cv",rsuffix="_cv_gene_feature_onehotCoding")
	cv = cv.join(test_Variation_feature_onehotCoding,lsuffix="_cv",rsuffix="_cv_Variation_feature_onehotCoding")
	print(cv.shape)

	# Before appliing model lets remove all nan value
	features=train.columns
	pd.options.mode.chained_assignment = None 	
	for i in features:
		print("Done")  
		train[i].fillna(0, inplace=True)


	features=test.columns
	pd.options.mode.chained_assignment = None 
	for i in features:
		test[i].fillna(0, inplace=True)

	features=cv.columns
	pd.options.mode.chained_assignment = None 
	for i in features:
		cv[i].fillna(0, inplace=True)
	return train,test,cv    
##########################################################
def logreg(train,test,cv):
	tunes_para=[10 ** x for x in range(-5, 1)]
	cv_array_loss=[]
	# want to tune for alpha in these code
	for i in tunes_para:
		print("for alpha =", i)
		clf = SGDClassifier(class_weight='balanced',alpha=i,penalty='l2',loss='log',random_state=42)
		clf.fit(train,y_tr)
		clf2 = CalibratedClassifierCV(clf, method="sigmoid")
		clf2.fit(train,y_tr)
		clf2_probs = clf2.predict_proba(cv)
		cv_array_loss.append(log_loss(y_cv, clf2_probs, labels=clf.classes_, eps=1e-15))
		# to avoid rounding error while multiplying probabilites we use log-probability estimates
		print("Log Loss :",log_loss(y_cv, clf2_probs)) 
###########################################################    
if __name__=="__main__":
	main()







