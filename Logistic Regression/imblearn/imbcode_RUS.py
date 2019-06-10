import imblearn
import numpy as np
import pandas as pd
import math
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
print("###########################################")


data=pd.read_csv('C:\\Users\\Utsav\\Desktop\\pulsar_prediction\\pulsar_stars.csv', sep=',',header=0)


pulsarData=data.values
pulsarData=np.array(pulsarData)
#split the data

train, test = train_test_split(data, test_size=0.2)
pulsar=np.array(train)
train=np.array(train)
test=np.array(test)

print(len(test))
print(len(train))
print("###########################################")


#copy labels of test & train data elsewhere
#replace all test and train labels by 1 (for intercept term)


y_train=np.zeros(len(train))
y_test=np.zeros(len(test))
for i in range(len(test)):
    y_test[i]=test[i][8]
    test[i][8]=1

for i in range(len(train)):
    y_train[i]=train[i][8]
    train[i][8]=1
   

#logistic regression using sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#### IMBLEARN

# Undersampling using random under sampler

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(train, y_train)
print("IMBLEARN RANDOM UNDER SAMPLING")
print("Number of examples")
print(len(X_resampled))
print(len(y_resampled))
print("###########################################")

print("CHECK: This number below should be half of above number to ensure equal sampling") 
print(sum(y_resampled))

#Logistic Regtression


logreg = LogisticRegression()
#logreg.fit(train, y_train)
logreg.fit(X_resampled, y_resampled)

y_pred = logreg.predict(test)


# calculate score

score=0
for i in range(len(test)):
    if y_pred[i]==y_test[i]:
        score=score+1
score=score*100/len(test)

print("Accuracy = " + "%.2f" % score + "%")


#calculate no of flase negatives
score=0

for i in range(len(test)):
    if y_pred[i]==0 and y_test[i]==1:
        score=score+1

        
#score=score*100/len(test)
percent = score/len(test)


print("No of false negatives = %d" % score)
print("No of positives in test data = %d" % sum(y_test))
print("No of positives predicted : %d" % sum(y_pred))