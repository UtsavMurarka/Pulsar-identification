#import imblearn
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
import datetime
from sklearn.preprocessing import MaxAbsScaler

scaler=MaxAbsScaler()



data=pd.read_csv('C:\\Users\\Utsav\\Desktop\\pulsar_prediction\\pulsar_stars.csv', sep=',',header=0)
pulsarData=data.values
pulsarData=np.array(pulsarData)
train, test = train_test_split(data, test_size=0.2)

train=scaler.fit_transform(train)
test=scaler.fit_transform(test)

print(len(test))
print(len(train))

pulsar=np.array(train)
train=np.array(train)
test=np.array(test)
y_train=np.zeros(len(train))
y_test=np.zeros(len(test))




for i in range(len(test)):
    y_test[i]=test[i][8]
    test[i][8]=1

for i in range(len(train)):
    y_train[i]=train[i][8]
    train[i][8]=1

positives = sum(y_train)
negatives = len(y_train) - positives

print("\n\n\n positives = %d \n negatives = %d \n\n\n" % (positives, negatives))


t1 = datetime.datetime.now()
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel='rbf')

clf.fit(train,y_train)
y_pred = clf.predict(test)

# calculate score
t2 = datetime.datetime.now()

score=0
for i in range(len(test)):
    if y_pred[i]==y_test[i]:
        score=score+1
score=score*100/len(test)
print("\n\n#####  rbf KERNEL  ######\n\n")
print("Accuracy = " + "%.2f" % score + "%")
runtime = t2 - t1
print("Runtime: %s" % (str(runtime)))

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

t1 = datetime.datetime.now()
clf = SVC(kernel='rbf', class_weight = 'balanced')
clf.fit(train,y_train)
y_pred = clf.predict(test)

t2 = datetime.datetime.now()


score=0
for i in range(len(test)):
    if y_pred[i]==y_test[i]:
        score=score+1
score=score*100/len(test)
print("\n\n##### rbf Kernel with class_weight = 'balanced' ######\n\n")
print("Accuracy = " + "%.2f" % score + "%")
runtime = t2 - t1
print("Runtime: %s" % (str(runtime)))


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

#SMOTE

t1 = datetime.datetime.now()

from imblearn.over_sampling import SMOTE, ADASYN
X_resampled, y_resampled = SMOTE().fit_resample(train, y_train)

print("\n\nIMBLEARN SMOTE (Synthetic Minority Over Sampling Technique)\n\n")
print("Number of examples")
print(len(X_resampled))
print(len(y_resampled))
print("###########################################")

print("\n\nCHECK: This number below should be half of above number to ensure equal sampling\n\n") 
print(sum(y_resampled))
print("###########################################")

clf = SVC(kernel='rbf', class_weight={1:1})
clf.fit(X_resampled,y_resampled)
y_pred = clf.predict(test)

t2 = datetime.datetime.now()


score=0
for i in range(len(test)):
    if y_pred[i]==y_test[i]:
        score=score+1
score=score*100/len(test)
print("\n\n##### rbf Kernel With SMOTE######\n\n")
print("Accuracy = " + "%.2f" % score + "%")
runtime = t2 - t1
print("Runtime: %s" % (str(runtime)))


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


#ROS

t1 = datetime.datetime.now()

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(train, y_train)



print("\n\nIMBLEARN ROS\n\n")
print("Number of examples")
print(len(X_resampled))
print(len(y_resampled))
print("###########################################")

print("\n\nCHECK: This number below should be half of above number to ensure equal sampling\n\n") 
print(sum(y_resampled))
print("###########################################")

clf = SVC(kernel='rbf')
clf.fit(X_resampled,y_resampled)
y_pred = clf.predict(test)

t2 = datetime.datetime.now()


score=0
for i in range(len(test)):
    if y_pred[i]==y_test[i]:
        score=score+1
score=score*100/len(test)
print("\n\n##### rbf Kernel With ROS######\n\n")
print("Accuracy = " + "%.2f" % score + "%")
runtime = t2 - t1
print("Runtime: %s" % (str(runtime)))


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



#RUS

t1 = datetime.datetime.now()

from imblearn.over_sampling import SMOTE, ADASYN
X_resampled, y_resampled = SMOTE().fit_resample(train, y_train)

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler()
X_resampled, y_resampled = rus.fit_resample(train, y_train)


print("\n\nIMBLEARN RUS\n\n")
print("Number of examples")
print(len(X_resampled))
print(len(y_resampled))
print("###########################################")

print("\n\nCHECK: This number below should be half of above number to ensure equal sampling\n\n") 
print(sum(y_resampled))
print("###########################################")

clf = SVC(kernel='rbf')
clf.fit(X_resampled,y_resampled)
y_pred = clf.predict(test)

t2 = datetime.datetime.now()


score=0
for i in range(len(test)):
    if y_pred[i]==y_test[i]:
        score=score+1
score=score*100/len(test)
print("\n\n##### rbf Kernel With RUS######\n\n")
print("Accuracy = " + "%.2f" % score + "%")
runtime = t2 - t1
print("Runtime: %s" % (str(runtime)))


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














