import numpy as np

import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)


from imblearn.over_sampling import SMOTE, ADASYN


data=pd.read_csv('C:\\Users\\Utsav\\Desktop\\pulsar_prediction\\pulsar_stars.csv', sep=',',header=0)


pulsarData=data.values
pulsarData=np.array(pulsarData)

#split the data

train, test = train_test_split(data, test_size=0.2)
pulsar=np.array(train)
train=np.array(train)
test=np.array(test)

from sklearn.preprocessing import MaxAbsScaler

scaler=MaxAbsScaler()
train=scaler.fit_transform(train)
test=scaler.fit_transform(test)
print(len(test))
print("len train before resmpling:")
print(len(train))

#copy labels of test & train data elsewhere
y_train=np.zeros(len(train))
y_test_1=np.zeros(len(test))
for i in range(len(test)):
    y_test_1[i]=test[i][8]
    test[i][8]=-1
#replace all test and train labels by -1
for i in range(len(train)):
    y_train[i]=train[i][8]
    train[i][8]=-1
 
train, y_train = SMOTE().fit_resample(train, y_train)
print("len train after resampling:")
print(len(train))

train = train[:,:8]
test = test[:,:8]



from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel='linear')
clf.fit(train,y_train)
y_pred = clf.predict(test)

# calculate score

score=0
for i in range(len(test)):
    if y_pred[i]==y_test_1[i]:
        score=score+1
score=score*100/len(test)

print("Accuracy = " + "%.2f" % score + "%")


#calculate no of flase negatives
score=0

for i in range(len(test)):
    if y_pred[i]==0 and y_test_1[i]==1:
        score=score+1

        
#score=score*100/len(test)
percent = score/len(test)


print("No of false negatives = %d" % score)
print("No of positives in test data = %d" % sum(y_test_1))
print("No of positives predicted : %d" % sum(y_pred))

X = train[:,:8]
y = y_train

print(np.shape(X))
print(np.shape(y))

from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=True)
# kf.get_n_splits(train)
print(kf.split(X))

p=0
y_final_pred = np.zeros((len(test), 10)) #10 is the number of folds
for train_index, test_index in kf.split(X):
#     print("Train Index: ", train_index)
#     print(len(train_index))
#     print("Test Index: ", test_index)
#     print(len(test_index), "\n")
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    clf = SVC(kernel='linear')
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    # predict on actual test set
    
    y_final_pred[:, p] = clf.predict(test)
    
    
    print("\n#### TRIAL NO. %s #### \n" % str(p+1))
    p = p+1
    # calculate score on validation set
    
    score=0
    for i in range(len(y_test)):
        if y_pred[i]==y_test[i]:
            score=score+1
    score=score*100/len(y_test)

    print("Accuracy = " + "%.2f" % score + "%")


    #calculate no of flase negatives
    score=0

    for i in range(len(y_test)):
        if y_pred[i]==0 and y_test[i]==1:
            score=score+1


    #score=score*100/len(test)
    percent = score/len(y_test)


    print("No of false negatives = %d" % score)
    print("No of positives in test data = %d" % sum(y_test))
    print("No of positives predicted : %d" % sum(y_pred))
    
print(np.shape(y_final_pred))

y_final = np.zeros(len(test))
print(np.shape(y_final))
for i in range(len(y_final)):
    y_final[i] = sum(y_final_pred[i,:])
    
print(y_final)
plt.hist(y_final)
plt.show()

pred=np.zeros(len(test))
print(np.shape(pred))
print(np.shape(test))

for i in range(len(test)):
    if y_final[i]>=5:
        pred[i] = 1
    else:
        pred[i] = 0
        
# calculate score

print("Threshold: 5/10 classifiers")

score=0
for i in range(len(test)):
    if pred[i]==y_test_1[i]:
        score=score+1
score=score*100/len(test)

print("Accuracy = " + "%.2f" % score + "%")


#calculate no of flase negatives
score=0

for i in range(len(test)):
    if pred[i]==0 and y_test_1[i]==1:
        score=score+1

        
#score=score*100/len(test)
percent = score/len(test)


print("No of false negatives = %d" % score)
print("No of positives in test data = %d" % sum(y_test_1))
print("No of positives predicted : %d" % sum(pred))


print("Threshold: 1/10 classifiers")

for i in range(len(test)):
    if y_final[i]>=1:
        pred[i] = 1
    else:
        pred[i] = 0


score=0
for i in range(len(test)):
    if pred[i]==y_test_1[i]:
        score=score+1
score=score*100/len(test)

print("Accuracy = " + "%.2f" % score + "%")


#calculate no of flase negatives
score=0

for i in range(len(test)):
    if pred[i]==0 and y_test_1[i]==1:
        score=score+1

        
#score=score*100/len(test)
percent = score/len(test)


print("No of false negatives = %d" % score)
print("No of positives in test data = %d" % sum(y_test_1))
print("No of positives predicted : %d" % sum(pred))




