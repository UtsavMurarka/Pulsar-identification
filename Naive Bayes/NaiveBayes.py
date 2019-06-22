import numpy as np
import pandas as pd
import math

from sklearn.model_selection import train_test_split


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
#print(train)

#copy labels of test & train data elsewhere
y_train=np.zeros(len(train))
y_test=np.zeros(len(test))
for i in range(len(test)):
    y_test[i]=test[i][8]
    test[i][8]=-1
#replace all test and train labels by -1
for i in range(len(train)):
    y_train[i]=train[i][8]
    train[i][8]=-1
    


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
naiveBayes = gnb.fit(train, y_train)
y_pred = naiveBayes.predict(test)


# calculate score
print("############## SKLEARN SCORE ####################")
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

print("############################### \n\n\n\n\n")

train=np.array(train)
NegativeCount=0
PositiveCount=0
for i in range(14318):
    if pulsar[i][8]==0:
        NegativeCount=NegativeCount+1
    if pulsar[i][8]==1:
        PositiveCount=PositiveCount+1
print(NegativeCount)
print(PositiveCount)
print(PositiveCount+NegativeCount)

positives=np.zeros((PositiveCount,9))
negatives=np.zeros((NegativeCount,9))
    
j=0
k=0
    
for i in range(14318):

    if(pulsar[i,8] == 1):
        positives[j,:] = pulsar[i,:]
        j=j+1
    if(pulsar[i,8] == 0):
        negatives[k,:] = pulsar[i,:]
        k=k+1

meanMatrixP = np.zeros(8)

for i in range(8):
    meanMatrixP[i] = np.mean(positives[:,i])

varMatrixP = np.zeros(8)

for i in range(8):
    varMatrixP[i] = np.var(positives[:,i])
    
    

meanMatrixNP = np.zeros(8)

for i in range(8):
    meanMatrixNP[i] = np.mean(negatives[:,i])

varMatrixNP = np.zeros(8)

for i in range(8):
    varMatrixNP[i] = np.var(negatives[:,i])


#make predictions:

y_pred1 = np.zeros(len(test))
print(sum(y_pred))

p_prob = PositiveCount/(PositiveCount+NegativeCount)
np_prob = NegativeCount/(PositiveCount+NegativeCount)
print(p_prob)
print(np_prob)

for i in range(len(test)):
    prod_pulsar=1
    for p in range(8):
        prod_pulsar = prod_pulsar*np.exp((-(test[i,p]-meanMatrixP[p])**2)/(2*varMatrixP[p]))/(np.sqrt(2*3.14*varMatrixP[p]))
    pulsar_prob = p_prob * (prod_pulsar)
    

    prod_nonpulsar=1
    for nonp in range(8):
        prod_nonpulsar = prod_nonpulsar*np.exp((-(test[i,nonp]-meanMatrixNP[nonp])**2)/(2*varMatrixNP[nonp]))/(np.sqrt(2*3.14*varMatrixNP[nonp]))
    nonpulsar_prob = np_prob * (prod_nonpulsar)
    
    if (pulsar_prob>=nonpulsar_prob):
        y_pred1[i] = 1
    else:
        y_pred1[i] = 0
    


# calculate score
print("###### MY IMPLEMENTATION SCORE ########")

score=0
for i in range(len(test)):
    if y_pred1[i]==y_test[i]:
        score=score+1
score=score*100/len(test)

print("Accuracy = " + "%.2f" % score + "%")


#calculate no of flase negatives
score=0

for i in range(len(test)):
    if y_pred1[i]==0 and y_test[i]==1:
        score=score+1

        
#score=score*100/len(test)
percent = score/len(test)


print("No of false negatives = %d" % score)
print("No of positives in test data = %d" % sum(y_test))
print("No of positives predicted : %d" % sum(y_pred1))

print("################## \n\n\n")