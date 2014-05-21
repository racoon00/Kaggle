import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier

filename_train = 'train.csv'
filename_test = 'test.csv'
dataset = np.genfromtxt(open(filename_train,'r'), delimiter=',', dtype='f8')[1:] 

target = [x[0] for x in dataset]
train = [x[1:] for x in dataset]

print "########Train Data#########"
rf = RandomForestClassifier(n_estimators=10)
rf.fit(train,target)

print "########Training is done#########"

test = np.genfromtxt(open(filename_test,'r'), delimiter=',', dtype='f8')[1:]  

print "########Write the label into submission#########"
np.savetxt('submissionrf.csv', rf.predict(test), delimiter=',', fmt='%d')


testdata = np.genfromtxt(open('submissionrf.csv','r'),delimiter = ',', dtype='f8')

Imageid = np.zeros((28000,2))
for i in range(28000):
    for j in range(2):
        if j == 0:
                Imageid[i][j] = i+1;    
        else:
                Imageid[i][j] = testdata[i]; 

print "#############Write the image id and label into submission########"        
np.savetxt('submissionrfint.csv', Imageid, delimiter=',', fmt='%d')

	


	
