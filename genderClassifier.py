from sklearn import tree
from sklearn import svm
from sklearn.linear_model import SGDClassifier

#[height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Decission Tree Model
classifier = tree.DecisionTreeClassifier()

classifier = classifier.fit(X,Y)

prediction = classifier.predict([[190,70,43]])

print "Decission Tree: " + str(prediction)

#Support Vector Machine Classification Model

classifier = svm.SVC()
classifier.fit(X, Y)  

prediction = classifier.predict([[190,70,43]])

print "Support Vector Machine:" + str(prediction)

#Stochastic Gradient Descent Classification Model

classifier = SGDClassifier(loss="hinge", penalty="l2")
classifier.fit(X, Y)  

prediction = classifier.predict([[190,70,43]])

print "Stochastic Gradient Descent:" + str(prediction)