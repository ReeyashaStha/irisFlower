
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pickle

iris=datasets.load_iris()
X=iris.data
y=iris.target
#test train 
x_train,x_test,y_train,y_test=train_test_split(X,y)
log_reg=LogisticRegression()
dec_tree=DecisionTreeClassifier()
#fit data into algorithm
log_reg=log_reg.fit(x_train,y_train)
dec_tree=dec_tree.fit(x_train,y_train)
#pickle file 
pickle.dump(log_reg,open('log_model.pkl','wb'))
pickle.dump(dec_tree,open('dec_tree_model.pkl','wb'))