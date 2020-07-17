import numpy as np
import pandas as pd
import scipy
import sklearn as sks
import seaborn as sns
import plotly
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder ,StandardScaler , MinMaxScaler
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import scipy
from sklearn.metrics import f1_score , log_loss , jaccard_score

import plotly.express as px

#loading the dataset
data = pd.read_csv('loan_train.csv')
print("Csv = \n", data.head())

#checking null values
print("Null Values Sum \n= " , data.isnull().sum())

#dropping useless column  Unnamed: 0, index
data.drop([ 'Unnamed: 0' , 'Unnamed: 0.1'  ],axis =1 ,  inplace=True )
print("After Dropping Columns \n" , data.head(20))

#since no null values we can proceed with eda

sns.boxplot(x='education' , y='age' , data = data)
plt.title('Plot of Education Level and Age ')
#plt.show()

sns.countplot(x='Gender' , data = data)
plt.title('Count Of Gender in Data')
#plt.show()

sns.distplot(data['Principal'])
plt.title('Principal Wise Distribution')
#plt.show()

#converting date time into acceptable format
print(data['due_date'].dtype)
print(data['effective_date'].dtype)
data['effective_date']=pd.to_datetime(data['effective_date'] , infer_datetime_format=True)
data['due_date']=pd.to_datetime(data['due_date'] , infer_datetime_format=True)

print(data['due_date'].dtype)
print(data['effective_date'].dtype)
print(data.head())

#Making  a new column due_days which the elapsed days from loan approval to due date
lst=[]
for row in data.itertuples():
 x = row.due_date - row.effective_date
 lst.append(x.days)

data['due_days'] = lst
print("New Column added = \n" , data['due_days'])


#converting categorical data to numerical type

#converting the target variable to numerical

print(data['loan_status'].unique())
data['loan_status'].replace(to_replace=['PAIDOFF','COLLECTION'], value=[0,1],inplace=True)
print("Target value replaced\n"  , data.head())

#doing the same to gender column
print(data['Gender'].unique())
data['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
print("Gender value replaced\n"  , data.head())

#doing one hot encoder for education column since it needs to be more expressive
# creating instance of one-hot-encoder
enc = OneHotEncoder()
# passing bridge-types-cat column (label encoded values of bridge_types)
enc_df = pd.DataFrame(enc.fit_transform(data[['education']]).toarray())
# merge with main df bridge_df on key values
data = data.join(enc_df)
print(data.head(20))
print(data.columns)

#now we can fit some models to the data

#doing Test/Train split on the data

#defining X and Y
X = data.drop([  'loan_status' , 'effective_date' , 'due_date' , 'education' ],axis =1 )
Y= data['loan_status']
print('X = \n' , X)
print("Y= \n" , Y)

#Scaling the Input Data Since it has multiple Units
Scaler = MinMaxScaler()
X = Scaler.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5, shuffle=True)
###################################################################################################


#Logistic Regression
logi = LogisticRegression(max_iter=200 , class_weight='balanced')
logi.fit(X_train , Y_train)

param_logi = { 'C' :[100,10,1,0.01,0.001 , 0.0001,0.00001 ,0.000001 ],
               'solver' : ['lbfgs'  ,'liblinear' ]}

#Using GridSearchCV to find the best HyperParameter
grid_1=GridSearchCV(estimator=logi , param_grid=param_logi  ,verbose=1 , scoring='roc_auc' )
grid_1.fit(X_train , Y_train)
logi_pred=grid_1.predict(X_test)
log_prob = grid_1.predict_proba(X_test)
#Score of Logistic Model
print()
print()
print("F1 Sore of the Logistic Regression model = " , f1_score(Y_test , logi_pred , average='weighted'))
print("Log Loss Score of the Logistic Regression model =  " ,log_loss(Y_test , log_prob ))
print("Jacard Score of the Logistic Regression model =  " ,  jaccard_score(Y_test , logi_pred, average = 'weighted'  ))
print()
print()

#############################################################################################################


#KNN Model
#now we use a KNN model for the same

#Now to find the best model we use GridSearchCV

grid_params = {
    'n_neighbors'  : [2,3,4,5,6,7,8,9,10] ,
     'weights' : ['uniform' ,'distance'] ,
}

grid = GridSearchCV ( estimator=KNeighborsClassifier() , param_grid=grid_params,verbose =1 ,scoring='roc_auc')
grid_results = grid.fit(X_train,Y_train)

knn_pred =grid.predict(X_test)
#Score of Knn Model
print()
print()
print("F1 Sore of the KNN Classification model = " , f1_score(Y_test , knn_pred , average='weighted'))
print("Jacard Score of the KNN Classification model =  " ,  jaccard_score(Y_test , knn_pred , average = 'weighted' ))
print("The Best K Value Has been Found as " , grid.best_estimator_)
print()



#################################################################################################


#Descion Tree

# We will again use GridSearchCV to find the best model

params = { 'max_leaf_nodes': list(range(2, 20)), 'min_samples_split': [2, 3, 4] ,
           'max_depth'  :  [2,4,6,8]}

grid_2 =GridSearchCV(estimator=DecisionTreeClassifier(min_samples_leaf=1), param_grid=params ,verbose=1 ,scoring='roc_auc')

grid_2.fit(X_train , Y_train)

tree_pred = grid_2.predict(X_test)

#Score of Descion Tree Model
print()
print()
print("F1 Sore of the Descion Tree model = " , f1_score(Y_test , tree_pred , average='weighted'))
print("Jacard Score of the Descion Tree model =  " ,  jaccard_score(Y_test , tree_pred, average = 'weighted' ))
print()
print()
# Plotting The Descion Tree
tree.plot_tree(grid_2.best_estimator_)
#plt.show()
#######################################################################################


#SVM Model
svm = svm.SVC(probability=True)

pram_svm = { 'C': [1,0.01,0.001 , 0.0001,0.00001 ,0.000001],
            'kernel' : ['rbf' , 'poly' ]}
grid_3=GridSearchCV(estimator=svm , param_grid=pram_svm ,  verbose = 1 ,scoring='roc_auc')
grid_3.fit(X_train , Y_train)
svm_pred=grid_3.predict(X_test)
#Score of Support Vector Classifier Model
print()
print()
print("F1 Sore of the SVM model = " , f1_score(Y_test , svm_pred , average='weighted'))
print("Jacard Score of the SVM model =  " ,  jaccard_score(Y_test , svm_pred , average = 'weighted' ))
print()
print()

########################################## Testing All The Models ##########################
data_2 = pd.read_csv('loan_test.csv')
data_2.drop( 'Unnamed: 0' , axis =1 , inplace=True )
data_2.reset_index()
data_2.drop( 'Unnamed: 0.1' , axis =1 , inplace=True )

#Applying the necessary transfomation

data_2['effective_date']=pd.to_datetime(data_2['effective_date'] , infer_datetime_format=True)
data_2['due_date']=pd.to_datetime(data_2['due_date'] , infer_datetime_format=True)
lst=[]
for row in data_2.itertuples():
 x = row.due_date - row.effective_date
 lst.append(x.days)
data_2['due_days'] = lst
data_2['loan_status'].replace(to_replace=['PAIDOFF','COLLECTION'], value=[0,1],inplace=True)
data_2['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
enc = OneHotEncoder()
enc_df = pd.DataFrame(enc.fit_transform(data_2[['education']]).toarray())
data_2 = data_2.join(enc_df)
X = data_2.drop([ 'loan_status' , 'effective_date' , 'due_date' , 'education' ],axis =1 )
Y= data_2['loan_status']
Scaler = StandardScaler()
X = Scaler.fit_transform(X)


#To Filter Warnings in python
import warnings
warnings.filterwarnings('always')

logi_pred=grid_1.predict(X)
log_prob1 = grid_1.predict_proba(X)

n_lst=[]
j_lst=[]
f_lst=[]
l_lst=[]

print()
print()
print("F1 Sore of the Logistic Regression model = " ,f1_score(Y , logi_pred , average='weighted',labels=np.unique(logi_pred)))
print("Log Loss Score of the Logistic Regression model =  " ,log_loss(Y , log_prob1 ))
print("Jacard Score of the Logistic Regression model =  " ,  jaccard_score(Y , logi_pred , average = 'weighted'  ))
print()
print()
f_lst.append(f1_score(Y , logi_pred , average='weighted', labels=np.unique(logi_pred)))
l_lst.append(log_loss(Y , log_prob1 ))
j_lst.append(jaccard_score(Y , logi_pred , average = 'weighted'  ))
n_lst.append('LogisticRegression')

#Score of KNN Model
knn_pred =grid.predict(X)
knn_prob1 = grid.predict_proba(X)

print()
print()
print("F1 Sore of the KNN Classification model = " , f1_score(Y , knn_pred , average='weighted', labels=np.unique(knn_pred)))
print("Jacard Score of the KNN Classification model =  " ,  jaccard_score(Y , knn_pred , average = 'weighted'))
print("The Best K Value Has been Found as " , grid.best_estimator_)
print()
print()
f_lst.append(f1_score(Y , knn_pred , average='weighted'  , labels=np.unique(knn_pred)))

j_lst.append(jaccard_score(Y , knn_pred  , average = 'weighted' ))
l_lst.append(log_loss(Y , knn_prob1 ))
n_lst.append('KNN')


#Score of Descion Tree Model
tree_pred = grid_2.predict(X)
tree_pred1 = grid_2.predict_proba(X)
print()
print()
print("F1 Sore of Descion Tree model = " , f1_score(Y , tree_pred , average='weighted' , labels=np.unique(tree_pred)))
print("Jacard Score of Descion Tree model =  " ,  jaccard_score(Y , tree_pred  , average = 'weighted'))
print()
print()
f_lst.append(f1_score(Y , tree_pred , average='weighted' , labels=np.unique(tree_pred)))
j_lst.append(jaccard_score(Y , tree_pred , average = 'weighted' ))
l_lst.append(log_loss(Y , tree_pred1 ))
n_lst.append('Descion Tree')

#Score of Support Vector Classifier Model
svm_pred=grid_3.predict(X)
svm_pred1 = grid_3.predict_proba(X)
print()
print()
print("F1 Sore of the SVM model = " , f1_score(Y , svm_pred , average='weighted' , labels=np.unique(svm_pred)))
print("Jacard Score of the SVM model =  " ,  jaccard_score(Y , svm_pred , average = 'weighted'))
print()
print()
f_lst.append(f1_score(Y , svm_pred , average='weighted' , labels=np.unique(svm_pred)))
j_lst.append(jaccard_score(Y , svm_pred , average = 'weighted' ))
l_lst.append(log_loss(Y , svm_pred1 ))
n_lst.append('SVM')

# Final Report
Report=pd.DataFrame(columns=['Algorithm' ,'Jaccard','F1-Score','LogLoss'])
Report['Algorithm'] = n_lst
Report['Jaccard'] = j_lst
Report['F1-Score'] = f_lst
Report['LogLoss'] = l_lst
print("Report\n"  , Report)