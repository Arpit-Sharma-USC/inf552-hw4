import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC,SVC
import seaborn as sns
from sklearn.metrics import accuracy_score

colNames=['variance','skewness','curtosis','entropy','class']
temp_dataframe = pd.read_csv("/media/ghost/Games And Images/INF552/hw4/data_banknote_authentication.csv",header=None, names=colNames)

temp_dataframe.to_csv('/media/ghost/Games And Images/INF552/hw4/data_banknote_authentication_with names.csv',index=False)
dataset=temp_dataframe.values

print(dataset.shape)
print("Shuffling...")

# shuffle the data
shuf_ds= shuffle(dataset, random_state=0)

print(shuf_ds.shape)

# abstract the test data

X_test=shuf_ds[:472,:4]
Y_test=shuf_ds[:472,4:]

# rest is train data
X_train_parent=shuf_ds[472:,:4]
Y_train_parent=shuf_ds[472:,4:]

counter=0

# hyperparameter setup
cv = KFold(10)
Cs=np.linspace(0.01,5,50)
tuned_parameters = [{'C': Cs}]
all_error_list=[]
all_errors=[]
x_axis_list=[]
# iterative training(online learning) for 90 SVMs

for i in range(1,91):
    X_train=X_train_parent[:10*i,:]
    Y_train=Y_train_parent[:10*i,:]
    print(X_train.shape)
    counter+=1
    # do the training and test error here calculation

    clf = GridSearchCV(LinearSVC(penalty='l1', dual=False), tuned_parameters, cv=cv, refit=True, n_jobs=2)

    clf.fit(X_train, Y_train.ravel())

    scores = clf.cv_results_['mean_test_score']
    scores_std = clf.cv_results_['std_test_score']

    #####optional plotting to visualize the best l1-penalty value
    #
    # plt.figure().set_size_inches(8, 6)
    # plt.semilogx(Cs, scores)
    # plt.show()

    print("Best L1-penalty parameter is ", clf.best_params_)
    print("Best Error-Rate with this parameter is", clf.best_score_)

    preds=clf.predict(X_test)
    # print(preds)
    test_score=accuracy_score(Y_test, preds)
    print("Test Score:",test_score)
    test_error=1-test_score
    all_error_list.append(test_error)
    all_errors=np.array(all_error_list)
    x_axis_list.append(i)

print(all_errors)


x_axis=np.array(x_axis_list)


print("\n")
print(counter)

plt.title('Test-errors for 90 SVMs- Passive Learning')
plt.xlabel('Number of SVMs')
plt.ylabel('Test Error')
plt.plot(x_axis,all_errors)
plt.show()




