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
    print("Best Score with this paramteter is", clf.best_score_)

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

passive_errors=all_errors



colNames=['variance','skewness','curtosis','entropy','class']
temp_dataframe = pd.read_csv("/media/ghost/Games And Images/INF552/hw4/data_banknote_authentication.csv",header=None, names=colNames)

temp_dataframe.to_csv('/media/ghost/Games And Images/INF552/hw4/data_banknote_authentication_with names.csv',index=False)
dataset=temp_dataframe.values

print(dataset.shape)
print("Shuffling...")

# shuffle the data
shuf_ds= shuffle(dataset, random_state=21)

print(shuf_ds.shape)

# abstract the test data

X_test=shuf_ds[:472,:4]
Y_test=shuf_ds[:472,4:]

# rest is train data
X_train_parent=shuf_ds[472:482,:4]
Y_train_parent=shuf_ds[472:482,4:]

X_rest=shuf_ds[482:,:4]
Y_rest=shuf_ds[482:,4:]

counter = 0

# hyperparameter setup
cv = KFold(10)
Cs = np.linspace(0.01, 5, 50)
tuned_parameters = [{'C': Cs}]

all_error_list = []
all_errors = []
x_axis_list_active = []


X_train = X_train_parent[:,:]
Y_train = Y_train_parent[:,:]

# iterative training(online learning) for 90 SVMs

for i in range(1, 90):


    print('\n##### Iteration:',i,' with Samples in Train Data:',10*i,'#####')
    counter += 1
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
    print("Best Score with this paramteter is", clf.best_score_)


    preds = clf.predict(X_test)
    test_score = accuracy_score(Y_test, preds)
    print("Test Score:", test_score)
    test_error = 1 - test_score
    all_error_list.append(test_error)
    all_errors = np.array(all_error_list)
    x_axis_list_active.append(i)


    X_margin = clf.decision_function(X_rest)
    X_margin = np.abs(X_margin)

    index_for = np.arange(0, len(X_margin), 1)

    data = {'id': index_for, 'Margin_dist': X_margin}
    df = pd.DataFrame(data=data)
    df = df.sort_values(by='Margin_dist')

    print("top 10..")

    df_10=pd.DataFrame()
    df_10=df.iloc[:10]

    check_arr=X_train
    check_Y_arr=Y_train

    to_use_indices=[]
    temp_X=X_rest
    temp_Y=Y_rest
    for j in range(0,10):
        temp=np.int(df_10.values[j,1])
        to_use_indices.append(temp)
        check_arr=np.vstack((check_arr,np.array(X_rest[temp].reshape(1,4))))
        check_Y_arr = np.vstack((check_Y_arr, np.array(Y_rest[temp].reshape(1, 1))))

    # print(check_arr.shape)
    temp_X = np.delete(temp_X, to_use_indices, axis=0)
    temp_Y = np.delete(temp_Y, to_use_indices, axis=0)
    # print(temp_X)
    print(to_use_indices)

    # overwrite X_train and Y_train
    X_train=check_arr
    Y_train=check_Y_arr

    # remove from Rest

    # stopper=len(X_margin)

    # for j in range(0,stopper):
    #     if(j in to_use_indices):
    #         X_rest=np.delete(X_rest, j, axis=0)
    #         Y_rest=np.delete(Y_rest, j, axis=0)

    Y_rest=temp_Y
    X_rest=temp_X



print(all_errors)
active_errors=all_errors


x_axis = np.array(x_axis_list)
x_axis_active = np.array(x_axis_list_active)

plt.title('Average Test-errors for 90 SVMs- Passive vs Active Learning')
plt.xlabel('Number of SVMs')
plt.ylabel('Test Error')
plt.plot(x_axis_active, active_errors,'r',label="active learning")
plt.plot(x_axis, passive_errors,'b',label="passive learning")
plt.legend()
plt.show()
