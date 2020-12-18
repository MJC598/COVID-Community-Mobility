import numpy as np # pylint: disable=import-error
import pandas as pds # pylint: disable=import-error
from hmmlearn import hmm # pylint: disable=import-error
from sklearn.preprocessing import MinMaxScaler # pylint: disable=import-error
from sklearn.linear_model import LinearRegression # pylint: disable=import-error
from sklearn.svm import SVR # pylint: disable=import-error
from sklearn.kernel_ridge import KernelRidge # pylint: disable=import-error
from sklearn.metrics import mean_squared_error # pylint: disable=import-error
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=import-error
import matplotlib.pyplot as plt # pylint: disable=import-error
import data_processing
pds.set_option('display.max_columns', None)

reshape_value = 257#261 #255
window_size = 8 #14
# state = 'New York'
states = ["Alaska", "Alabama", "Arkansas", "Arizona", "California", "Colorado", 
          "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", 
          "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", 
          "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", 
          "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", 
          "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", 
          "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", 
          "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", 
          "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]

state_scores = {state: [] for state in states}
for state in states:
    print(state)
    if state == 'Florida' or state == 'West Virginia':
        continue
    state_google, state_nytime = data_processing.get_df(state)

    state_nytime_np = state_nytime.to_numpy()

    scaler = MinMaxScaler()

    reta = (state_google['retail_and_recreation_percent_change_from_baseline'].to_numpy()).reshape(-1, 1)
    groc = (state_google['grocery_and_pharmacy_percent_change_from_baseline'].to_numpy()).reshape(-1, 1)
    park = (state_google['grocery_and_pharmacy_percent_change_from_baseline'].to_numpy()).reshape(-1, 1)
    tran = (state_google['grocery_and_pharmacy_percent_change_from_baseline'].to_numpy()).reshape(-1, 1)
    work = (state_google['grocery_and_pharmacy_percent_change_from_baseline'].to_numpy()).reshape(-1, 1)
    resi = (state_google['grocery_and_pharmacy_percent_change_from_baseline'].to_numpy()).reshape(-1, 1)

    comm_move_data = np.concatenate((reta, groc, park, tran, work, resi), axis=1)

    using_var = reta

    indices = np.array([i for i in range(263)]).reshape(-1, 1)

    windowed_move_data = data_processing.rolling_window(comm_move_data, window_size).reshape(reshape_value, (window_size * 6))

    data = np.concatenate((indices[:reshape_value], state_nytime_np[:reshape_value], windowed_move_data), axis=1)

    train_data = data[1:].reshape(-1,((window_size*6)+2),1)[::2].reshape(-1,((window_size*6)+2))
    test_data = data[:reshape_value].reshape(-1,((window_size*6)+2),1)[::2].reshape(-1,((window_size*6)+2))

    idx_OUT_columns = [1]
    idx_IN_columns = [i for i in range(np.shape(data)[1]) if i not in idx_OUT_columns]

    svm_model = SVR(kernel='linear').fit(train_data[:,idx_IN_columns],train_data[:,1])
    svm_predict = svm_model.predict(test_data[:,idx_IN_columns])
    lr_model = LinearRegression().fit(train_data[:,idx_IN_columns],train_data[:,1])
    lr_predict = lr_model.predict(test_data[:,idx_IN_columns])
    kr_model = KernelRidge(kernel='linear', alpha=0.9).fit(train_data[:,idx_IN_columns],train_data[:,1])
    kr_predict = kr_model.predict(test_data[:,idx_IN_columns])
    # print("SVM Score: " + str(svm_model.score(test_data[:,idx_IN_columns], test_data[:,1])))
    # print("SVM MSE: " + str(mean_squared_error(test_data[:,1], svm_predict)))
    # state_scores[state].append(svm_model.score(test_data[:,idx_IN_columns], test_data[:,1]))
    state_scores[state].append(mean_squared_error(test_data[:,1], svm_predict))
    # print("LR Score: " + str(lr_model.score(test_data[:,idx_IN_columns], test_data[:,1])))
    # print("LR MSE: " + str(mean_squared_error(test_data[:,1], lr_predict)))
    # state_scores[state].append(lr_model.score(test_data[:,idx_IN_columns], test_data[:,1]))
    state_scores[state].append(mean_squared_error(test_data[:,1], lr_predict))
    # print("KR Score: " + str(kr_model.score(test_data[:,idx_IN_columns], test_data[:,1])))
    # print("KR MSE: " + str(mean_squared_error(test_data[:,1], kr_predict)))
    # state_scores[state].append(kr_model.score(test_data[:,idx_IN_columns], test_data[:,1]))
    state_scores[state].append(mean_squared_error(test_data[:,1], kr_predict))

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # indices = indices.reshape(1, -1)
    # reta = reta.reshape(1, -1)
    # state_nytime_np = state_nytime_np.reshape(1, -1)
    # ax.scatter(test_data[:,0], test_data[:,2], test_data[:,1], marker='o')
    # plt.plot(test_data[:,0], test_data[:,2], svm_predict, color='red')
    # plt.plot(test_data[:,0], test_data[:,2], lr_predict, color='green')
    # plt.plot(test_data[:,0], test_data[:,2], kr_predict, color='darkviolet')

    # ax.set_xlabel('Date from March 7, 2020')
    # ax.set_zlabel('New Cases')
    # ax.set_ylabel('Retail Percent Change from Baseline')
    # ax.set_ylim3d(-75, 75)

    # plt.show()

for state in state_scores.keys():
    print(state + str(state_scores[state]))
# print(state_scores)
with open('r2.csv', 'w') as f:
    for key in state_scores.keys():
        if key == 'Florida' or key == 'West Virginia':
            continue
        else:
            f.write("{},{},{},{}\n".format(key, state_scores[key][0], state_scores[key][1], state_scores[key][2]))