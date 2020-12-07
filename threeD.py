import numpy as np # pylint: disable=import-error
import pandas as pds # pylint: disable=import-error
from hmmlearn import hmm # pylint: disable=import-error
from sklearn.preprocessing import MinMaxScaler # pylint: disable=import-error
from sklearn.linear_model import LinearRegression # pylint: disable=import-error
from sklearn.svm import SVR # pylint: disable=import-error
from sklearn.kernel_ridge import KernelRidge # pylint: disable=import-error
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=import-error
import matplotlib.pyplot as plt # pylint: disable=import-error
import data_processing
pds.set_option('display.max_columns', None)

reshape_value = 261 #255
window_size = 8 #14
state = 'New York'

state_google, state_nytime = data_processing.get_df(state)

state_nytime_np = state_nytime.to_numpy()

scaler = MinMaxScaler()

# retail_norm = scaler.fit_transform((state_google['retail_and_recreation_percent_change_from_baseline'].to_numpy()).reshape(-1, 1))
# grocery_norm = scaler.fit_transform((state_google['grocery_and_pharmacy_percent_change_from_baseline'].to_numpy()).reshape(-1, 1))
# parks_norm = scaler.fit_transform((state_google['grocery_and_pharmacy_percent_change_from_baseline'].to_numpy()).reshape(-1, 1))
# transit_norm = scaler.fit_transform((state_google['grocery_and_pharmacy_percent_change_from_baseline'].to_numpy()).reshape(-1, 1))
# work_norm = scaler.fit_transform((state_google['grocery_and_pharmacy_percent_change_from_baseline'].to_numpy()).reshape(-1, 1))
# residential_norm = scaler.fit_transform((state_google['grocery_and_pharmacy_percent_change_from_baseline'].to_numpy()).reshape(-1, 1))
# normalized_info = np.concatenate((retail_norm, grocery_norm, parks_norm, transit_norm, work_norm, state_nytime_np), axis=1)
# retail_norm = normalize((state_google['retail_and_recreation_percent_change_from_baseline'].to_numpy()).reshape(-1, 1))
# print(normalized_info)
# print(normalized_info.shape)
# print(state_nytime_np.shape)

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

# print(indices.shape)
# print(state_nytime_np.shape)
# print(comm_move_data.shape)
print(windowed_move_data.shape)

data = np.concatenate((indices[:reshape_value], state_nytime_np[:reshape_value], windowed_move_data), axis=1)
# print(new_data.shape)
# data = np.concatenate((indices, state_nytime_np, using_var), axis=1)

# train_data = data[1:].reshape(-1,3,1)[::2].reshape(-1,3)
# test_data = data[:263].reshape(-1,3,1)[::2].reshape(-1,3)

train_data = data[1:].reshape(-1,((window_size*6)+2),1)[::2].reshape(-1,((window_size*6)+2))
# train_data = train_data[1:].reshape(-1,44,1)[::2].reshape(-1,44) #do it again (25%)
# train_data = train_data[1:].reshape(-1,44,1)[::2].reshape(-1,44) #do it again again (12.5%)
test_data = data[:reshape_value].reshape(-1,((window_size*6)+2),1)[::2].reshape(-1,((window_size*6)+2))

idx_OUT_columns = [1]
idx_IN_columns = [i for i in range(np.shape(data)[1]) if i not in idx_OUT_columns]

svm_model = SVR(kernel='linear').fit(train_data[:,idx_IN_columns],train_data[:,1])
svm_predict = svm_model.predict(test_data[:,idx_IN_columns])
lr_model = LinearRegression().fit(train_data[:,idx_IN_columns],train_data[:,1])
lr_predict = lr_model.predict(test_data[:,idx_IN_columns])
kr_model = KernelRidge(kernel='linear', alpha=0.9).fit(train_data[:,idx_IN_columns],train_data[:,1])
kr_predict = kr_model.predict(test_data[:,idx_IN_columns])
print("SVM Score: " + str(svm_model.score(test_data[:,idx_IN_columns], test_data[:,1])))
print("LR Score: " + str(lr_model.score(test_data[:,idx_IN_columns], test_data[:,1])))
print("KR Score: " + str(kr_model.score(test_data[:,idx_IN_columns], test_data[:,1])))

fig = plt.figure()
ax = Axes3D(fig)
indices = indices.reshape(1, -1)
reta = reta.reshape(1, -1)
state_nytime_np = state_nytime_np.reshape(1, -1)
ax.scatter(test_data[:,0], test_data[:,2], test_data[:,1], marker='o')
plt.plot(test_data[:,0], test_data[:,2], svm_predict, color='red')
plt.plot(test_data[:,0], test_data[:,2], lr_predict, color='green')
plt.plot(test_data[:,0], test_data[:,2], kr_predict, color='darkviolet')

# ax.text2D(0.05, 0.95, 'Linear Regression, Kernel Ridge, and SVR over 7 Day Time Series')
ax.set_xlabel('Date from March 7, 2020')
ax.set_zlabel('New Cases')
ax.set_ylabel('Retail Percent Change from Baseline')
ax.set_ylim3d(-75, 75)

plt.show()

###############################
#
# X = how much does society care about COVID (very, somewhat, little)
# starting = {'very': 0.1, 'somewhat': 0.1, 'little': 0.8} #baseline back in February with little to no impact on daily life
# observations = retail, grocery, parks, transit, workplaces, residential, num cases
#
###############################