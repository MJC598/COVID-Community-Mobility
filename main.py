import numpy as np # pylint: disable=import-error
import pandas as pds # pylint: disable=import-error
from hmmlearn import hmm # pylint: disable=import-error
from sklearn.preprocessing import MinMaxScaler # pylint: disable=import-error
from sklearn.linear_model import LinearRegression # pylint: disable=import-error
from sklearn.svm import SVR # pylint: disable=import-error
from sklearn.kernel_ridge import KernelRidge # pylint: disable=import-error
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=import-error
import matplotlib.pyplot as plt # pylint: disable=import-error

pds.set_option('display.max_columns', None)

PATH_TO_GOOGLE_CSV = '/home/matt/data/covid_community_mobility/Global_Mobility_Report_11_27_20.csv'
PATH_TO_NYTIME_CSV = '/home/matt/data/covid_community_mobility/us-states.csv'

df_google = pds.read_csv(PATH_TO_GOOGLE_CSV, low_memory=False)
df_nytime = pds.read_csv(PATH_TO_NYTIME_CSV)
only_mo_google = (df_google.loc[df_google['sub_region_1'] == 'Missouri'].sort_values('date'))[['sub_region_1', 'sub_region_2', 'metro_area', 'date', 
    'retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline', 'parks_percent_change_from_baseline',
    'transit_stations_percent_change_from_baseline', 'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline']]
mo_google = ((only_mo_google.loc[pds.isna(only_mo_google['sub_region_2'])].sort_values('date'))[['date', 
    'retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline', 'parks_percent_change_from_baseline',
    'transit_stations_percent_change_from_baseline', 'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline']]
                .set_index('date')).loc['2020-03-07':]
mo_nytime = ((((df_nytime.loc[df_nytime['state'] == 'Missouri'].sort_values('date'))[['date', 'new_cases']])
                .set_index('date')).loc[:'2020-11-24'])
mo_nytime_np = mo_nytime.to_numpy()

scaler = MinMaxScaler()

# retail_norm = scaler.fit_transform((mo_google['retail_and_recreation_percent_change_from_baseline'].to_numpy()).reshape(-1, 1))
# grocery_norm = scaler.fit_transform((mo_google['grocery_and_pharmacy_percent_change_from_baseline'].to_numpy()).reshape(-1, 1))
# parks_norm = scaler.fit_transform((mo_google['grocery_and_pharmacy_percent_change_from_baseline'].to_numpy()).reshape(-1, 1))
# transit_norm = scaler.fit_transform((mo_google['grocery_and_pharmacy_percent_change_from_baseline'].to_numpy()).reshape(-1, 1))
# work_norm = scaler.fit_transform((mo_google['grocery_and_pharmacy_percent_change_from_baseline'].to_numpy()).reshape(-1, 1))
# residential_norm = scaler.fit_transform((mo_google['grocery_and_pharmacy_percent_change_from_baseline'].to_numpy()).reshape(-1, 1))
# normalized_info = np.concatenate((retail_norm, grocery_norm, parks_norm, transit_norm, work_norm, mo_nytime_np), axis=1)
# retail_norm = normalize((mo_google['retail_and_recreation_percent_change_from_baseline'].to_numpy()).reshape(-1, 1))
# print(normalized_info)
# print(normalized_info.shape)
# print(mo_nytime_np.shape)

reta = (mo_google['retail_and_recreation_percent_change_from_baseline'].to_numpy()).reshape(-1, 1)
groc = (mo_google['grocery_and_pharmacy_percent_change_from_baseline'].to_numpy()).reshape(-1, 1)
park = (mo_google['grocery_and_pharmacy_percent_change_from_baseline'].to_numpy()).reshape(-1, 1)
tran = (mo_google['grocery_and_pharmacy_percent_change_from_baseline'].to_numpy()).reshape(-1, 1)
work = (mo_google['grocery_and_pharmacy_percent_change_from_baseline'].to_numpy()).reshape(-1, 1)
resi = (mo_google['grocery_and_pharmacy_percent_change_from_baseline'].to_numpy()).reshape(-1, 1)

using_var = reta

indices = np.array([i for i in range(263)]).reshape(-1, 1)

data = np.concatenate((indices, mo_nytime_np, using_var), axis=1)

train_data = data[1:].reshape(-1,3,1)[::2].reshape(-1,3)
test_data = data[:263].reshape(-1,3,1)[::2].reshape(-1,3)

svm_model = SVR(kernel='linear').fit(train_data[:,[0,1]],train_data[:,2])
svm_predict = svm_model.predict(test_data[:,[0,1]])
lr_model = LinearRegression().fit(train_data[:,[0,1]],train_data[:,2])
lr_predict = lr_model.predict(test_data[:,[0,1]])
kr_model = KernelRidge(kernel='rbf', alpha=0.7).fit(train_data[:,[0,1]],train_data[:,2])
kr_predict = kr_model.predict(test_data[:,[0,1]])
print("SVM Score: " + str(svm_model.score(test_data[:,[0,1]], test_data[:,2])))
print("LR Score: " + str(lr_model.score(test_data[:,[0,1]], test_data[:,2])))
print("KR Score: " + str(kr_model.score(test_data[:,[0,1]], test_data[:,2])))

fig = plt.figure()
ax = Axes3D(fig)
indices = indices.reshape(1, -1)
reta = reta.reshape(1, -1)
mo_nytime_np = mo_nytime_np.reshape(1, -1)
ax.scatter(train_data[:,0], train_data[:,1], train_data[:,2], marker='o')
plt.plot(test_data[:,0], test_data[:,1], svm_predict, color='red')
plt.plot(test_data[:,0], test_data[:,1], lr_predict, color='green')
plt.plot(test_data[:,0], test_data[:,1], kr_predict, color='darkviolet')

ax.set_xlabel('Date from March 7, 2020')
ax.set_ylabel('New Cases')
ax.set_zlabel('Retail Percent Change from Baseline')
ax.set_zlim3d(-50, 50)

plt.show()

###############################
#
# X = how much does society care about COVID (very, somewhat, little)
# starting = {'very': 0.1, 'somewhat': 0.1, 'little': 0.8} #baseline back in February with little to no impact on daily life
# observations = retail, grocery, parks, transit, workplaces, residential, num cases
#
###############################