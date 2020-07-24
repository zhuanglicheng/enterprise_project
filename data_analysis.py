import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np

# 参考地址：https://blog.csdn.net/sumaoyan1787/article/details/91855340
data = pd.read_csv("housing.csv")
# ---------------------------------------------数据获取
# print(data.info())
# print(data.head())
# # 查看ocean_proximity这列有几个类别
# vc = data['ocean_proximity'].value_counts()
# print(vc)
'''
count—总行数

mean—平均值

min—最小值

max—最大值

std—标准差，用来测量数值的离散程度

25%、50%和75%--百分位数，表示一组观测值中给定百分比的观测值都低于该值。
'''
# print(data.describe())


# data.hist(bins=50,figsize=(20,15))
# plt.show()


# ---------------------------------------------数据探索
# 散点图
# data.plot(kind='scatter', x='longitude', y='latitude')
# data.plot(kind='scatter', x='longitude', y='latitude',alpha=0.1)
# plt.show()


# data.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,s=data["population"]/100,
#           label="population",c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)
# plt.legend()
# plt.show()

# 皮尔逊系数
# corr_matrix = data.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))
# 系统推荐：协同过滤算法
#
# attributes = ["median_house_value","median_income","total_rooms","data_median_age"]
# scatter_matrix(data[attributes],figsize=(12,8))
# plt.show()

# data.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1)
# plt.show()

# data["rooms_per_household"] = data["total_rooms"]/data["households"]
# data["bedrooms_per_room"] = data["total_bedrooms"]/data["total_rooms"]
# data["population_per_household"] = data["population"]/data["households"]
# corr_matrix = data.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

# ----------------------------4、数据准备------------------------------
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,LabelBinarizer,MinMaxScaler
# 创建对象
imputer = SimpleImputer(strategy="median")
#axis使用0值表示沿着每一列或行标签\索引值向下执行方法
#axis使用1值表示沿着每一行或者列标签模向执行对应的方法
housing_num = data.drop("ocean_proximity",axis=1)
#计算每个属性的中位数值
imputer.fit(housing_num)
#将缺失值替换成中位数值,返回包含转换后特征的Numpy数组
X = imputer.transform(housing_num)
# print(X)
encoder = LabelBinarizer()
housing_cat = data["ocean_proximity"]
housing_cat_1hot = encoder.fit_transform(housing_cat)
# print(housing_cat_1hot)
data0 = np.hstack((X,housing_cat_1hot))
columns = ['longitude','latitude','housing_median_age',
                                      'total_rooms','total_bedrooms','population',
                                      'households','median_income','median_house_value','OCEAN','INLAND','NEAR BAY','NEAR OCEAN']
data_df = pd.DataFrame(data0,columns=columns)
# print(data_df)
# 同比缩放
scaler = MinMaxScaler(feature_range=(0, 1))  #自动将dtype转换成float64
scaled = scaler.fit_transform(data_df)
scaled_df =pd.DataFrame(scaled,columns=columns)
# print(scaled_df,'\n')
#
# # 标准化
# inv_a = scaler.inverse_transform(scaled)
# inv_a_df =pd.DataFrame(inv_a,columns=columns)
# print('a-inversed:\n',inv_a_df)

# 数据集拆分:训练集和测试集
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(scaled_df, test_size=0.2, random_state=42)
print(len(train_set), "train +", len(test_set), "test")


# --------------------------------5、选择和训练模型--------------------
# 线性回归模型
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(train_set.drop('median_house_value',axis=1), train_set['median_house_value'])
print("Predictions:", lin_reg.predict(train_set.drop('median_house_value',axis=1)))
print("Labels:", train_set['median_house_value'])

# from sklearn.metrics import mean_squared_error
# housing_predictions = lin_reg.predict(train_set.drop("median_house_value",axis=1))
# lin_mse = mean_squared_error(train_set["median_house_value"], housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse)

# 决策树模型 rmse=0过拟合
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(train_set.drop('median_house_value',axis=1), train_set['median_house_value'])
print("Predictions:", tree_reg.predict(train_set.drop('median_house_value',axis=1)))
print("Labels:", train_set['median_house_value'])

# 随机森林
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(train_set.drop('median_house_value',axis=1), train_set['median_house_value'])
print("Predictions:", forest_reg.predict(train_set.drop('median_house_value',axis=1)))
print("Labels:", train_set['median_house_value'])