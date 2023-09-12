import numpy as np import pandas as pd import seaborn as sns import
matplotlib.pyplot as plt %matplotlib inline from sklearn.linear_model
import LinearRegression from sklearn.model_selection import
train_test_split from sklearn.metrics import mean_squared_error,
r2_score from scipy import stats

# read the data

day = pd.read_csv(r"I:`\Metterial`{=tex}`\Assignment 13`{=tex}
sept`\day`{=tex}.csv") day.head()

# Check the descriptive information

day.info()

day.describe()

# Check the shape of df

print(day.shape)

# data set found 730 rows and 16 colnms

# creating dummy variables

## We will create DUMMY variables for 4 categorical variables 'mnth', 'weekday', 'season' & 'weathersit'.

# Check the datatypes before convertion

day.info()

# Convert to 'category' data type

day\['season'\]=bike_new\['season'\].astype('category')
day\['weathersit'\]=bike_new\['weathersit'\].astype('category')
day\['mnth'\]=bike_new\['mnth'\].astype('category')
day\['weekday'\]=bike_new\['weekday'\].astype('category')

day.info()

# This code does 3 things:

# 1) Create Dummy variable

# 2) Drop original variable for which the dummy was created

# 3) Drop first dummy variable for each set of dummies created.

day = pd.get_dummies(day, drop_first=True) day.info()

from sklearn.model_selection import train_test_split

# We should specify 'random_state' so that the train and test data set always have the same rows, respectively

np.random.seed(0) df_train, df_test = train_test_split(day, train_size =
0.70, test_size = 0.30, random_state = 333)

#### Verify the info and shape of the dataframes after split:

df_train.info()

df_train.shape

df_test.info()

df_test.shape

EXPLORATORY DATA ANALYSIS

#We need to perform the EDA on TRAINING (df_train) Dataset. #Visualising
Numeric Variables #Let's make a pairplot of all the numeric variables.

df_train.info()

df_train.columns

# Create a new dataframe of only numeric variables:

day=df_train\[\[ 'temp', 'atemp', 'hum', 'windspeed','cnt'\]\]

sns.pairplot(day, diag_kind='kde') plt.show()

# Sample code assumes you have your dataset and variables ready

# X represents your independent variables

# y represents your dependent variable

# Step 1: Split the data into a training set and a test set

X_train, X_test, y_train, y_test = train_test_split(day, day,
test_size=0.2, random_state=42)

# Step 2: Build a linear regression model

model = LinearRegression() model.fit(X_train, y_train)

# Step 3: Make predictions on the test set

y_pred = model.predict(X_test)

# Step 4: Residual analysis

# 4.1 Linearity

# Check linearity using a scatterplot of predicted vs. actual values

plt.figure(figsize=(8, 6)) plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values") plt.ylabel("Predicted Values")
plt.title("Linearity Check") plt.show()

# 4.2 Independence of Residuals

# Plot residuals vs. predicted values

residuals = y_test - y_pred plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals) plt.xlabel("Predicted Values")
plt.ylabel("Residuals") plt.title("Independence of Residuals Check")
plt.show()

df_test.head()

df_test.describe()

# Let's check the correlation coefficients to see which variables are highly correlated. Note:

# here we are considering only those variables (dataframe: bike_new) that were chosen for analysis

plt.figure(figsize = (25,20)) sns.heatmap(day.corr(), annot = True,
cmap="RdBu") plt.show()
