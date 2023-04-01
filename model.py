import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from lazypredict.Supervised import LazyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.svm import SVC
import math
data = pd.read_csv("StudentScore.xls")
target="math score"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2    )

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

norm_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("scaler", OneHotEncoder(handle_unknown="ignore"))
])
education_level = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree",
                   "master's degree"]
gender = x["gender"].unique()
lunch = x["lunch"].unique()
test_course = x["test preparation course"].unique()

ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", OrdinalEncoder(categories=[education_level, gender, lunch, test_course]))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, ["reading score", "writing score"]),
    ("norm_feature", norm_transformer, ["race/ethnicity"]),
    ("ord_feature", norm_transformer, ["parental level of education", "gender", "lunch", "test preparation course"])
])
cls = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", SVC)
])
param_grid = {
    "classifier__C":[math.pow(10, -3), math.pow(10, -2), math.pow(10, -1), math.pow(10, 0),
                     math.pow(10, 1), math.pow(10, 2), math.pow(10, 3)],
    "classifier__kernel":['linear', 'poly', 'sigmoid', 'rbf'],
}
cls_grid_search = GridSearchCV(param_grid=param_grid, verbose=0, cv=5)
cls_grid_search.fit(x_train, y_train)
print(cls_grid_search.best_params_)
# x_train = preprocessor.fit_transform(x_train)
# x_test = preprocessor.transform(x_test)
reg = LinearRegression()
reg.fit(x_train, y_train)
y_predicted = reg.predict(x_test)
print("R2 {}".format(r2_score(y_test, y_predicted)))
print("MAE {}".format(mean_absolute_error(y_test, y_predicted)))




