import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


df = pd.read_csv('student_habits_performance.csv')

print(df.describe())

df.dropna(axis=0, inplace=True)

X = df.drop(columns=['student_id','exam_score'])
y = df['exam_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sklearn.set_config(transform_output="pandas")

encoder = ColumnTransformer(
    [
        ("onehot", OneHotEncoder(sparse_output=False, min_frequency=5, handle_unknown="infrequent_if_exist"), 
         ["gender", "part_time_job", "diet_quality", "parental_education_level", "internet_quality", "extracurricular_participation"]),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
    force_int_remainder_cols=False
)

pipe = Pipeline(
 [
 ("encoder", encoder),
 ("standardization", StandardScaler()),
 ("regressor", LinearRegression())
 ]
)

pipe.fit(X_train, y_train)

y_test_pred = pipe.predict(X_test)

mae = mean_absolute_error(y_test, y_test_pred)
mape = mean_absolute_percentage_error(y_test, y_test_pred)
print(f"""
MAE: {mae}
MAPE: {mape}
""")

joblib.dump(pipe, "modello_regressione.joblib")