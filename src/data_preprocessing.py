from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def data_preprocessing(df):

    # 1 Remove duplicates
    df = df.drop_duplicates()

    # 2 Split X and y
    X = df.drop(columns=["AnyHealthcare","NoDocbcCost","CholCheck","Diabetes_binary"])
    y = df["Diabetes_binary"]

    # 3 Train Test Split
    X_train,X_test,y_train,y_test = train_test_split(
        X,y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4 All columns are numeric
    numerical_col = X_train.columns

    # 5 Numerical pipeline
    Numerical_Pipeline = Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",MinMaxScaler())
    ])

    # 6 Column transformer
    preprocessor = ColumnTransformer(transformers=[
        ("num",Numerical_Pipeline,numerical_col)
    ])

    # Apply preprocessing
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # 7 Apply SMOTE
    sm = SMOTE(random_state=42)
    X_train,y_train = sm.fit_resample(X_train,y_train)

    return X_train,X_test,y_train,y_test,preprocessor