from src.data_ingestion import data_ingestion
from src.data_preprocessing import data_preprocessing
from src.model_building import model_building

def main():
    df=data_ingestion()
    print (df.shape)
    X_train,X_test,y_train,y_test,preprocessor=data_preprocessing(df)
    model=model_building(X_train,X_test,y_train,y_test,preprocessor)
    print(model)

main()