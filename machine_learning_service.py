from flask_app import application
from services.data_processing.data_preparation import data_preparation, load_national_gdp
from services.data_processing.feature_engineering import feature_engineering
from services.model_training.model_selection import model_selection
from services.model_training.prediction import prediction

### orchestrate the entire service

try:
    # 1. prepare data
    quarterly = data_preparation() 
    # or using feature added data 
    #quarterly = feature_engineering()
    gdp_df = load_national_gdp()
    df = quarterly.join(gdp_df)

    # 2. model select
    X = df[df['national_gdp'].notnull()].drop('national_gdp', axis=1).values
    y = df.loc[df['national_gdp'].notnull(), 'national_gdp'].values
    selected_models = model_selection(X, y)

    # 3. prediction
    result_df = prediction(df, selected_models)
    
except Exception as e:
    application.logger.error(str(e))