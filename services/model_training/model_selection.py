from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import StackingRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
from utils.aws.s3 import upload_file_to_s3, download_file_from_s3
import pickle

N_MODELS = 3

def find_best_parameters(X_train, y_train, random_state = 19):
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=random_state)

    #Decision Tree Regressor
    dtr = DecisionTreeRegressor(random_state=random_state)
    params_dtr = {
        'criterion' : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
    }
    dtr_gridcv = GridSearchCV(dtr, param_grid=params_dtr, cv=cv, n_jobs=-1, verbose=3)
    dtr_gridcv.fit(X_train,y_train)

    #Bagging Regressor
    bg = BaggingRegressor()
    params_bg = {
        'n_estimators' : [200, 500, 750, 1000],
        'base_estimator': [
            DecisionTreeRegressor(random_state=random_state, max_depth=3)
        ],
        'max_samples': [0.5, 0.7, 0.9]
    }
    bagging_gridcv = GridSearchCV(estimator=bg, param_grid=params_bg, return_train_score=True, cv=cv, n_jobs =-1, verbose=3)
    bagging_gridcv.fit(X_train,y_train)
    print(f'Best Params for BaggingRegressor: {bagging_gridcv.best_params_}')

    #Ada Boost Regressor
    ada = AdaBoostRegressor(random_state=random_state)
    params_ada = {
        'learning_rate' : np.linspace(0.1,1,10),
        'n_estimators' : [100,200,500,700,1000],
        'loss' : ['linear', 'square']
    }
    ada_gridcv = GridSearchCV(ada, param_grid=params_ada, cv=cv, n_jobs=-1, verbose=3)
    ada_gridcv.fit(X_train, y_train)
    print(f'The Best Parameters for AdaBoostRegressor: {ada_gridcv.best_params_}')

    #XGBoost Regressor
    xgr = XGBRegressor()
    params_xgr= {
        'n_estimators' : [100, 500, 750, 1000],
        'objective' : ['reg:squarederror'],
        'learning_rate': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    xgr_gridcv = GridSearchCV(xgr, param_grid=params_xgr, cv=cv, n_jobs=-1, verbose=3)
    xgr_gridcv.fit(X_train, y_train)
    print(f'The Best Parameters for XGBRegressor: {xgr_gridcv.best_params_}')

    #LightGBM Regressor
    lgr = LGBMRegressor()
    params_lgr = {
        'learning_rate' : [0.01, 0.05, 0.1, 0.3, 0.5, 0.7],
        'n_estimators' : [100, 500, 700]
    }
    lgr_gridcv = GridSearchCV(lgr, param_grid=params_lgr, cv=cv, n_jobs=-1, verbose=3)
    lgr_gridcv.fit(X_train, y_train)
    print(f'The Best Parameters for LGBMRegressor: {lgr_gridcv.best_params_}')

    #CatBoost Regressor
    cbc = CatBoostRegressor(logging_level='Silent')
    params_cbc= {
        'n_estimators' : [100, 500, 750, 1000],
        'learning_rate': [0.1, 0.3, 0.5, 0.7, 0.9],
        'max_depth': [3,4,5]
    }
    gscv = GridSearchCV (estimator = cbc, param_grid = params_cbc, scoring ='accuracy', cv = cv)
    gscv.fit(X,y)
    return bagging_gridcv, ada_gridcv, xgr_gridcv, lgr_gridcv, gscv
    
def parameter_store():

    bagging_gridcv, ada_gridcv, xgr_gridcv, lgr_gridcv, gscv = find_best_parameters()

    ## for the voting and stacking regressor
    models_list = [
        (
            'BaggingRegressor', BaggingRegressor(
                n_estimators = bagging_gridcv.best_params_['n_estimators'],
                base_estimator = DecisionTreeRegressor(random_state=random_state, max_depth=3),
                max_samples = bagging_gridcv.best_params_['max_samples']
            )
        ),
        (
            'AdaBoostRegressor', AdaBoostRegressor(
                learning_rate = ada_gridcv.best_params_['learning_rate'],
                n_estimators = ada_gridcv.best_params_['n_estimators'],
                loss = ada_gridcv.best_params_['loss']
            )
        ),
        (
            'XGBRegressor', XGBRegressor(
                n_estimators = xgr_gridcv.best_params_['n_estimators'],
                objective = xgr_gridcv.best_params_['objective'],
                learning_rate = xgr_gridcv.best_params_['learning_rate']
            )
        ),
        (
            'LGBMRegressor', LGBMRegressor(
                learning_rate = lgr_gridcv.best_params_['learning_rate'],
                n_estimators = lgr_gridcv.best_params_['n_estimators']
            )
        ),
        ('CatBoostRegressor', CatBoostRegressor(
            n_estimators = gscv.best_params_['n_estimators'],
            learning_rate = gscv.best_params_['learning_rate'],
            max_depth = gscv.best_params_['max_depth'],
            logging_level='Silent'
        )
        ),
        (
            'LinearRegressor', LinearRegression()
        )
    ]

    models_cv = models_list + [
        (
        'VotingRegressor', VotingRegressor(estimators=models_list
                                        )
        ),
        (
            'StackingRegressor', StackingRegressor(estimators = models_list, final_estimator= LinearRegression(), cv=5
                                                )
        )]

    with open('models_cv.pkl', 'wb') as f:
        pickle.dump(models_cv, f)
    
    upload_file_to_s3('models_cv.pkl', 'machine_learning/models/models_cv.pkl')
    os.remove('models_cv.pkl')

def model_selection(X, y, random_state= 17):
    download_file_from_s3('models_cv.pkl', 'machine_learning/models/models_cv.pkl')
    with open('models_cv.pkl', 'rb') as f:
        models_cv = pickle.loads(f)
    os.remove('models_cv.pkl')
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    df_scores_regression = pd.DataFrame()
    accuracy_scores_regression = []
    accuracy_scores_true = []

    for name, model in models_cv:
        model_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        model.fit(X_train,y_train)
        accuracy_scores_regression.append(MSE(y_test, model.predict(X_test))**(1/2))
        accuracy_scores_true.append(MSE(y, model.predict(X))**(1/2))
        df_scores_regression[name] = model_scores

    model_result = pd.DataFrame(df_scores_regression.mean(), columns=['AvgModelScore'])
    model_result['AccuracyRegression'] = accuracy_scores_regression
    model_result['AccuracyRegression'] *= -1
    model_result['AccuracyTrue'] = accuracy_scores_true
    model_result['AccuracyTrue'] *= -1
    for col in model_result.columns:
        model_result[col] = model_result[col].rank()
    model_result['Total'] = model_result.sum(axis=1)
    model_result.sort_values('Total', ascending=False, inplace=True)
    return model_result.head(N_MODELS).index