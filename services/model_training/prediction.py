from model_selection import model_selection
from statsmodels.tsa.api import SimpleExpSmoothing
import pickle

ALPHA = 0.7

def prediction(df):
    nan_length = len(df[df['national_gdp'].isnull()])
    for col in df.columns:
        length = len(df[pd.isnull(df[col])])
        if length > 0:
            print(col, length)
            if col == 'national_gdp':
                nan_length = length * -1

    result = df[['national_gdp']].iloc[int(len(df) * ALPHA):].copy()
    result.columns = ['Actual']

    # Benchmark 1: Naive
    result = result.join(df[['national_gdp']].shift(1).iloc[int(len(df) * ALPHA):])
    result.rename(columns={'national_gdp':'Benchmark_Naive'}, inplace=True)

    # Benchmark 2: Exponential Smoothing
    result['Benchmark_ES']= np.nan
    for i in range(int(len(df) * ALPHA), len(df)):
        x = df.iloc[:i]['national_gdp']
        fit = SimpleExpSmoothing(x, initialization_method="heuristic").fit(smoothing_level=0.6,optimized=False)
        value = fit.forecast(1).values[0]
        idx = fit.forecast(1).index[0].strftime('%Y-%m-%d')
        result.loc[idx, 'Benchmark_ES'] = value

    # load models
    download_file_from_s3('models_cv.pkl', 'machine_learning/models/models_cv.pkl')
    with open('models_cv.pkl', 'rb') as f:
        models_cv = pickle.loads(f)
    os.remove('models_cv.pkl')

    selected_models = model_selection()

    start_time = dt.now()
    size = int(len(df) * ALPHA)
    for name, model in models_cv:
        if name not in selected_models:
            continue
        y_pred = []
        for i in np.arange(size, len(df) + (nan_length + 1)):
            #print(gdp.index[i])
            X = df.iloc[:i].drop(['national_gdp'], axis=1).values
            y = df.iloc[:i]['national_gdp'].values
            model.fit(X,y)
            y_pred.append(model.predict(df.iloc[[i]].drop(['national_gdp'], axis=1).values)[0])
            if nan_length < -1:
                if i == len(df) + nan_length:
                    X = df.iloc[:i].drop(['national_gdp'], axis=1).values
                    y = df.iloc[:i]['national_gdp'].values
                    model.fit(X,y)
                    y_pred.append(model.predict(df.iloc[[i+1]].drop(['national_gdp'], axis=1).values)[0])
            if name == 'XGBRegressor':
                xgb_model = model
        print(name, dt.now() - start_time)
        result[name] = y_pred

    # combine models together
    result['Final'] = result[selected_models].mean(axis=1)
    return result