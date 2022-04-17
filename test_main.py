
import datetime


from cart import DecisionTreeRegressor
import pandas as pd
import numpy as np


if __name__=='__main__':
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    df_a=pd.DataFrame(data=housing['data'], columns=housing['feature_names'])
    df_a['target']=housing['target']
    # df_a
    list_feature_cols=housing['feature_names']
    target_col='target'
    df_w=df_a.copy()
    
    ''' 
    df_w=pd.DataFrame([])
    df_w=pd.read_csv('weather_data_HanedaJan23rd2022.csv', header=0, encoding='cp932')
    df_w.columns=np.arange(0, df_w.shape[1], 1)
    df_w=df_w[[0,1,2,3,4,5,6]]
    df_w=df_w[5:].reset_index(drop=True)
    df_w=df_w.rename(columns={0:'datetime', 1:'temperature', 2:'high', 3:'low', 4:'wind', 5:'quolity', 6:'direction'})
    df_w['datetime']=pd.to_datetime(df_w['datetime'], format='%Y/%m/%d %H:%M')
    df_w['hour']=df_w['datetime'].dt.hour

    list_feature_cols=['hour', 'wind']
    target_col='temperature'
    ''' 
    
    print('features=', list_feature_cols)
    for jcol in list_feature_cols + [target_col]:
        df_w[jcol]=df_w[jcol].fillna(value=0).astype(np.float64)
    #df_w[:5]
    itrain0=np.int64(df_a.shape[0]*0.7)
    print('int=', itrain0)
    df_train=df_w[:itrain0]
    df_val=df_w[itrain0:]
    dtr_inst=DecisionTreeRegressor(node_level_max=4);
    [df_res_train, df_res_val]=dtr_inst(df_train=df_train.copy(), df_val=df_val.copy(), list_feature_cols=list_feature_cols, target_col=target_col)
    
    print('res train=\n', df_res_train)
    print('res val=\n', df_res_val[:17])
    print('res val=\n', df_res_val[33:60])
    #
    print('normal exit at ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))