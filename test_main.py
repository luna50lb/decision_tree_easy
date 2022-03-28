
import datetime


from cart import DecisionTreeRegressor
import pandas as pd
import numpy as np


if __name__=='__main__':
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
    for jcol in list_feature_cols + [target_col]:
        df_w[jcol]=df_w[jcol].fillna(value=0).astype(np.float64)
    #df_w[:5]
    dtr0=DecisionTreeRegressor();
    df_summary=dtr0.optimize(df_input=df_w.copy(), list_feature_cols=list_feature_cols, target_col=target_col)
    print('summary=\n', df_summary )
    #
    print('normal exit at ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))