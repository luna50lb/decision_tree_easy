
import datetime


from cart import DecisionTreeRegressor
import pandas as pd
import numpy as np

import sys


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

if __name__=='__main__':
    
    df=pd.DataFrame({'x0':np.arange(-5,10,0.2)} )
    df['y']=1.5 / (1.2 + np.cos( df['x0'] ) ) + np.random.randint(low=-50,high=50, size=(df.shape[0], ) ) * 0.01
    print('df=\n',df)
    #print('columns=', df.columns)
    train_val_split=0.78
    train_val_split_idx=np.int64(df.shape[0] * train_val_split)
    df_train=df[: train_val_split_idx]
    df_val=df[train_val_split_idx:]
    list_feature_cols=['x0']
    target_col='y'
    dtr_inst=DecisionTreeRegressor(node_level_max=4);
    df_res_train, df_res_val=dtr_inst(df_train=df_train.copy(), df_val=df_val.copy(), list_feature_cols=list_feature_cols, target_col=target_col)
    print('res train=\n',df_res_train )
    print('res train columns=\n', df_res_train.columns)
    #df_agg=dtr_inst.infer(df_infer=df_train.copy(), list_feature_cols=list_feature_cols, target_col=target_col)
    #print('agg=\n', df_agg[['node_id', 'df_left', 'df_right', 'parent_node_id', ]] )
    #print('agg=\n', df_agg[['node_id',  'parent_node_id', ]] )
    #print('agg=\n', df_agg[['node_id',  'mean_left', 'mean_right', 'isleafnode']] )
    #print('agg columns=\n', df_agg.columns)
    #print('exit')
    from matplotlib import pyplot as plt
    ax=plt.subplot()
    ax.plot(df_res_train['x0'], df_res_train['y'], label='train y')
    ax.plot(df_res_train['x0'], df_res_train['y_pred'], label='train y_pred')
    ax.plot(df_res_val['x0'], df_res_val['y'], label='val y')
    ax.plot(df_res_val['x0'], df_res_val['y_pred'], label='val y_pred')
    ax.legend()
    plt.show()
    plt.close('all')
    sys.exit()
    #
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    df_a=pd.DataFrame(data=housing['data'], columns=housing['feature_names'])
    df_a['target']=housing['target']
    # df_a
    list_feature_cols=housing['feature_names']
    target_col='target'
    df_w=df_a.copy()
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