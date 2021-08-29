import os 
import datetime 
import sys

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance 

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()




from matplotlib import style as style0
style0.use('seaborn-colorblind');

from matplotlib.font_manager import FontProperties
dir_fonts=FontProperties(fname='/System/Library/Fonts/Menlo.ttc', size=12, style='oblique', variant='normal'); 
plt.rcParams['font.family'] = dir_fonts.get_name()

import graphviz

CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'


if __name__=='__main__':
    df_data=pd.DataFrame(pd.date_range(start='2020-05-25', end='2020-09-30', freq='D'), columns=['date0']);
    df_data['month0']=df_data['date0'].dt.month;
    df_data['day0']=df_data['date0'].dt.day;
    df_data['noise0']=np.random.randint(low=-1,high=1,size=(df_data.shape[0],))
    df_data['v1']=df_data['month0'] * 0.1 +  np.sin( df_data['day0'] * 1.0) / ( 2 + 1.5 * np.sin(df_data['day0'] * 1.5) ) + df_data['noise0'];
    #
    print('df_data=\n', df_data.head(n=5))
    df_data['v1a']=df_data.apply(lambda jrow: 10 if jrow['day0']==10 else jrow['v1'], axis=1);
    df_data['v1']=df_data['v1a'];
    #    
    df_train=df_data.iloc[:105,:]
    df_validation=df_data.iloc[105:,:]
    print('train=\n', df_train[['date0', 'month0', 'day0', 'noise0']].to_numpy())
    regressor0 = DecisionTreeRegressor(random_state=0, max_depth=3) #Fit the regressor object to the dataset. 
    clf0=regressor0.fit(df_train[['month0', 'day0']].to_numpy(), df_train['v1'].to_numpy().reshape(-1,1))
    from sklearn.tree import export_graphviz
    tree_data0=export_graphviz(regressor0, out_file=None, feature_names=['month0', 'day0'], class_names=['v1'], rounded=True, filled=True)
    gv0=graphviz.Source(tree_data0);
    gv0.format='png'
    gv0.render(datetime.datetime.now().strftime('%Y%m%d_') + "chart_.gv", view=True)
    list_impurities=regressor0.tree_.impurity
    print('impurities=\n', list_impurities);
    print('value=\n', regressor0.tree_.value);
    print('children_left=\n', regressor0.tree_.children_left);
    print('node_count=\n', regressor0.tree_.node_count)
    print('feature=\n', regressor0.tree_.feature);
    #
    print('.....')
    df_validation=df_validation.reset_index(drop=True);
    df_validation['forecast1']=regressor0.predict(df_validation[['month0', 'day0']].to_numpy() )
    #df_validation['forecast1']=np.nan;
    #df_validation.at[:, 'forecast1']
    #dfpr=pd.DataFrame({'forecast1':regressor0.predict(df_validation[['month0', 'day0']] ) })
    #print('nexxxt')
    #print(dfpr)
    #df_validation['forecast1']=dfpr['forecast1']
    #print('type=', type(forecast1))
    print(df_validation)

    imp_a0=permutation_importance(clf0, df_train[['month0', 'day0']].to_numpy(), df_train['v1'].to_numpy().reshape(-1,1) )
    R2v=r2_score(y_true=df_validation['v1'].to_numpy().reshape(-1,1), y_pred=df_validation['forecast1'].to_numpy().reshape(-1,1) )
    print('importance=\n', imp_a0)
    print('coeff of determination=\n', R2v);
    fig0=plt.figure(figsize=(15,7));
    gspc0=fig0.add_gridspec(1,1)
    ax0=fig0.add_subplot(gspc0[0,0])
    #ax0.plot(df_train['date0'], df_train['v1'], lw=2, label='training')
    ax0.plot(df_validation['date0'], df_validation['forecast1'], lw=2, label='forecast')
    ax0.plot(df_data['date0'], df_data['v1'], ls='--', lw=2, label='actual')
    ax0.tick_params(axis='both', which='major', labelsize=15)
    ax0.grid();
    ax0.legend()
    plt.setp(ax0.get_legend().get_texts(), fontsize=17)
    plt.tight_layout()
    plt.show();
    plt.close("all")



print('now=', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#sys.exit();
