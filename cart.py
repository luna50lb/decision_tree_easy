import pandas as pd
import numpy as np




"""
クラス名: Cart
概要: 決定木の構築

information_gainがnullとなった場合には、それ以降の分枝は行われないという仕様。
"""

class Cart():
    def __init__(self, node_level_max=4, n_quantiles=6, verbose=1):
        self.verbose=verbose
        if self.verbose==1:
            print('class Cart is used')
        self.n_quantiles=n_quantiles
        self.node_level_max=node_level_max
    #
    # 今の所 modeはvalidationのみ(エンハンス必要)
    def __call__(self, df_train, df_val, list_feature_cols, target_col, mode='validation'):
        vt0=self.optimize(df_input=df_train.copy(), list_feature_cols=list_feature_cols, target_col=target_col)
        df_res_train, df_0=self.infer(df_infer=df_train.copy(), list_feature_cols=list_feature_cols, target_col=target_col)
        df_res_val, df_1=self.infer(df_infer=df_val.copy(), list_feature_cols=list_feature_cols, target_col=target_col)
        
        ''' 
        df_res_val=self.infer(df_infer=df_val.copy(), list_feature_cols=list_feature_cols, target_col=target_col)
        df_res_val=df_res_val[df_res_val['isleafnode']==True].reset_index(drop=True)
        df_leaf=pd.DataFrame([])
        for jrow in df_res_val.itertuples():
            jrow.df_left['mean']=jrow.mean_left
            jrow.df_right['mean']=jrow.mean_right
            df_j=pd.concat([jrow.df_left, jrow.df_right], axis=0)
            df_j=df_j.reset_index(drop=True)
            #print('jrow df_left=\n', jrow.df_left)
            df_leaf=pd.concat([df_leaf,df_j] ,axis=0)
            del(df_j)
        del(df_res_val)
        df_leaf=df_leaf.reset_index(drop=True)
        df_res_val=df_leaf.copy()
        return [df_res_train, df_res_val];
        '''
        return df_res_train, df_res_val
    #
    # やっと推論部分の実装に着手
    def infer(self, df_infer, list_feature_cols, target_col):
        if 'index' in df_infer.columns.tolist():
            print('Error! df_infer should not contain column whose name is index.')
            raise ValueError
        df_infer=df_infer.reset_index(drop=True).reset_index(drop=False).rename(columns={'index':'record_id' } )
        #
        node_level_max=self.node_level_max
        df_instruct=self.df_summary[['node_id', 'node_level', 'feature', 'threshold', 'parent_node_id']].copy()
        df_records_agg=pd.DataFrame([])
        for node_id in df_instruct['node_id'].values.tolist():
            df_compass=df_instruct[df_instruct['node_id']==node_id].reset_index(drop=True)
            if self.verbose==1:
                print('node_level=', df_compass.at[0, 'node_level'], ' node=', node_id)
            threshold_value=df_compass.at[0, 'threshold']
            feature_col=df_compass.at[0, 'feature']
            parent_node_id=df_compass.at[0, 'parent_node_id']
            if node_id in ['root']:
                df_left=df_infer[df_infer[feature_col]<threshold_value].reset_index(drop=True)
                df_right=df_infer[df_infer[feature_col]>=threshold_value].reset_index(drop=True)
            else:
                df_temp=df_records_agg[df_records_agg['node_id']==parent_node_id].reset_index(drop=True)
                df_node=pd.DataFrame([])
                if node_id.endswith('l'):
                    df_node=df_temp.at[0,'df_left']
                elif node_id.endswith('r'):
                    df_node=df_temp.at[0,'df_right']
                else:
                    print('Error!, node_id=', node_id)
                    raise ValueError;
                del(df_temp)
                df_left=df_node[df_node[feature_col]<threshold_value].reset_index(drop=True)
                df_right=df_node[df_node[feature_col]>=threshold_value].reset_index(drop=True)
                del(df_node)
            del(df_compass)
            df_j=pd.DataFrame({'node_id':[node_id], 'df_left':[df_left], 'df_right':[df_right], 'parent_node_id':[parent_node_id], 
                               'mean_left':[df_left[target_col].mean() ], 'mean_right':[df_right[target_col].mean() ], } )
            df_records_agg=pd.concat([df_records_agg, df_j],axis=0).reset_index(drop=True)
            del(df_j)
        df_records_agg=df_records_agg.reset_index(drop=True)
        
        # leaf node(葉ノード)の決定
        df_parent_nodes=pd.DataFrame({'node_id':df_records_agg['parent_node_id'].unique().tolist() })
        df_parent_nodes['isleafnode']=False;
        df_records_agg=df_records_agg.merge(df_parent_nodes, on=['node_id'], how='left', validate='1:1', suffixes=('', '_parent'))
        df_records_agg['isleafnode']=df_records_agg['isleafnode'].apply(lambda bool0: True if pd.isnull(bool0) else bool0)
        df_records_agg=df_records_agg.reset_index(drop=True)
        #
        df_chosen=df_records_agg[df_records_agg['isleafnode']==True].reset_index(drop=True)
        df_pred=pd.DataFrame([])
        for jrow in df_chosen.itertuples():
            df_left=jrow.df_left.copy()
            df_left['y_pred']=jrow.mean_left
            df_right=jrow.df_right.copy()
            df_right['y_pred']=jrow.mean_right
            df_pred=pd.concat([df_pred, df_left[['record_id', 'y_pred']], df_right[['record_id', 'y_pred']] ],axis=0)
            del(df_left)
            del(df_right)
        df_pred=df_pred.reset_index(drop=True)
        df_infer=df_infer.merge(df_pred, on=['record_id'], how='left', validate='1:1', suffixes=('', '_pred') )
        #print('D0: infer=\n', df_infer)
        return df_infer, df_records_agg;
    
    def search_thresholds(self, df_p, list_feature_cols, n_quantiles=6):
        df_grid=pd.DataFrame([])
        df_quantile=df_p[list_feature_cols].quantile(q=np.linspace(0, 1, n_quantiles).tolist()[1:-1] ).reset_index().reset_index(drop=True).rename(columns={'index':'quantile'})
        # print('quantile=\n', df_quantile)
        for jcol in list_feature_cols:
            df_j=df_quantile[[jcol]].reset_index(drop=True)
            df_j['feature']=jcol
            df_j=df_j.rename(columns={jcol:'value'} )
            df_grid=pd.concat([df_grid, df_j],axis=0)
            del(df_j)
        df_grid=df_grid.reset_index(drop=True)
        df_grid=df_grid[['feature', 'value']]
        return df_grid;

    def get_nodes(self, df_p, df_grid, target_col):
        df_agg=pd.DataFrame([])
        for jlist in df_grid.values:
            jfeat=jlist[0] # 'hour'
            feat_threshold=jlist[1] # 8
            #
            cost_p=df_p[target_col].std(ddof=0) # parentノードのコスト
            n_sample_p=df_p.shape[0]
            #
            df_left=df_p[df_p[jfeat]<feat_threshold].reset_index(drop=True)
            cost_l=df_left[target_col].std(ddof=0)
            n_sample_l=df_left.shape[0]
            #target_value_left=df_left[target_col].mean()
            #
            df_right=df_p[df_p[jfeat]>=feat_threshold].reset_index(drop=True)
            cost_r=df_right[target_col].std(ddof=0)
            n_sample_r=df_right.shape[0]
            #target_value_right=df_right[target_col].mean()
            #
            information_gain=cost_p - (n_sample_l * cost_l / n_sample_p) - (n_sample_r * cost_r / n_sample_p)
            # print('feat={}, {:.2f}, {:.3f}, {:.3f}, {:.3f}, {:.4f}'.format( jfeat, feat_threshold, cost_p, cost_l, cost_r, information_gain) )
            df_j=pd.DataFrame({'feature':[jfeat], 'threshold':[feat_threshold], 'information_gain':[information_gain], 'df_left':[df_left], 'df_right':[df_right] } )
            df_agg=pd.concat([df_agg, df_j],axis=0)
            del(df_left)
            del(df_right)
        #
        df_agg=df_agg.sort_values(by=['information_gain'],ascending=[False]*1).reset_index(drop=True)
        return df_agg.values.tolist()[0]

    def optimize(self, df_input, list_feature_cols, target_col, ): 
        a_region=np.zeros(shape=(len(list_feature_cols),2) )
        a_region[:,0]=-np.inf
        a_region[:,1]=np.inf
        #
        node_level_max=self.node_level_max
        df_w0=df_input.copy()
        n_quantiles=self.n_quantiles
        df_summary=pd.DataFrame([])
        # ノードを追加していく
        for node_level in np.arange(0,node_level_max,1):
            print('node_level=', node_level)
            if node_level==0:
                df_grid=self.search_thresholds(df_p=df_w0.copy(), list_feature_cols=list_feature_cols, n_quantiles=n_quantiles)
                jfeat, feat_threshold, information_gain, df_left, df_right=self.get_nodes(df_p=df_w0.copy(), df_grid=df_grid.copy(), target_col=target_col )
                if df_left.shape[0] + df_right.shape[0]!=df_w0.shape[0]:
                    print('Error! {}, {}, n_sample={} '.format(df_left.shape[0], df_right.shape[0], df_w0.shape[0]) )
                    raise ValueError;
                df_node_agg=pd.DataFrame({'node_id':[ 'root' ], 'node_level':[node_level], 'feature':[jfeat], 'threshold':[feat_threshold], 
                                      'information_gain':[information_gain], 'n_left':[df_left.shape[0]], 'n_right':[df_right.shape[0] ], 
                                          'target_left':[df_left[target_col].mean() ], 'target_right':[df_right[target_col].mean() ], 'df_left':[df_left], 'df_right':[df_right], 'parent_node_id':["n/a"],
                                         'a_region':[a_region] } )
                del(jfeat)
                del(feat_threshold)
                del(information_gain)
                del(df_left)
                del(df_right)
            elif node_level>=1:
                #df_node_parent=df_summary[df_summary['node_level']==node_level-1].reset_index(drop=True)
                df_node_parent=df_summary[(df_summary['node_level']==node_level-1) & (~df_summary['information_gain'].isnull() )].reset_index(drop=True)
                df_node_agg=pd.DataFrame([])
                for jdx, jrow in enumerate(df_node_parent.itertuples()):
                    a_region_l=jrow.a_region.copy()
                    a_region_r=jrow.a_region.copy()
                    a_region_l[ list_feature_cols.index(jrow.feature)  ,1]=np.min( [jrow.threshold, jrow.a_region[list_feature_cols.index(jrow.feature), 1]  ] )
                    a_region_r[ list_feature_cols.index(jrow.feature)  ,0]=np.max( [jrow.threshold, jrow.a_region[list_feature_cols.index(jrow.feature), 0]  ] )
                    #
                    # print('**')
                    df_left_parent=jrow.df_left.copy()
                    # print('left parent type=', type(df_left_parent))
                    df_grid=self.search_thresholds(df_p=df_left_parent.copy(), list_feature_cols=list_feature_cols, n_quantiles=n_quantiles)
                    jfeat, feat_threshold, information_gain, df_left, df_right=self.get_nodes(df_p=df_left_parent.copy(), df_grid=df_grid.copy(), target_col=target_col )
                    if df_left.shape[0] + df_right.shape[0]!=df_left_parent.shape[0]:
                        print('Error! {}, {}, left parent n_sample={} '.format(df_left.shape[0], df_right.shape[0], df_left_parent.shape[0]) )
                        raise ValueError;
                    df_jl=pd.DataFrame({'node_id':[ str(node_level) + '_' + str(jdx) + 'l' ], 'node_level':[node_level], 'feature':[jfeat], 'threshold':[feat_threshold], 
                                      'information_gain':[information_gain], 'n_left':[df_left.shape[0]], 'n_right':[df_right.shape[0] ], 
                                        'df_left':[df_left], 'df_right':[df_right], 
                                        'target_left':[df_left[target_col].mean() ], 'target_right':[df_right[target_col].mean() ], 
                                        'parent_node_id':[jrow.node_id], 'parent_threshold':[jrow.threshold], 'parent_feature':[jrow.feature], 'a_region':[a_region_l] })
                    df_node_agg=pd.concat([df_node_agg, df_jl],axis=0)
                    del(df_jl)
                    del(df_left_parent)
                    del a_region_l
                    # 
                    df_right_parent=jrow.df_right.copy()
                    df_grid=self.search_thresholds(df_p=df_right_parent.copy(), list_feature_cols=list_feature_cols, n_quantiles=n_quantiles)
                    jfeat, feat_threshold, information_gain, df_left, df_right=self.get_nodes(df_p=df_right_parent.copy(), df_grid=df_grid.copy(), target_col=target_col )
                    if df_left.shape[0] + df_right.shape[0]!=df_right_parent.shape[0]:
                        print('Error! {}, {}, right parent n_sample={} '.format(df_left.shape[0], df_right.shape[0], df_right_parent.shape[0]) )
                        raise ValueError;
                    df_jr=pd.DataFrame({'node_id':[ str(node_level) + '_' + str(jdx) + 'r' ], 'node_level':[node_level], 'feature':[jfeat], 'threshold':[feat_threshold], 
                                      'information_gain':[information_gain], 'n_left':[df_left.shape[0]], 'n_right':[df_right.shape[0] ], 
                                        'df_left':[df_left], 'df_right':[df_right], 
                                        'target_left':[df_left[target_col].mean() ], 'target_right':[df_right[target_col].mean() ], 
                                        'parent_node_id':[jrow.node_id], 'parent_threshold':[jrow.threshold], 'parent_feature':[jrow.feature], 'a_region':[a_region_r] })
                    df_node_agg=pd.concat([df_node_agg, df_jr],axis=0)
                    del(df_jr)
                    del(df_right_parent)
                    del a_region_r
                del(df_node_parent)
                df_node_agg=df_node_agg.reset_index(drop=True)
            else:
                print('Error! node_level=', node_level)
                raise ValueError
            df_summary=pd.concat([df_summary, df_node_agg],axis=0).reset_index(drop=True)
            del(df_node_agg)
        df_summary=df_summary.reset_index(drop=True)
        self.df_summary=df_summary.copy()
        return df_summary;

    # 枝刈りについてはエンハンスが必要
    def prune(self):
        return -1;

    


# Cartのサブクラスとして定義
class DecisionTreeRegressor(Cart):
    def __init__(self, node_level_max=4, n_quantiles=6, verbose=1):
        #self.n_quantiles=n_quantiles
        #self.node_level_max=node_level_max
        super().__init__(node_level_max=node_level_max, n_quantiles=n_quantiles, verbose=verbose)
        if self.verbose==1:
            print('class DecisionTreeRegressor is used')
      
    # 今の所 modeはvalidationのみ(エンハンス必要)
    def __call__(self, df_train, df_val, list_feature_cols, target_col, mode='validation'):
        df_summary=self.optimize(df_input=df_train.copy(), list_feature_cols=list_feature_cols, target_col=target_col)
        df_res_train, df_0=self.infer(df_infer=df_train.copy(), list_feature_cols=list_feature_cols, target_col=target_col)
        df_res_val, df_1=self.infer(df_infer=df_val.copy(), list_feature_cols=list_feature_cols, target_col=target_col)
        #
        print('summary columns=\n', df_summary.columns)
        print('summary df=\n', df_summary[['node_id', 'parent_node_id', 'threshold', 'parent_threshold', 'feature', 'parent_feature', 'node_level', 'information_gain', 'n_left', 'n_right', 'a_region'] ])
        return df_res_train, df_res_val
    
    # Cartのエンハンスを行う。サンプルが0だった場合にはinformation gainがNAN
    def get_nodes(self, df_p, df_grid, target_col):
        df_agg=pd.DataFrame([])
        for jlist in df_grid.values:
            jfeat=jlist[0] # 'hour'
            feat_threshold=jlist[1] # 8
            #
            cost_p=df_p[target_col].std(ddof=0) # parentノードのコスト
            n_sample_p=df_p.shape[0]
            #
            df_left=df_p[df_p[jfeat]<feat_threshold].reset_index(drop=True)
            cost_l=df_left[target_col].std(ddof=0)
            n_sample_l=df_left.shape[0]
            #target_value_left=df_left[target_col].mean()
            #
            df_right=df_p[df_p[jfeat]>=feat_threshold].reset_index(drop=True)
            cost_r=df_right[target_col].std(ddof=0)
            n_sample_r=df_right.shape[0]
            #target_value_right=df_right[target_col].mean()
            #
            #print('n sample p, l, r=', n_sample_p, n_sample_l, n_sample_r, ' | cost p, l, r=', np.round(cost_p,2), np.round(cost_l,2), np.round(cost_r,2) )
            information_gain=cost_p - (n_sample_l * cost_l / n_sample_p) - (n_sample_r * cost_r / n_sample_p)
            # print('feat={}, {:.2f}, {:.3f}, {:.3f}, {:.3f}, {:.4f}'.format( jfeat, feat_threshold, cost_p, cost_l, cost_r, information_gain) )
            df_j=pd.DataFrame({'feature':[jfeat], 'threshold':[feat_threshold], 'information_gain':[information_gain], 'df_left':[df_left], 'df_right':[df_right] } )
            df_agg=pd.concat([df_agg, df_j],axis=0)
            del(df_left)
            del(df_right)
        #
        #print('df_agg=\n', df_agg[['feature', 'threshold', 'information_gain']])
        df_agg=df_agg.sort_values(by=['information_gain'],ascending=[False]*1).reset_index(drop=True)
        return df_agg.values.tolist()[0]
