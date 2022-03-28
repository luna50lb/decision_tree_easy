import pandas as pd
import numpy as np






class DecisionTreeRegressor():
    def __init__(self, node_level_max=4, n_quantiles=6, verbose=1):
        if verbose==1:
            print('DecisionTreeRegressor class used')
        self.n_quantiles=n_quantiles
        self.node_level_max=node_level_max

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
            #
            df_right=df_p[df_p[jfeat]>=feat_threshold].reset_index(drop=True)
            cost_r=df_right[target_col].std(ddof=0)
            n_sample_r=df_right.shape[0]
            #
            information_gain=cost_p - (n_sample_l * cost_l / n_sample_p) - (n_sample_r * cost_r / n_sample_p)
            # print('feat={}, {:.2f}, {:.3f}, {:.3f}, {:.3f}, {:.4f}'.format( jfeat, feat_threshold, cost_p, cost_l, cost_r, information_gain) )
            df_j=pd.DataFrame({'feature':[jfeat], 'threshold':[feat_threshold], 'information_gain':[information_gain], 'df_left':[df_left], 'df_right':[df_right]})
            df_agg=pd.concat([df_agg, df_j],axis=0)
            del(df_left)
            del(df_right)
        #
        df_agg=df_agg.sort_values(by=['information_gain'],ascending=[False]*1).reset_index(drop=True)
        return df_agg.values.tolist()[0]

    def optimize(self, df_input, list_feature_cols, target_col, ): 
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
                                          'df_left':[df_left], 'df_right':[df_right], 'parent_node_id':["n/a"] } )
                del(jfeat)
                del(feat_threshold)
                del(information_gain)
                del(df_left)
                del(df_right)
            elif node_level>=1:
                df_node_parent=df_summary[df_summary['node_level']==node_level-1].reset_index(drop=True)
                df_node_agg=pd.DataFrame([])
                for jdx, jrow in enumerate(df_node_parent.itertuples()):
                    # print('**')
                    df_left_parent=jrow.df_left
                    # print('left parent type=', type(df_left_parent))
                    df_grid=self.search_thresholds(df_p=df_left_parent.copy(), list_feature_cols=list_feature_cols, n_quantiles=n_quantiles)
                    jfeat, feat_threshold, information_gain, df_left, df_right=self.get_nodes(df_p=df_left_parent.copy(), df_grid=df_grid.copy(), target_col=target_col )
                    if df_left.shape[0] + df_right.shape[0]!=df_left_parent.shape[0]:
                        print('Error! {}, {}, left parent n_sample={} '.format(df_left.shape[0], df_right.shape[0], df_left_parent.shape[0]) )
                        raise ValueError;
                    df_jl=pd.DataFrame({'node_id':[ str(node_level) + '_' + str(jdx) + 'l' ], 'node_level':[node_level], 'feature':[jfeat], 'threshold':[feat_threshold], 
                                      'information_gain':[information_gain], 'n_left':[df_left.shape[0]], 'n_right':[df_right.shape[0] ], 
                                        'df_left':[df_left], 'df_right':[df_right], 'parent_node_id':[jrow.node_id]})
                    df_node_agg=pd.concat([df_node_agg, df_jl],axis=0)
                    del(df_jl)
                    del(df_left_parent)
                    # 
                    df_right_parent=jrow.df_right
                    df_grid=self.search_thresholds(df_p=df_right_parent.copy(), list_feature_cols=list_feature_cols, n_quantiles=n_quantiles)
                    jfeat, feat_threshold, information_gain, df_left, df_right=self.get_nodes(df_p=df_right_parent.copy(), df_grid=df_grid.copy(), target_col=target_col )
                    if df_left.shape[0] + df_right.shape[0]!=df_right_parent.shape[0]:
                        print('Error! {}, {}, right parent n_sample={} '.format(df_left.shape[0], df_right.shape[0], df_right_parent.shape[0]) )
                        raise ValueError;
                    df_jr=pd.DataFrame({'node_id':[ str(node_level) + '_' + str(jdx) + 'r' ], 'node_level':[node_level], 'feature':[jfeat], 'threshold':[feat_threshold], 
                                      'information_gain':[information_gain], 'n_left':[df_left.shape[0]], 'n_right':[df_right.shape[0] ], 
                                        'df_left':[df_left], 'df_right':[df_right], 'parent_node_id':[jrow.node_id]})
                    df_node_agg=pd.concat([df_node_agg, df_jr],axis=0)
                    del(df_jr)
                    del(df_right_parent)
                del(df_node_parent)
                df_node_agg=df_node_agg.reset_index(drop=True)
            else:
                print('Error! node_level=', node_level)
                raise ValueError
            df_summary=pd.concat([df_summary, df_node_agg],axis=0).reset_index(drop=True)
            del(df_node_agg)
        return df_summary;

    # infer method under construction 
    def infer(self):
        return -1; 
