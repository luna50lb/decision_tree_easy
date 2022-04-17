# decision_tree_easy 
- 名称: decision_tree_easy
- 概要: 決定木のコード、回帰 


### エンハンス 
- 分岐図を作成するなどして、どのような分岐プロセスになっているかを視覚化できるように
- 枝かりの部分


### 使い方
```
dtr_inst=DecisionTreeRegressor(node_level_max=4);
[df_res_train, df_res_val]=dtr_inst(df_train=df_train.copy(), df_val=df_val.copy(), list_feature_cols=list_feature_cols, target_col=target_col)
    
```
