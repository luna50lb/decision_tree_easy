# レポジトリ decision_tree_easy 
## 名称: decision_tree_easy




## クラス DecisionTreeRegressor 
### 概要
- Cartクラスのサブクラス
- 決定木のコード、回帰
- information_gainがnullになったnodeについてはleaf nodeとしてそれ以降の分枝は行われない。

### 使い方
```
dtr_inst=DecisionTreeRegressor(node_level_max=4, n_quantiles=6);
df_res_train, df_res_val=dtr_inst(df_train=df_train.copy(), df_val=df_val.copy(), list_feature_cols=list_feature_cols, target_col=target_col)
    
```


### エンハンス 
- 分岐図を作成するなどして、どのような分岐プロセスになっているかを視覚化できるように
- 枝かりの部分





