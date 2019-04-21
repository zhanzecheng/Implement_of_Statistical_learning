# Adaboost
使用 逻辑回归(Logical Regression) 作为Adaboost的基分类器

## 使用方式

    # define the base classify
    clf = LogisticRegression()
    
    # using the adaboost method
    er_i = adaboost_clf(y, X, y_test, X_test, i, clf)

    
## 任务说明

使用adaboost来做分类任务
