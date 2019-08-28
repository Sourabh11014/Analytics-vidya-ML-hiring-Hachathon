import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import xgboost
import matplotlib.pyplot as plt
df_train = pd.read_csv('train.csv', header=0)
df_test = pd.read_csv('test.csv',header=0)

df_train_0 = df_train[df_train['m13'] == 0]
df_train_1 = df_train[df_train['m13'] == 1]

downsample = 55
oversample = 2.5
df_train_0_1 = df_train_0[:downsample*len(df_train_1)]
df_train_0_2 = df_train_0[downsample*len(df_train_1):2*downsample*len(df_train_1)]
df_train_0_3 = df_train_0[2*downsample*len(df_train_1):3*downsample*len(df_train_1)]
df_train_0_4 = df_train_0[3*downsample*len(df_train_1):]
df_train_1 = df_train_1.sample(int(oversample*len(df_train_1)), replace=True)

df = pd.concat([df_train_1, df_train_0_1])
df_1 = pd.concat([df_train_1, df_train_0_2])
df_2 = pd.concat([df_train_1, df_train_0_3])
df_3 = pd.concat([df_train_1, df_train_0_4])


def Model(df, flag, mdel):
    df = pd.get_dummies(df, columns=['source', 'financial_institution', 'loan_purpose'])
    df['origination_date'] = pd.to_datetime(df['origination_date'], format="%Y/%m/%d")
    df['first_payment_date'] = pd.to_datetime(df['first_payment_date'], format="%m/%Y")
    df['term'] = df['loan_term'].astype('timedelta64[ns]')
    df['day'] = df['first_payment_date'] - df['origination_date']
    df['app'] = df['day'] / df['term']
    df['app'] = df['app'].astype('float')

    y = df['m13']
    x = df.drop(columns=['m13', 'loan_id', 'origination_date', 'first_payment_date', 'day', 'term'])

    x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size=0.3)

    dt = DecisionTreeClassifier(max_depth=21,
                                min_samples_split=2,
                                max_features=None,
                                random_state=None,
                                max_leaf_nodes=None,
                                )

    if flag:
        model = xgboost.XGBClassifier(base_estimator=dt, n_estimators=500, xgb_model=mdel,scale_pos_weight=4)
    else:
        model = xgboost.XGBClassifier(base_estimator=dt, n_estimators=500, scale_pos_weight=4)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_cv)

    print(classification_report(y_cv, y_pred))

    return model


model_1 = Model(df, False, None)
model_2 = Model(df_1, True, model_1)
model_3 = Model(df_2, True, model_2)

#print(model_3.feature_importances_)



df_test = pd.get_dummies(df_test, columns=['source', 'financial_institution', 'loan_purpose'])
df_test['origination_date'] = pd.to_datetime(df_test['origination_date'], format="%d/%m/%y")
df_test['first_payment_date'] = pd.to_datetime(df_test['first_payment_date'], format="%b-%y")
df_test['term'] = df_test['loan_term'].astype('timedelta64[ns]')
df_test['day'] = df_test['first_payment_date'] - df_test['origination_date']
df_test['app'] = df_test['day']/df_test['term']
df_test['app'] = df_test['app'].astype('float')
df_test.drop(columns=['loan_id', 'origination_date', 'first_payment_date', 'day', 'term'], inplace=True)

y = model_3.predict(df_test)
df_test['m13'] = y
df_test = df_test[['m13']]
df_test['loan_id'] = df_test.index + 1

df_test.to_csv('final_submission.csv', index=False)

feat_importance = pd.Series(model_3.feature_importances_, index=col)
feat_importance.nlargest(30).plot(kind='barh')
#plt.show()