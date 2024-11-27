import pandas as pd
import lightgbm

#数据预处理
data1=pd.read_csv('diabetes比赛训练集.csv', encoding='gbk')
data2=pd.read_csv('diabetes比赛测试集.csv', encoding='gbk')
#label标记为-1
data1['患有糖尿病标识']=-1
#训练集和测试机合并
data=pd.concat([data1,data2],axis=0,ignore_index=True)
#将舒张压特征中的缺失值填充为-1
data['舒张压']=data['舒张压'].fillna(-1)

#----------------特征工程----------------
"""
将出生年份换算成年龄
"""
data['年龄']=2022-data['出生年份']  #换成年龄


"""
人体的成人体重指数正常值是在18.5-24之间
低于18.5是体重指数过轻
在24-27之间是体重超重
27以上考虑是肥胖
高于32了就是非常的肥胖。
"""
def BMI(a):
    if a<18.5:
        return 0
    elif 18.5<=a<=24:
        return 1
    elif 24<a<=27:
        return 2
    elif 27<a<=32:
        return 3
    else:
        return 4

data['BMI']=data['体重指数'].apply(BMI)

#糖尿病家族史
"""
无记录
叔叔或者姑姑有一方患有糖尿病/叔叔或姑姑有一方患有糖尿病
父母有一方患有糖尿病
"""
def FHOD(a):
    if a=='无记录':
        return 0
    elif a=='叔叔或者姑姑有一方患有糖尿病' or a=='叔叔或姑姑有一方患有糖尿病':
        return 1
    else:
        return 2


data['糖尿病家族史']=data['糖尿病家族史'].apply(FHOD)
"""
舒张压范围为60-90
"""
def DBP(a):
    if a<60:
        return 0
    elif 60<=a<=90:
        return 1
    elif a>90:
        return 2
    else:
        return a
data['DBP']=data['舒张压'].apply(DBP)


#------------------------------------
#将处理好的特征工程分为训练集和测试集，其中训练集是用来训练模型，测试集用来评估模型准确度
#其中编号和患者是否得糖尿病没有任何联系，属于无关特征予以删除
train=data[data['患有糖尿病标识'] !=-1]
test=data[data['患有糖尿病标识'] ==-1]
train_label=train['患有糖尿病标识']
train=train.drop(['编号','患有糖尿病标识','出生年份'],axis=1)
test=test.drop(['编号','患有糖尿病标识','出生年份'],axis=1)

#使用Lightgbm方法训练数据集，使用5折交叉验证的方法获得5个测试集预测结果
from sklearn.model_selection import KFold
def select_by_lgb(train_data,train_label,test_data,random_state=2022,n_splits=5,metric='auc',num_round=10000): #early_stopping_rounds=100
   kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
   fold=0
   result=[]
   for train_idx, val_idx in kfold.split(train_data):
       random_state+=1
       train_x = train_data.loc[train_idx]
       train_y = train_label.loc[train_idx]
       test_x = train_data.loc[val_idx]
       test_y = train_label.loc[val_idx]
       clf=lightgbm
       train_matrix=clf.Dataset(train_x,label=train_y)
       test_matrix=clf.Dataset(test_x,label=test_y)
       params={
               'boosting_type': 'gbdt',
               'objective': 'binary',
               'learning_rate': 0.1,
               'metric': metric,
               'seed': 2020,
               'nthread':-1,

               }
       model=clf.train(params,train_matrix,num_round,valid_sets=test_matrix)#early_stopping_rounds=early_stopping_rounds
       pre_y=model.predict(test_data)
       result.append(pre_y)
       fold+=1
   return result

test_data=select_by_lgb(train,train_label,test)
#test_data就是5折交叉验证中5次预测的结果
pre_y=pd.DataFrame(test_data).T
#将5次预测的结果求取平均值，当然也可以使用其他的方法
pre_y['averge']=pre_y['averge'].mean(axis=0)
#pre_y['averge']=pre_y[[i for i in range(5)]].mean(axis=1)
#因为竞赛需要你提交最后的预测判断，而模型给出的预测结果是概率，因此我们认为概率>0.5的即该患者有糖尿病，概率<=0.5的没有糖尿病
pre_y['label']=pre_y['averge'].apply(lambda x:1 if x>0.5 else 0)


result=pd.read_csv('diabetes提交示例.csv')
result['label']=pre_y['label']
result.to_csv('result.csv',index=False)
