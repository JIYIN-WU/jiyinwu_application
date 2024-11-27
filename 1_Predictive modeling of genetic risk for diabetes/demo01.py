import pandas as pd
import warnings
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb


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
df_y=train['患有糖尿病标识']
df_x=train.drop(['编号','患有糖尿病标识','出生年份'],axis=1)
#test=test.drop(['编号','患有糖尿病标识','出生年份'],axis=1)

# 划分为5折交叉验证数据集
#df_y=data_all['status']
#df_X=data_all.drop(columns=['status'])
df_x=scale(df_x,axis=0)  #将数据转化为标准数据
#构建模型

lr = LogisticRegression(random_state=2018,tol=1e-6)  # 逻辑回归模型

tree = DecisionTreeClassifier(random_state=2018) #决策树模型

svm = SVC(probability=True,random_state=2018,tol=1e-6)  # SVM模型

forest=RandomForestClassifier(n_estimators=100,random_state=2018) #　随机森林

Gbdt=GradientBoostingClassifier(random_state=2018) #GBDT

Xgbc=XGBClassifier(random_state=2018)  #Xgbc

gbm=lgb.LGBMClassifier(random_state=2018)  #lgb


def muti_score(model):
    warnings.filterwarnings('ignore')
    accuracy = cross_val_score(model, df_x, df_y, scoring='accuracy', cv=5)
    precision = cross_val_score(model, df_x, df_y, scoring='precision', cv=5)
    recall = cross_val_score(model, df_x, df_y, scoring='recall', cv=5)
    f1_score = cross_val_score(model, df_x, df_y, scoring='f1', cv=5)
    auc = cross_val_score(model, df_x, df_y, scoring='roc_auc', cv=5)
    print("准确率:",accuracy.mean())
    print("精确率:",precision.mean())
    print("召回率:",recall.mean())
    print("F1_score:",f1_score.mean())
    print("AUC:",auc.mean())



model_name=["lr","tree","svm","forest","Gbdt","Xgbc","gbm"]
for name in model_name:
    model=eval(name)
    print(name)
    muti_score(model)


'''
lr
准确率: 0.7890191148682617
精确率: 0.6542724662896913
召回率: 0.3377975457965613
F1_score: 0.44525012166067884
AUC: 0.7840451024530857
tree
准确率: 0.6962524533638791
精确率: 0.39920670173446693
召回率: 0.4157413593052284
F1_score: 0.40705496051057793
AUC: 0.6029856787858856
svm
准确率: 0.787758390223099
精确率: 0.7351623295760905
召回率: 0.24060335431243626
F1_score: 0.36179547264664874
AUC: 0.7640376541388867
forest
准确率: 0.7921756804332226
精确率: 0.7135700690071172
召回率: 0.2867128441334693
F1_score: 0.40835414886475174
AUC: 0.7752164698827589
Gbdt
准确率: 0.7938590063951863
精确率: 0.6604108594441386
召回率: 0.36633732991104395
F1_score: 0.4708811551285791
AUC: 0.7888240065764295
Xgbc
准确率: 0.7982740847293591
精确率: 0.6829783239831001
召回率: 0.3663162336064133
F1_score: 0.47673826685376613
AUC: 0.7914190511145234
gbm
准确率: 0.79049080811139
精确率: 0.6421783397519263
召回率: 0.3730354066312717
F1_score: 0.47150438344663004
AUC: 0.7776116341798183
'''
