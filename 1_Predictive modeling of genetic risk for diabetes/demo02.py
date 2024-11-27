#安装相关依赖库 如果是windows系统，cmd命令框中输入pip安装，参考上述环境配置
#!pip install sklearn
#!pip install pandas
#---------------------------------------------------
#导入库
#----------------数据探索----------------
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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

#----------------模型训练----------------
# model = DecisionTreeClassifier()            ##决策树模型
# model.fit(train, train_label)
# y_pre=model.predict(test)

model=RandomForestClassifier(n_estimators=100,random_state=2022) #　随机森林
model.fit(train, train_label)
y_pre=model.predict(test)

#----------------结果输出----------------
result=pd.read_csv('diabetes提交示例.csv')
result['label']=y_pre
result.to_csv('RF-result-diabetes.csv',index=False)
