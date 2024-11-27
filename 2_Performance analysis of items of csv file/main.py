import requests
import pandas as pd
import numpy as np
import json
import os
import warnings
import tkinter as tk
from tkinter import *
import tkinter.messagebox
warnings.simplefilter(action='ignore', category=FutureWarning)

# 创建主窗口
root = tk.Tk()
root.title("Performance")

#设置数据存储和标题
var = tk.StringVar()
var2=tk.StringVar()
label = tk.Label(root, text="设置产品")
label.grid(row=1,column=0, pady='5', ipadx='2', ipady='2')
label21 = tk.Label(root, text="设置分支")
label21.grid(row=1,column=1, pady='5', ipadx='2', ipady='2')
label22 = tk.Label(root, text="请将csv文件放置于同目录下")
label22.grid(row=7,column=1, pady='5', ipadx='2', ipady='2')
label22 = tk.Label(root, text="最近新增:最近报告中新增衰退，或原有基础上更加衰退的，且未修复\n高优先，超时比例过大（大于50%），或波动剧烈\n中优先:不稳定或稳定但长期不过脚本")
label22.grid(row=8,column=1, pady='5', ipadx='2', ipady='2')


#设置产品选项和按钮位置
gap=tk.Radiobutton(root, indicatoron=0, text="gap", variable=var, value="gap")
gst=tk.Radiobutton(root, indicatoron=0,text="gst", variable=var, value="gst")
gmep=tk.Radiobutton(root, indicatoron=0,text="gmep", variable=var, value="gmep")
gap.grid(row=2, column=0,pady='5', ipadx='2', ipady='2')
gst.grid(row=4, column=0,pady='5', ipadx='2', ipady='2')
gmep.grid(row=6, column=0,pady='5', ipadx='2', ipady='2')

#设置分支选项和按钮位置
pre=tk.Radiobutton(root, indicatoron=0, text="feature/gnc_pre_integration", variable=var2, value="feature/gnc_pre_integration")
master=tk.Radiobutton(root, indicatoron=0,text="master", variable=var2, value="master")
release=tk.Radiobutton(root, indicatoron=0,text="release/release_v_0_44", variable=var2, value="release/release_v_0_44")
pre.grid(row=2, column=1,pady='5', ipadx='2', ipady='2')
master.grid(row=4, column=1,pady='5', ipadx='2', ipady='2')
release.grid(row=6, column=1,pady='5', ipadx='2', ipady='2')

#设置参数输入框控件
label = tk.Label(root, text="对本次衰退点的超时阈值")
label.grid(row=1, column=2,pady='0', ipadx='2', ipady='2')
rate1=tk.Entry(root)
rate1.insert(0,'20')
rate1.grid(row=2, column=2,pady='0', ipadx='2', ipady='2')

label2 = tk.Label(root, text="对衰退点历史均值的超时阈值")
label2.grid(row=3, column=2,pady='0', ipadx='2', ipady='2')
averate1=tk.Entry(root)
averate1.insert(0,'20')
averate1.grid(row=4, column=2,pady='0', ipadx='2', ipady='2')

label3 = tk.Label(root, text="对衰退点历史数据的标准差阈值")
label3.grid(row=5, column=2,pady='0', ipadx='2', ipady='2')
STD1=tk.Entry(root)
STD1.insert(0,'0.03')
STD1.grid(row=6, column=2,pady='0', ipadx='2', ipady='2')

label3 = tk.Label(root, text="开始计算日期")
label3.grid(row=7, column=2,pady='0', ipadx='2', ipady='2')
datestart1=tk.Entry(root)
datestart1.insert(0,'2024-09-10')
datestart1.grid(row=8, column=2,pady='0', ipadx='2', ipady='2')

label4 = tk.Label(root, text="csv文件名")
label4.grid(row=9, column=2,pady='0', ipadx='2', ipady='2')
filename1=tk.Entry(root)
filename1.insert(0,'full_result.csv')
filename1.grid(row=10, column=2,pady='0', ipadx='2', ipady='2')


#设置主按钮功能
def button_click():
    AIM = var.get();filename = filename1.get();branch = var2.get()
    rate = float(rate1.get());averate = float(averate1.get());STD = float(STD1.get());datestart = datestart1.get();excel = 'regression.xlsx';

    GAP = 'GAP_Performance\\';GMEP = 'Performance\\';GST = 'GST_Performance\\';GMEP2 = 'Performance_Collaboration\\'
    if 'gap' in AIM:
        product = 'gap';CASE = GAP;Pbranch = branch + "_" + AIM
    elif 'gmep' in AIM:
        product = 'gmep';CASE = GMEP;Pbranch = branch + "_" + AIM
    else:
        product = 'gst';CASE = GST;Pbranch = branch + "_" + AIM

    if 'master' in branch:
        Pbranch = branch
    if 'release/release_v_0_44' in branch:
        Pbranch = branch

    # regression读取csv提取衰退行数据，删除非衰退数据
    current_path = os.getcwd()
    csv_files = [f for f in os.listdir(current_path) if f.endswith('.csv')]
    filename = os.path.join(current_path, filename)
    regression = pd.read_csv(os.path.join(current_path, filename), on_bad_lines='warn', encoding='ANSI')
    indexNames = regression[regression["结果"] != "衰退"].index
    regression.drop(indexNames, inplace=True)

    # 删除部分多余数据。csv的表头‘标准时间’带空格，通过重新命名来除掉空格，否则后续apply计算识别不了
    # 部分数据标准时间未设置，其对应超时上限一般为0.1，因此这里修改为0.11，数据正常计算，在末尾替换为无数据标识
    regression.drop(columns=['业务场景', '竞品值'], inplace=True)
    regression.rename(columns=lambda x: x.strip(), inplace=True)
    regression['标准时间'].replace(0, 0.11, inplace=True)
    if 'gst' in AIM:
        regression.drop(columns=['是否达标'], inplace=True)

    # 计算差值百分比并对高于阈值增加列进行标签
    regression['超时百分比'] = regression.apply(lambda x: (x.实测值 - x.标准时间) / x.标准时间 * 100, axis=1)
    regression['分析结论'] = 0;regression['平均差值'] = 0;regression['均差分析'] = 0;regression['总体方差'] = 0;regression['方差分析'] = 0;
    regression['优先级'] = '低';regression.loc[regression['超时百分比'] > rate, '分析结论'] = '超时较大'

    # 将两列数据分别转化为数组，以便接口传参
    test = regression['测试脚本名称'].to_numpy()
    point = regression['性能检查点'].to_numpy()
    l = len(regression)

    # 对每一条衰退点进行接口传参，获取历史性能数据进行计算
    i = 0;body = {};d = [];
    while i < l:
        payload = {'product': product, 'daterangepicker_start': datestart, 'branch': Pbranch, 'case': CASE + test[i],
                   'checkpoint': point[i]}
        headers = {'X-Requested-With': 'XMLHttpRequest'}
        proxies = {'http': None, 'https': None}
        url = f'http://10.5.67.94/metrics/performance?'
        response = requests.get(url=url, headers=headers, json=body, params=payload, proxies=proxies)
        data1 = json.loads(response.text, strict=False)

        # 机电特殊，两种传参，Performance_Collaboration和Performance，因此要对失败的再用下一个参数再发送一遍
        if 'selectTestData' in data1:
            payload = {'product': product, 'daterangepicker_start': datestart, 'branch': Pbranch,
                       'case': GMEP2 + test[i], 'checkpoint': point[i]}
            response = requests.get(url=url, headers=headers, json=body, params=payload, proxies=proxies)
            data1 = json.loads(response.text, strict=False)
        data2 = json.loads(data1['duration'])

        # 对单条返回值进行处理，抽取性能数据形成数组进行计算
        n = 0;data = 0;d = []
        while n <= len(data2) - 1:
            data3 = json.loads(data1['duration'])[n]['y']
            data = (data + data3)
            d.append(data3)
            n = n + 1
        # 对部分无数据的脚本，不进行计算和标记
        if len(data2) != 0:
            value2 = (data / len(data2) - int(regression.iloc[i, 2])) / regression.iloc[i, 2] * 100
            regression.iloc[i, 10] = value2
            regression.iloc[i, 12] = np.var(d)
            if value2 > averate:
                regression.iloc[i, 11] = '长期不过'
            if np.var(d) > STD:
                regression.iloc[i, 13] = '不稳定'
        i = i + 1

    # 整理表格，新增列判断优先级，导出excel
    regression['标准时间'].replace(0.11, '无数据', inplace=True)
    regression.drop(columns=['超时标准', '容忍值'], inplace=True)

    regression.loc[(regression['方差分析'] == '不稳定') & (regression['分析结论'] == '超时较大'), '优先级'] = '中'
    regression.loc[(regression['方差分析'] == 0) & (regression['均差分析'] == '长期不过'), '优先级'] = '中'
    regression.loc[(regression['均差分析'] == 0) & (regression['方差分析'] == '不稳定') & (regression['分析结论'] == '超时较大'), '优先级'] = '高'
    regression.loc[(regression['超时百分比'] >= 50), '优先级'] = '高'
    regression.loc[(regression['均差分析'] == 0) & (regression['方差分析'] == '不稳定') & (regression['分析结论'] == '超时较大'), '优先级'] = '最近新增'

    # 对本身就衰退的，更加衰退进行标记
    m=0;
    while m<l:
        value3=(regression.iloc[m,6]/regression.iloc[m,8])
        if value3>1.5:
            regression.iloc[m,12] = '最近新增'
        m=m+1

    # 颜色控制
    def color_background(col):
        if col == "低":
            return 'background-color:#FAFAD2'
        elif col == '最近新增':
            return 'background-color:#F08080'
        elif col == '高':
            return 'background-color:#FFA07A'
        elif col == '中':
            return 'background-color:#FFB6C1'
        else:
            return ''

    styled_df = regression.style.applymap(color_background)
    excelname = os.path.join(current_path, excel)
    styled_df.to_excel(excelname, sheet_name='性能衰退分析', float_format='%.2f', na_rep=0)

    print(regression)

    tkinter.messagebox.showinfo("消息", "跑完啦")

#创建主按钮
button = tk.Button(root, text="开跑！", command=button_click)
button.grid(row=10, column=1,pady='5', ipadx='2', ipady='2')


root.mainloop()
