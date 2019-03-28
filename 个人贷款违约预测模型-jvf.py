#!/usr/bin/env python
# coding: utf-8

# ## 数据背景

# #### 本数据为一家银行的个人金融业务数据集，可以作为银行场景下进行 个人客户业务分析和数据挖掘的示例。这份数据中涉及到5300个银行 客户的100万笔的交易，而且涉及700份贷款信息与近900张信用卡的数 据。通过分析这份数据可以获取与银行服务相关的业务知识。例如， 提供增值服务的银行客户经理，希望明确哪些客户有更多的业务需求， 而风险管理的业务人员可以及早发现贷款的潜在损失。
# #### 可否根据客户贷款前的属性、状态信息和交易行为预测其贷款违约行为?

# ## 数据含义

# #### 名称      标签         说明    
# #### disp_id  权限号
# #### loan_id  贷款号 (主键)
# #### account_ id账户号
# #### date     发放贷款日期
# #### amount   贷款金额
# #### duration 贷款期限
# #### payments 每月归还额
# #### status   还款状态     A代表合同终止.没问题;B代表合同终止,贷款没有支付;C代表合同处于执行期,至今正常;D代表合同处于执行期,欠债状态。

# ## 分析数据

# In[1]:


import pandas as pd
import numpy as np
import os


# In[2]:


os.getcwd()     #查看当前工作目录


# In[3]:


loanfile = os.listdir()
loanfile
createVar = locals()
for i in loanfile :
    if i.endswith('csv'):
        createVar[i.split('.')[0]]=pd.read_csv(i,encoding='gbk')   
        print (i.split('.')[0])


# #####   loans        : 贷款表
# #####  accounts ：账户表
# #####  disp         ：权限分配表
# #####  clients    ：客户信息表     每条记录代表客户和账户之间的关系，及客户操作账户的权限
# #####  order     ：支付命令表      每条记录代表一个支付交易命令
# #####  trans     ：支付命令表      每条记录代表每个账户每一笔交易记录，1056210条记录
# #####  district   ：人口地区统计表      每条记录代表每个地区人口统计信息，GDP等
# #####  card      ：信用卡信息表      每条记录代表每个账户信用卡信息

# ### 2.1 定义信用评级

# In[4]:


loans['status'].value_counts()


# In[5]:


bad_good = {'A':0,'B':1,'D':1,'C':2}   
loans['bad_good']=loans.status.map(bad_good)


# #####  A代表合同终止.没问题;B代表合同终止,贷款没有支付;C代表合同处于执行期,至今正常;D代表合同处于执行期,欠债状态。

# ##### map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。

# In[6]:


loans.head()


# ### 2.2 合并表格

# #### 2.2.1合并loans表和disp表

# In[7]:


disp.head()    #对比disp表和Loans表，共同存在account_id列


# In[8]:


data2 = pd.merge(loans,disp,on='account_id',how='left')   #合并两张表
data2.shape[0]   #查看dataframe行数


# In[9]:


data2.head()


# #### 2.2.2合并data2表和clients表

# In[10]:


clients.head()    #对比data2表和clients表，共同存在client_id列


# In[11]:


data3 = pd.merge(data2,clients,on='client_id',how='left')
data3.head()


# In[12]:


district.head()  #A1列=district_id列，指地区号


# #### 2.2.3合并data3,district表

# In[13]:


data4 = pd.merge(data3,district,left_on='district_id',right_on='A1',how='left')
data4.head()
#data4[['district_id','A1']].head(3)


# #### 2.2.4合并loans表和trans表
# ##### 求贷款前一年账户的平均余额，余额的标准差，平均收入和平均支出比例

# In[14]:


loans.sort_values(['account_id','date']).head()   #贷款表  ，date:发放贷款的日期


# In[15]:


trans.sort_values('account_id').head()   #交易表：每条记录代表每个账户上的一条记录


# ##### account_id:账户号  type: 借贷类型   operation:交易类型      balance:账户余额   date :交易日期

# In[16]:


data5 = pd.merge(loans[['account_id','date']],trans[['account_id','type','amount','balance','date']],on='account_id')
data5.columns = ['account_id','date','type','amount','balance','t_date']
data5.sort_values(['account_id','t_date'])
data5['date']=pd.to_datetime(data5['date'])
data5['t_date']=pd.to_datetime(data5['t_date'])
data5.head(10)


# #### 清洗数据

# 对账户余额进行清洗,去掉$符号

# In[17]:


data5['balance2']=data5['balance'].map(lambda x : int (''.join(x[1:].split(','))))
data5['amount2']=data5['amount'].map(lambda x : int (''.join(x[1:].split(','))))


# In[18]:


data5.head()


# #### 只取贷款日期前一年的交易记录

# In[19]:


import datetime
data6 =data5[data5.date>data5.t_date][data5.date<data5.t_date+datetime.timedelta(days=365)]


# In[20]:


data6.head()


# #### 求账户余额，账户余额标准差，变异系数

# In[21]:


data7 = data6.groupby('account_id')['balance2'].agg([('avg_balance','mean'),('std_balance','std')]) #账户余额和账户标准差


# In[22]:


data7['cv_balance'] = data7[['avg_balance','std_balance']].apply(lambda x :x[1]/x[0],axis=1)   #变异系数


# ##### 变异系数=标准差/账户余额   ，变异系数越大，说明经济状况越不稳定

# In[23]:


data7.head()


# ### 求收入支出比

# In[24]:


type_dict = {'借':'out','贷':'income'}
data6['type2']=data6.type.map(type_dict)


# In[25]:


data8 = data6.groupby(['account_id','type2'])[['amount2']].sum()


# In[26]:


data8.head()


# In[27]:


data9 = pd.pivot_table(data8,values='amount2',index='account_id',columns='type2') #将该数组转置
data9['out/in'] = data9.apply(lambda x : x[1]/x[0],axis=1)      #求支出收入比，支出/收入


# In[28]:


data9.head()


# ### 合并为总表

# In[29]:


data = pd.merge(data7,data9,on='account_id',how='left')
data = pd.merge(data4,data,on='account_id',how='left')
data=data.sort_values('account_id')


# In[30]:


data.head()


# #### 求贷存比和代收比

# In[31]:


data['r_lb'] = data[['amount','avg_balance']].apply(lambda x : x[0]/x[1] ,axis=1)
data['r_lincome'] =data[['amount','income']].apply(lambda x: x[0]/x[1],axis=1)


# In[32]:


data.head()


# ### 建立模型

# #### 查看数据列名

# In[33]:


data.columns


# #### 创建训练集和测试集

# In[34]:


data_model = data[data.status != 'C']  #C代表合同处于执行期,至今正常。
for_predict = data[data.status == 'C']


# In[35]:


#data_model.shape[0]
#for_predict.shape[0]


# In[36]:


train = data_model.sample(frac=0.7,random_state=1235).copy()
test = data_model[~data_model.index.isin(train.index)].copy()
print('样本数量：%i \n训练样本数量：%i \n测试样本数量：%i'%(len(data_model),len(train),len(test)))


# In[37]:


#train.head()
test.head()


# ### 建模(使用逻辑回归模型)

# #### 向前逐步法

# In[38]:


def forward_select(data, response):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    while remaining:
        aic_with_candidates=[]
        for candidate in remaining:
            formula = "{} ~ {}".format(
                response,' + '.join(selected + [candidate]))
            aic = smf.glm(
                formula=formula, data=data, 
                family=sm.families.Binomial(sm.families.links.logit)
            ).fit().aic
            aic_with_candidates.append((aic, candidate))
        aic_with_candidates.sort(reverse=True)
        best_new_score, best_candidate=aic_with_candidates.pop()
        if current_score > best_new_score: 
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print ('aic is {},continuing!'.format(current_score))
        else:        
            print ('forward selection over!')
            break
            
    formula = "{} ~ {} ".format(response,' + '.join(selected))
    print('final formula is {}'.format(formula))
    model = smf.glm(
        formula=formula, data=data, 
        family=sm.families.Binomial(sm.families.links.logit)
    ).fit()
    return(model)


# #### 选择特征

# In[39]:


data.columns


# In[40]:


candidates = ['bad_good', 'A1', 'GDP', 'A4', 'A10', 'A11', 'A12','amount', 'duration',
       'A13', 'A14', 'A15', 'a16', 'avg_balance', 'std_balance',
       'cv_balance', 'income', 'out', 'out/in', 'r_lb', 'r_lincome']
data_for_select = train[candidates]
lg_m1 = forward_select(data=data_for_select,response='bad_good')
lg_m1.summary().tables[1]


# In[ ]:


import sklearn.metrics as metrics
import matplotlib.pyplot as plt
fpr, tpr, th = metrics.roc_curve(test.bad_good, lg_m1.predict(test))
plt.figure(figsize=[6, 6])
plt.plot(fpr, tpr, 'b--')
plt.title('ROC curve')
plt.show()


# In[ ]:


print('AUC = %.4f' %metrics.auc(fpr, tpr))


# In[ ]:


for_predict['prob']=lg_m1.predict(for_predict)
for_predict[['account_id','prob']].head()

