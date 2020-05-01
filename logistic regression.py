#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[2]:


df=pd.read_csv('D:\Project\Machine_learning_project1\data2.csv')
df=df.dropna()
print(list(df.shape))
print(list(df.columns))


# In[3]:


df.head()


# In[4]:


df=df.drop(['id', 'program_id','test_id','trainee_id','test_type'],axis=1)


# In[5]:


df.head()


# In[6]:


df["program_type"].unique()


# In[7]:


df["city_tier"].unique()


# In[8]:


df["difficulty_level"].unique()


# In[9]:


df["program_duration"].unique()


# In[10]:


duration_pass=pd.crosstab(df["is_pass"],df["program_duration"])
duration_pass.plot(kind="bar",stacked=True)
plt.show()


# In[11]:


#Data Exploration


# In[12]:


df["is_pass"].value_counts()


# In[13]:


sns.countplot(x="is_pass",data=df)
plt.show()
plt.savefig("count_plot")


# In[14]:


count_fail = len(df[df['is_pass']==0])
count_pass = len(df[df['is_pass']==1])
pct_of_fail = count_fail/(count_fail+count_pass)
print("percentage of no subscription is", pct_of_fail*100)
pct_of_pass = count_pass/(count_fail+count_pass)
print("percentage of subscription", pct_of_pass*100)


# In[15]:


df.groupby("is_pass").mean()


# In[16]:


df.groupby("is_pass").std()


# In[17]:


df.head()


# In[18]:


handicapped=pd.crosstab(df["is_pass"],df["is_handicapped"]).apply(lambda x:x/x.sum(),axis=0)
handicapped


# In[19]:


#since the no. fo handicapped is very small and also the ratio of the pass and fail is in line for handicapped and non-handicapped, the feature is not being considered.


# In[20]:


df=df.drop("is_handicapped",axis=1)


# In[21]:


df.info()


# In[22]:


sns.set_style("darkgrid")


# In[23]:


pd.crosstab(df.gender,df.is_pass).plot(kind="bar")
plt.xlabel("Gender of the trainees")
plt.ylabel("number of trainees")
plt.title("Gender Vs result")
plt.show()


# In[24]:


pd.crosstab(df.city_tier,df.is_pass).plot(kind="bar")
plt.xlabel("city")
plt.ylabel("number of trainees")
plt.title("city vs result")
plt.show()


# In[25]:


pd.crosstab(df.education,df.is_pass).plot(kind="bar")
plt.xlabel("level of education")
plt.ylabel("number of trainees")
plt.title("education vs result")
plt.show()


# In[26]:


pd.crosstab(df.difficulty_level,df.is_pass).plot(kind="bar")
plt.xlabel("level of difficulty")
plt.ylabel("number of trainees")
plt.title("difficulty vs result")
plt.show()


# In[27]:


pd.crosstab(df.program_type,df.is_pass).plot(kind="bar")
plt.xlabel("type of program")
plt.ylabel("number of trainees")
plt.title("type of program vs result")
plt.show()


# In[28]:


pd.crosstab(df.program_duration,df.is_pass).plot(kind="bar")
plt.xlabel("program duration")
plt.ylabel("number of trainees")
plt.title("program duration vs result")
plt.show()


# In[29]:


#to be converted into 3 or 4 groups


# In[30]:


pd.crosstab(df.trainee_engagement_rating,df.is_pass).plot(kind="bar")
plt.xlabel("trainee rating")
plt.ylabel("number of trainees")
plt.title("trainee rating vs result")
plt.show()


# In[31]:


pd.crosstab(df.total_programs_enrolled,df.is_pass,normalize="index")


# In[32]:


df['total_programs_enrolled']=np.where(df['total_programs_enrolled'] ==1, '<=3',df["total_programs_enrolled"])  
for i in range (2,15):
    if i<4:
        df['total_programs_enrolled']=np.where(df['total_programs_enrolled'] ==str(i),'<=3',df['total_programs_enrolled'])
    elif i<7:
        df['total_programs_enrolled']=np.where(df['total_programs_enrolled'] ==str(i),'[4-6]',df['total_programs_enrolled'])
    elif i<10:
        df['total_programs_enrolled']=np.where(df['total_programs_enrolled'] ==str(i),'[7-9]',df['total_programs_enrolled'])
    else:
        df['total_programs_enrolled']=np.where(df['total_programs_enrolled'] ==str(i),'> 9',df['total_programs_enrolled'])
    i+=1


# In[33]:


df.head()


# In[34]:


# an inverse relation is observed. Grouping the programs enrolled and dividing into 4 groups.


# In[35]:


df.info()


# In[36]:


df=df.drop(["program_type","program_duration","gender","city_tier"],axis=1)


# In[37]:


df.head()


# In[38]:


#create dummy variables


# In[39]:


for_dummies=["difficulty_level","education","total_programs_enrolled","trainee_engagement_rating"]
for i in for_dummies:
    df2=pd.get_dummies(df[i],prefix=[i])


# In[40]:


df2.head()
df3=df


# In[41]:


cat_vars=["difficulty_level","education","total_programs_enrolled","trainee_engagement_rating"]
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(df[var], prefix=var,drop_first=True)
    data1=df.join(cat_list)
    df=data1
cat_vars=["difficulty_level","education","total_programs_enrolled","trainee_engagement_rating"]
data_vars=df.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]


# In[42]:


to_keep


# In[43]:


df.head()


# In[51]:


data_final=df[to_keep]
data_final.columns.values


# In[55]:


Y=data_final['is_pass']
X=data_final.drop(['is_pass'],axis=1)


# In[57]:



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1)


# In[59]:


logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)


# In[60]:


predictions=logmodel.predict(X_test)


# In[61]:


from sklearn.metrics import confusion_matrix


# In[62]:


confusion_matrix(y_test,predictions)


# In[64]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)

