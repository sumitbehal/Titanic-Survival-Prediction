
# coding: utf-8

# In[75]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('train.csv')

#Classification on the basis of class

survived_class = df[df['Survived']==1]['Pclass'].value_counts()
dead_class = df[df['Survived']==0]['Pclass'].value_counts()
df_class=pd.DataFrame([survived_class,dead_class])
df_class.index=['Survived','Dead']
df_class.columns=['Class 1','Class 2','Class 3']
print(df_class)
df_class.plot(kind='bar')
plt.ylabel('No. of people',size=15,color='green')
plt.xlabel('Survival',size=20,color='blue')
plt.show()
Class1_survived= df_class.iloc[0,0]/df_class.iloc[:,0].sum()*100
Class2_survived = df_class.iloc[0,1]/df_class.iloc[:,1].sum()*100
Class3_survived = df_class.iloc[0,2]/df_class.iloc[:,2].sum()*100
print('Percentage of Class1 passenger survived is ',round(Class1_survived),'%')
print('Percentage of Class2 passenger survived is ',round(Class2_survived),'%')
print('Percentage of Class3 passenger survived is ',round(Class3_survived),'%')

#Classification on the basis of the gender

survived_gender=df[df['Survived']==1]['Sex'].value_counts()
dead_gender=df[df['Survived']==0]['Sex'].value_counts()
df_gender=pd.DataFrame([survived_gender,dead_gender])
df_gender.columns=['Survived','Dead']
df_gender.index=['Female','Male']
print(df_gender)
df_gender.plot(kind='bar')
plt.ylabel('No. of people',size=15,color='green')
plt.xlabel('Sex',size=20,color='blue')
plt.show()
female_survived=df_gender.iloc[0,0]/df_gender.iloc[0,:].sum()*100
male_survived=df_gender.iloc[1,0]/df_gender.iloc[1,:].sum()*100
print('Percentage of male passengers survived is ',round(male_survived),'%')
print('Percentage of female passengers survived is ',round(female_survived),'%')

#Classififcation on the basis of embarkment

survived_embark=df[df['Survived']==1]['Embarked'].value_counts()
dead_embark=df[df['Survived']==0]['Embarked'].value_counts()
df_embark=pd.DataFrame([survived_embark,dead_embark])
df_embark.index=['Survived','Dead']
print(df_embark)
s_embark=df_embark.iloc[0,0]/df_embark.iloc[:,0].sum()*100
c_embark=df_embark.iloc[0,1]/df_embark.iloc[:,1].sum()*100
q_embark=df_embark.iloc[0,2]/df_embark.iloc[:,2].sum()*100
df_embark.plot(kind='bar')
plt.ylabel('No. of people',size=15,color='green')
plt.xlabel('Embarkment Classification',size=20,color='blue')
plt.show()
print('Percentage of Embark S that survived :',round(s_embark),'%')
print('Percentage of Embark C that survived :',round(c_embark),'%')
print('Percentage of Embark Q that survived :',round(q_embark),'%\n')

#Classification on the basis of Age
df["Age"] = df["Age"].fillna(0)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
df['AgeGroup'] = pd.cut(df["Age"], bins, labels = labels)
print(df[['AgeGroup','Survived']].groupby(['AgeGroup'],as_index=False).mean())
sns.barplot(x="AgeGroup",y="Survived",data=df)
plt.xlabel('AgeGroup',color='blue',size=18)
plt.ylabel('Survival Rate',color='green',size=18)
plt.title('Babies got the maximum survival rate',color='Black',size=20)
plt.show()


#Classification on the basis of Parch
print(df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Parch', ascending=True))
sns.barplot(x="Parch",y="Survived",data=df)
plt.xlabel('Parents/Children',color='blue',size=18)
plt.ylabel('Survival Rate',color='green',size=18)
plt.show()


#Classification on the basis of Fare
df['Fare']=df['Fare'].fillna(df['Age'].mean())
bins=[0,100,250,600]
labels=['Economic Class','Business Class','First Class']
df['Class']=pd.cut(df['Fare'],bins,labels=labels)
print(df[['Class','Survived']].groupby(['Class'],as_index=False).mean())
sns.barplot(x="Class",y="Survived",data=df)
plt.xlabel('Class',color='blue',size=18)
plt.ylabel('Survival Rate',color='green',size=18)
plt.title('First Class Passengers got the maximum survival rate',color='Black',size=20)
plt.show()
print(df.head(10))

