#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Introduction around Maths for Datascience:

# NUMPY(Numerical Python)

# is the segment which will be acting as the foundation of Data Science , with it's three pillars.

# Subset of Mathematics that will be used in implement all the techniques for exploration, modelling and final response expecting in terms of prediction or forecasting depending upon the problem statement.

# Three pillars :

# 1. Linear algebra: field of mathematics ,for solving the linear equations and finding the unkown values.

# Matrix , as the enitre dataset is represent in the form of matrix.

# operations related to matrix: Transponse,dot,scalerproduct, eigen values, eigen vectors

# cost of computation(with which dataset it is higher)

# Dataset 1---> 100columns , 1000 rows---> ML ---> good cost of computation---> dimensionality reduction--->5 columns.

# Dataset 2---> 10 columns , 1000 rows.



# 2. Statistics: The branch of mathematics is delaing with collection,analysis, preprocessing,inferencing the useful insights out the raw the data.

# Machine learning models will be called as statistical Models ----> Applying the results the of statistics ---> to solve the problems--> model building .

# Analyzing the raw data ,summarising it , model building ---> model training ---> Expecting out of the model.

# Descriptive statistics : central tendencies ------> Mean,Median ,Mode,ranges ,min,max, Quatriles -----> use to understand and analyse.

# inferential statisitcs: Probability distribution function,poison,normal distribution,regression ,classification---> prediction and forecast.

# 3. Probability:  possibility

#ADF ,hypothesis testing 

# to predict whether today there is rain or not:

# Yes---> 0.7   NO ---> 0.3


# In[2]:


# will be supporting the Machine Learning algorithms.---> Linear regrssion , Logistic Regression , K means clustering , 
# Decision Tree Classifier (classification)


# In[3]:


# Heart Disease Dataset

import pandas as pd


# In[4]:


heart=pd.read_csv(r'C:\Users\PCC\Downloads\heart_disease_uci.csv')


# In[5]:


heart.head()


# In[6]:


heart.describe()


# In[5]:


heart.head()


# In[6]:


# 100-----> persons.----> height ,weight of the person

# Q1 -->25%
#Q2---> 50%
#Q3---> 75%

#  25 ---->165 cm 

# 50 ----> median ----> mid point---> 172cm 

#75-----> Q3-----> 0-183


# In[7]:


# 1 2 3 4 5 6 7 8 90----> Median 

# mean =5, median =5

#mean = 14, median 


# In[8]:


1+2+3+4+5+6+7+8+90


# In[9]:


126/8


# In[10]:


126/9


# In[11]:


# Linear algebra: study of vectors and linear computaions.


# Vectors : it's the most Mathematical fundamental unit or object in machine learning.
# vectors is used to represent the attributes of different entities.(age,gender,credit , number of cylinder).

# vector : ordered finite list of information.


# In[12]:


# snow , fire , eat , apple :

# Machine -------> Word to vector----> 0---1,0---1,0---1,0-----1----> implememnted


# In[13]:


#matrix: ordered collection in vectors arranged in two dimesional(row n column ) structure.


# 1 dimensional array----> vector

# 2 dimensional array ----> Matrix.


# In[14]:


# creating a matrix , matrix name to be depicted in Upper Case.


# In[15]:


import numpy as np


# In[18]:


x=np.array([1,2,3,4])


# In[19]:


print(x)


# In[20]:


x.ndim   # Dimension of the array


# In[21]:


y=np.array([[1,2],[3,4]])


# In[22]:


y


# In[23]:


y.ndim


# In[24]:


x=x.reshape(2,2)


# In[25]:


x.ndim


# In[26]:


x


# In[27]:


# Matrix : helps to create the Matrix in python 


m=np.matrix([[4,5],[9,8]])


# In[28]:


m


# In[29]:


m.ndim


# In[30]:


# Accesing and extracting 


n= np.array([1,2,3,4,5,6,7,8,9]).reshape(3,3)


# In[31]:


n


# In[32]:


n.ndim


# In[33]:


n


# In[34]:


# Synatx:

# individual : indexing

# Matrix[Row index, column index]

# Multiple element :

# Matrix [RS:RE,CS:CE]


# In[35]:


n


# In[36]:


n[2,2]


# In[37]:


n[1,0]


# In[38]:


n[0,1]


# In[39]:


# slicing

n[2,0:2]


# In[40]:


n[0:2,0:2]


# In[41]:


# Airthmetic operations :

# Addition of the Matrix:

# Syntax:

# m1=[a00,a01]   + m2=[b00,b01]   = m3[a00+b00,a01+b01]
#    [a10,a11]        [b10,b11]       [a10+b10   a11+b11]


# In[42]:


m1=np.array([12,45,78,96,3,2,25,45,37]).reshape(3,3)


# In[43]:


m1


# In[44]:


m2=np.array([63,95,47,86,32,14,2,74,5]).reshape(3,3)


# In[45]:


m2


# In[46]:


m3 = m1 + m2


# In[47]:


m3


# In[48]:


m4=np.array([1,2,3,4]).reshape(2,2)


# In[49]:


m4


# In[50]:


m4.ndim


# In[51]:


m3.ndim


# In[52]:


m3 + m4


# In[54]:


# Subtraction operation 



# Syntax:

# M1=[a00,a01]   + M2=[b00,b01]   = M3[a00-b00,a01-b01]
#    [a10,a11]        [b10,b11]       [a10-b10   a11-b11]

m5=m1-m2


# In[55]:


m2


# In[56]:


m1


# In[57]:


m5


# In[58]:


# Matrix -Scaler multiplication :

# M =[a00,a01] * n =  result[n*a00,n*a01]
#    [a10,a11]              [n*a10,n*a11]


# In[59]:


m1*10


# In[60]:


# Dot product: Matrix -vector multiplication 

# Matrix to be multiplied with vector which yield another vector.


# M=[a00,a01]  * V=[b0] = result vector
#   [a10,a11]      [b1]


# result vector= b0[a00] +b1[a01]= [b0a00+b1a01]=NV[V0]
#                  [a10]    [a11]  [b0a10+b1a11]   [v1]


# In[61]:


m2 # Matrix


# In[62]:


m2.ndim


# In[63]:


v2=np.array([10,20,30])


# In[64]:


v2.ndim # vector


# In[65]:


# Dot product :

# np .dot() other way @

np.dot(m2,v2)


# In[66]:


# @

nv=m2 @ v2


# In[67]:


nv


# In[68]:


nv.ndim


# In[69]:


# Matrix Multiplication :

# M1=[a00,a01]  * M2=[b00,b01]= result[a00*b00+a01*b10,    a00*b01+a01*b11]
#    [a10,a11]       [b10,b11]        [a10*b00+a11*b10,    a10*b01+a11*b11]


# Matrix Matrix Multiplication 

# 1.If both the square matrix have the same shape then the matrix multiplication will be valid.

# 2. if column number of first matrix willl be equal to the row number of the second matrix then the multiplication will be valid .

# M1 = 2*2 =p*q

# M2=2*3= n*m

# q==n-----> shape of the resultant matrix= p*m= 2*3


# In[70]:


n1=np.array([1,2,3,4]).reshape(2,2)


# In[71]:


n1


# In[72]:


n1.ndim


# In[73]:


n2=np.array([5,6,7,8]).reshape(2,2)


# In[74]:


n2


# In[75]:


n2.ndim


# In[76]:


n1 * n2


# In[77]:


n1@n2  # Matrix Multiplication will be implemented with the same.


# In[78]:


#e1=1*5+2*7=5+14=19
#e2=1*6+8*2=6+16=22
#e3=3*5+4*7=43
#e4=3*6+4*8=50


# In[79]:


n2


# In[80]:


n3=np.array([1,2,3,4,5,6]).reshape(2,3)


# In[81]:


n3


# In[82]:


n2@n3


# In[83]:


# Problem Statement: 

# M1=[45,67,89
#    100,45,32
#    78,92,48]


# M2=[76,90,65
#     101,69,42
#      56,12,34]

# V4=[38,
#     42
#     21]


# Addition 
# Scaler 
# Dot product

# Mulitplication.


# In[84]:


# Transform Matrix


# In[85]:


m7 = np.random.randint(87,size=(3,3))


# In[86]:


m7


# In[87]:


# 1st Way
m7.transpose()


# In[88]:


# 2nd Way
np.transpose(m7)


# In[89]:


# 3rd Way
m7.T


# In[90]:


# Identity or unit Matrix

# 1st Way


np.eye(3,3)


# In[91]:


# 2nd way


np.identity(3, dtype='int')


# In[92]:


# matrix matrix multiplication


# In[93]:


i = np.identity(3, dtype='int')


# In[94]:


# m6*i=m


# In[95]:


m7@i


# In[96]:


# Determinant of the 2*2 matrix


# In[97]:


m2 = np.array([1,2,3,4]).reshape(2,2)


# In[98]:


m2


# In[99]:


# functionality of nympy

np.linalg.det(m2)


# In[100]:


# traditional method

# +(1*4)-(2*3)

4-6


# In[101]:


# calculating diterminating with 3*3 matrix:


m3 = np.array([1,2,3,4,5,6,7,8,9]).reshape(3,3)


# In[102]:


m3


# In[103]:


np.linalg.det(m3)


# In[104]:


# Adjoint of the matrix : 

# Transpose of the cofactor matrix---- > adjoint or adjugate matrix :


# In[105]:


# Inverse of matrix


# In[106]:


m10 = np.array([34,45,56,67,78,89,90,91,12]).reshape(3,3)


# In[107]:


m10


# In[108]:


inv = np.linalg.inv(m10)


# In[109]:


inv


# In[110]:


m10@inv


# In[111]:


# Matrix Multiplication

# Matrix Division -----> 


# In[112]:


d = m10@inv


# In[113]:


inv@d


# In[114]:


90/5


# In[115]:


90*(1/5)


# In[116]:


m10


# In[117]:


# Trace of matrix: 
m10


# In[118]:


m10.trace()


# In[119]:


# Hands on for the matrix operations
# Multiplication of m1 and m2
# Compote the vector formula by scaler multiplication of v1 with m1 and m2 respectively
# Determinate of m1 and m2 
# Inverse of m1 and m2 
# Trace of m1 and m2 
# Adjoint matrix of m1 and m2


# In[120]:


m1=np.array([12,45,78,96,3,2,25,45,37])


# In[121]:


m1


# In[122]:


m1=np.array([12,45,78,96,3,2,25,45,37]).reshape(3,3)


# In[123]:


m1


# In[124]:


# Trace of m1
m1.trace()


# In[ ]:





# In[125]:


m2=np.array([63,95,47,86,32,14,2,74,5])


# In[126]:


m2


# In[127]:


m2=np.array([63,95,47,86,32,14,2,74,5]).reshape(3,3)


# In[128]:


m2


# In[129]:


# Trace of m2
m2.trace()


# In[130]:


# Multiplication of m1 and m2

m1 * m2


# In[131]:


# Inverse of m1 and m2

np.linalg.inv(m1)


# In[132]:


np.linalg.inv(m2)


# In[133]:


# Determinate of m1 and m2


# In[134]:


np.linalg.det(m1)


# In[135]:


np.linalg.det(m2)


# In[136]:


# Determinate of m10 and m12 


# In[137]:


m10 = np.array([47,5,60,20,3,40,7,80,9]).reshape(3,3)


# In[138]:


m10


# In[139]:


np.linalg.det(m10)


# In[140]:


m12 = np.array([89,56,34,2,23,34,45,5,60,]).reshape(3,3)


# In[141]:


m12


# In[142]:


np.linalg.det(m12)


# In[143]:


np.linalg.inv(m10)


# In[144]:


np.linalg.inv(m12)


# In[145]:


# Transpose

w1 = np.array([[1,2,3],[4,5,6],[8,9,10]])
w2 = w1.transpose()


# In[146]:


w1


# In[147]:


w2


# In[148]:


# Compote the vector formula by scaler multiplication of v1 with m1 and m2 respectively


# In[149]:


m1


# In[150]:


m2


# In[151]:


v1 = np.array([45,67,78])


# In[152]:


v1


# In[153]:


v1 * m1


# In[154]:


v1 * m2


# In[155]:


# Adjoint matrix of m1 and m2


# In[156]:


# # Adjoint matrix of m1
m1


# In[157]:


# cofactor = [21,3502,4245,]
#            [-1845,-1506,-585,]
#            [-144,-7464,-4284 ]

#            [21,+1845,-144],
#            [-3502,-1506,+7464],
#            [4245,+585,-4284]


# In[158]:


# Transpose
co =np.array([[21,-3502,4245],[+1845,-1506,+585],[-144,+7464,-4284]])
tra =co.transpose()


# In[159]:


tra


# In[160]:


# Adjoint matrix of m2
m2


# In[161]:


# cofactor = [-876,402,6300,],
#            [-3003,221,4472,],
#            [-174,-3160,-6154]



#           [-876,+3003,-174],
#           [-402,221,+3160],
#           [6300,-4472,-6154]


# In[162]:


co2 =np.array([[-876,-402,6300],[+3003,221,-4472],[-174,+3160,-6154]])
tra2 = co2.transpose()


# In[163]:


tra2


# In[164]:


# Inverse = Adjoint of Matrix/ determinant of the matrix .

#  Inverse* determinanant of the matrix= Adjoint matrix.


# In[16]:


m1inv*m1d


# In[19]:


# Introduction to Eigen values and vactor ------->

# Eigen Vector -------->

# Eigen values -------->

