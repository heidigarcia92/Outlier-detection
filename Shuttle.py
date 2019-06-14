#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.svm import OneClassSVM
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, minmax_scale
from operator import itemgetter


# In[2]:


shuttle = pd.read_csv('../input/shuttleX2.csv', header=None, sep=',', quoting=False).astype('float64')
shuttle_y = pd.read_csv('../input/shuttleY.csv', header=None)
shuttle.head()


# In[3]:


shuttle = scale(shuttle)


# In[4]:


shuttle_y.iloc[:, 0].value_counts().plot.bar()
plt.savefig('img.png')
plt.show()


# In[5]:


shuttle_y[0].value_counts()


# In[6]:


gamma_values, err_values_gamma = [], []
for g in np.linspace(0.0000015, 0.00015, 10):
    onesvm = OneClassSVM(nu= shuttle_y.mean(), gamma=g)
    onesvm.fit(shuttle)
    yhat = onesvm.predict(shuttle)
    yhat = ((yhat - 1) * -1) / 2
    acc = accuracy_score(shuttle_y, yhat)
    err = 1 - acc
    gamma_values.append(g)
    err_values_gamma.append(err)


# In[7]:


plt.subplots(figsize=(10, 5))
plt.plot(gamma_values, err_values_gamma, 'o-')
plt.xlabel('gamma')
plt.ylabel('error')
plt.show()


# In[8]:


max(err_values_gamma), min(err_values_gamma)


# In[9]:


## Unsupervised (OneClassSVM)


# In[10]:


onesvm = OneClassSVM(nu = shuttle_y.mean(), gamma=0.00002)
onesvm.fit(shuttle)


# In[11]:


yhat = onesvm.predict(shuttle)
yhat = ((yhat - 1) * -1) / 2


# In[12]:


print(accuracy_score(shuttle_y, yhat))
print(confusion_matrix(shuttle_y, yhat))
print(classification_report(shuttle_y, yhat))


# In[13]:


sns.heatmap(confusion_matrix(shuttle_y, yhat), annot=True)
plt.savefig('heatmap_shuttle.png')


# In[14]:


tsne = TSNE(n_components=2)
shuttle2 = tsne.fit_transform(shuttle)


# In[15]:


shuttle2 = pd.DataFrame(shuttle2, columns=['x','y'])
shuttle2['ytrue'] = shuttle_y[0]
shuttle2['yhat']  = yhat
shuttle_y.shape, shuttle2.shape


# In[16]:


sns.lmplot(data=shuttle2, x='x', y='y', hue='ytrue', fit_reg=False)


# In[17]:


sns.lmplot(data=shuttle2, x='x', y='y', hue='yhat', fit_reg=False)


# In[18]:


## Supervised (SVC)


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(shuttle, shuttle_y, test_size=0.3, shuffle=False, random_state=13)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=False, random_state=13)
svm = SVC(gamma = 'auto')
svm.fit(X_train, y_train.values.ravel())


# In[20]:


gamma_values, err_gamma = [], []
for g in np.linspace(0.000005, 0.0005, 10):
    svm = SVC(gamma=g, C=1.0)
    svm.fit(X_train, y_train.values.ravel())
    yhat = svm.predict(X_val)
    acc = accuracy_score(y_val, yhat)
    err = 1 - acc
    gamma_values.append(g)
    err_gamma.append(err)


# In[21]:


gamma_c_values, err_gamma_c = [], []
for g in np.linspace(10e-5, 0.01, 5):
    for c in np.linspace(0.4, 10, 10):
        svm = SVC(gamma=g, C=c)
        svm.fit(X_train, y_train.values.ravel())
        yhat = svm.predict(X_val)
        acc = accuracy_score(y_val, yhat)
        err = 1 - acc
        gamma_c_values.append((g,c))
        err_gamma_c.append(err)


# In[22]:


gamma_v = list(map(itemgetter(0), gamma_c_values))
c_v = list(map(itemgetter(1), gamma_c_values))


# In[23]:


fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(gamma_v, c_v, err_gamma_c, shade=True, color='green')
ax.scatter(gamma_v, c_v, err_gamma_c, color='red')
ax.set_xlabel('gamma')
ax.set_ylabel('C')
ax.set_zlabel('err')


# In[24]:


max(err_gamma_c), min(err_gamma_c)


# In[25]:


plt.subplots(figsize=(10, 5))
plt.plot(gamma_values, err_gamma, 'o-')
plt.xlabel('gamma')
plt.ylabel('error')
plt.show()


# In[26]:


max(err_gamma), min(err_gamma)


# In[27]:


best_gamma_c = min(zip(gamma_c_values, err_gamma_c), key=lambda p: p[1])[0]
svm = SVC(gamma=best_gamma_c[0], C=best_gamma_c[1])
svm.fit(X_train, y_train.values.ravel())


# In[28]:


yhat_shuttle = svm.predict(X_test)


# In[29]:


print(accuracy_score(y_test, yhat_shuttle))
print(confusion_matrix(y_test, yhat_shuttle))
print(classification_report(y_test, yhat_shuttle))


# In[30]:


sns.heatmap(confusion_matrix(y_test, yhat_shuttle), annot=True)
plt.savefig('heatmap_shuttle.png')


# In[31]:


test_shuttle2 = TSNE(n_components=2).fit_transform(X_test)
test_shuttle2 = pd.DataFrame(test_shuttle2, columns=['x','y'])
test_shuttle2['ytrue'] = y_test.values
test_shuttle2['yhat'] = yhat_shuttle
sns.lmplot(data=test_shuttle2, x='x', y='y', hue='ytrue', fit_reg=False)


# In[32]:


sns.lmplot(data=test_shuttle2, x='x', y='y', hue='yhat', fit_reg=False)

