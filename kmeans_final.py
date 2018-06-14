
# coding: utf-8

# In[2]:


from pandas import read_excel, merge


# In[3]:


from numpy import arange


# In[4]:


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[5]:


km = KMeans(n_clusters=3, init='k-means++', 
            max_iter=100, n_init=1, verbose=0, random_state=3425)


# In[6]:


import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
plotly.offline.init_notebook_mode()


# In[7]:


import pandas as pd
df=pd.read_csv("results.csv")


# In[8]:


df


# In[9]:


cluster=KMeans(n_clusters=3)


# In[10]:


df["cluster"]= cluster.fit_predict(df[df.columns[2:]])


# In[11]:


df


# In[12]:


pca=PCA(n_components=2)
df['x']=pca.fit_transform(df[df.columns[1:]])[:,0]
df['y']=pca.fit_transform(df[df.columns[1:]])[:,1]
df=df.reset_index()


# In[13]:


df


# In[23]:


df=df[["x","y","cluster"]]


# In[24]:


df


# In[25]:


trace0= go.Scatter(x=df[df.cluster==0]["x"],
                   y=df[df.cluster==0]["y"],
                   name="cluster1")
trace1= go.Scatter(x=df[df.cluster==1]["x"],
                   y=df[df.cluster==1]["y"],
                   name="cluster2")
trace2= go.Scatter(x=df[df.cluster==2]["x"],
                   y=df[df.cluster==2]["y"],
                   name="cluster3")
data=[trace0,trace1,trace2]
iplot(data)


# In[17]:


df


# In[18]:


df1=df.head()


# In[27]:


cluster1=df.loc[df['cluster'] == 0]
cluster2=df.loc[df['cluster'] == 1]
cluster3=df.loc[df['cluster'] == 2]


# In[28]:


scatter1 = dict(
    mode = "markers",
    name = "Cluster 1",
    type = "scatter3d",    
    x = cluster1.as_matrix()[:,0], y = cluster1.as_matrix()[:,1], z = cluster1.as_matrix()[:,2],
    marker = dict( size=2, color='green')
)
scatter2 = dict(
    mode = "markers",
    name = "Cluster 2",
    type = "scatter3d",    
    x = cluster2.as_matrix()[:,0], y = cluster2.as_matrix()[:,1], z = cluster2.as_matrix()[:,2],
    marker = dict( size=2, color='blue')
)
scatter3 = dict(
    mode = "markers",
    name = "Cluster 3",
    type = "scatter3d",    
    x = cluster3.as_matrix()[:,0], y = cluster3.as_matrix()[:,1], z = cluster3.as_matrix()[:,2],
    marker = dict( size=2, color='red')
)


# In[29]:


cluster1 = dict(
    alphahull = 5,
    name = "Cluster 1",
    opacity = .1,
    type = "mesh3d",    
    x = cluster1.as_matrix()[:,0], y = cluster1.as_matrix()[:,1], z = cluster1.as_matrix()[:,2],
    color='green', showscale = True
)
cluster2 = dict(
    alphahull = 5,
    name = "Cluster 2",
    opacity = .1,
    type = "mesh3d",    
    x = cluster2.as_matrix()[:,0], y = cluster2.as_matrix()[:,1], z = cluster2.as_matrix()[:,2],
    color='blue', showscale = True
)
cluster3 = dict(
    alphahull = 5,
    name = "Cluster 3",
    opacity = .1,
    type = "mesh3d",    
    x = cluster3.as_matrix()[:,0], y = cluster3.as_matrix()[:,1], z = cluster3.as_matrix()[:,2],
    color='red', showscale = True
)
layout = dict(
    title = 'Interactive Cluster Shapes in 3D',
    scene = dict(
        xaxis = dict( zeroline=True ),
        yaxis = dict( zeroline=True ),
        zaxis = dict( zeroline=True ),
    )
)
fig = dict( data=[scatter1, scatter2, scatter3, cluster1, cluster2, cluster3], layout=layout )
# Use py.iplot() for IPython notebook
plotly.offline.iplot(fig, filename='mesh3d_sample')

