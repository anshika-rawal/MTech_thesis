
Conversation opened. 2 messages. All messages read.

Skip to content
Using Gmail with screen readers
Search


finale.py 

Gmail
COMPOSE
Labels
Inbox (112)
Starred
Important
Chats
Sent Mail
Drafts (4)
All Mail
Spam (47)
Trash
Categories
UPSC
More labels 
Hangouts

 
 
Move to Inbox More 
1 of 2  
 
Collapse all Print all In new window
Project Code 
Inbox
x 

Pankaj Kharkwal <pkharkwal1994@gmail.com>
Attachments8/2/17

to Sarvesh, Heroine 

Attachments area

Sarvesh Sawant <ssarvesh93@gmail.com>
8/2/17

to me 
Heroine ... LOL

Regards,
Sarvesh Sawant
NIT Goa

On Wed, Aug 2, 2017 at 12:37 PM, Pankaj Kharkwal <pkharkwal1994@gmail.com> wrote:



	
Click here to Reply or Forward
5.2 GB (34%) of 15 GB used
Manage
Terms - Privacy
Last account activity: 1 hour ago
Details

# coding: utf-8

# In[1]:

import scipy as sp
import networkx as nx
import networkx as nx
import numpy as np
import openpyxl
import pandas as pd
from hidden_markov import hmm
import math
import time
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist


# In[3]:

#load dataset and read it from file
location = r'/home/anshika/Desktop/dataset/sms-call-internet-mi-2013-11-01.txt'
data = pd.read_csv(location, sep='\t',names = ["SqID", "Time","C_Cod", "SMSin", "SMSout","Callin", "Callout","Internet"])
#removing unneccessary information
del data['C_Cod']
del data['SMSin']
del data['SMSout']
del data['Callin']
del data['Callout']
#removing dataset with empty records
data=data.dropna()
data.reset_index(drop=True, inplace=True)


# In[4]:

#converting into matrix for fast calculationsb
data=data.sort_values("SqID",kind='mergesort')
newData = data.as_matrix()


# In[9]:

#aggregate the data set hour wise
i=0;
t=3600000.0
l=len(newData)
for j in range(l):
    if(newData[i][0]==newData[j][0] and i<=l):
        if((newData[j][1]-newData[i][1])<t):
            newData[i][2]+=newData[j][2]
            newData[j][2]=0
        else:
            i=j
    else:
        i=j
            
        
end = time.time()


# In[10]:

data=pd.DataFrame(newData )
data.columns = ["SqID", "Time","Internet"]
#arranging the time and removing the unnecessary data
data.loc[:,"Time"]+=3600000
data=data[data['Internet'] != 0.0]
data.reset_index(drop=True, inplace=True)


# In[11]:

#distributing the data into different time interval and summing the data for Vi
start = time.time()
newData = data.as_matrix()
i=0;
t= 21600000
l=len(newData)
night = []
morning = []
evening = []
noon = []
idMatrix = []
nightSum = {}
morningSum = {}
noonSum = {}
eveningSum = {}
n=0
m=0
no=0
e=0
l=len(newData)
for j in range(l):
    if(newData[i][0]==newData[j][0] and i<=l ):
        if(0<=(newData[j][1]-newData[i][1]) and (newData[j][1]-newData[i][1])<t):
            night.append(newData[j])
            n+=newData[j][2]
        elif(t<=(newData[j][1]-newData[i][1]) and (newData[j][1]-newData[i][1])<2*t):
            morning.append(newData[j])
            m+=newData[j][2]
        elif(2*t<=(newData[j][1]-newData[i][1]) and (newData[j][1]-newData[i][1])<3*t):
            noon.append(newData[j])
            no+=newData[j][2]
        elif(3*t<=(newData[j][1]-newData[i][1]) and (newData[j][1]-newData[i][1])<4*t):
            evening.append(newData[j])
            e+=newData[j][2]
        if(j==l-1):
            idMatrix.append(newData[i][0])
            nightSum[newData[i][0]]=n
            morningSum[newData[i][0]]=m
            noonSum[newData[i][0]]=no
            eveningSum[newData[i][0]]=e
    else:
        night.append(newData[j])
        n+=newData[j][2]  
        idMatrix.append(newData[i][0])
        nightSum[newData[i][0]]=n
        morningSum[newData[i][0]]=m
        noonSum[newData[i][0]]=no
        eveningSum[newData[i][0]]=e
        i=j
        m=0
end = time.time()
print(end-start)


# In[12]:

print(len(morning))


# In[13]:

start = time.time()
n=int(len(morning)/6)
SIM_MAT=[]
for i in range (0,n):
    row=[]
    for j in range (0,n):
        if i==j:
            row.append(1)
        else:
            row.append(0)
    SIM_MAT.append(row)
end = time.time()
print(end-start)
#Traffic distribution similarity
start = time.time()
for i in range(0,n-1):
    row=[]
    for j in range(i+1,n):
        sum=0
        for k in range(0,6):
            s=(morning[i*6+k][2]/morningSum[morning[i*6+k][0]])-(morning[j*6+k][2]/morningSum[morning[j*6+k][0]])
            sum+=s*s
        if(sum!=0):
            SIM_MAT[i][j]=SIM_MAT[j][i]=1/math.sqrt(sum)
        else:
            SIM_MAT[i][j]=SIM_MAT[j][i]=1
end = time.time()
print(end-start)


# In[122]:

#Traffic distribution similarity
def sim_mat(timeofday,sumofall):
    n=len(timeofday)/6
    mat = [[0 for x in range(len(timeofday)/6)] for y in range(len(timeofday)/6)]   
    for i in range(0,n-1):
        row=[]
        for j in range(i+1,n):
            sum=0
            for k in range(0,6):
                s=(timeofday[i*6+k][2]/sumofall[timeofday[i*6+k][0]])-(timeofday[j*6+k][2]/sumofall[timeofday[j*6+k][0]])
                sum+=s*s
            if(sum!=0):
                mat[i][j]=mat[j][i]=1/math.sqrt(sum)
            else:
                mat[i][j]=mat[j][i]=1
    return mat


# In[123]:

start = time.time()
morn_sim = sim_mat(morning,morningSum)
noon_sim = sim_mat(noon,noonSum)
eve_sim = sim_mat(evening, eveningSum)
night_sim = sim_mat(night, nightSum)
end = time.time()
print(end-start)


# In[124]:

def graphplot(matrix):
    dt = (('weight',float))
    All = np.matrix(matrix)
    G = nx.from_numpy_matrix(All)
    nx.draw(G, data=True, node_shape='8',width=0.2)
    G=nx.Graph()
    edge_labels=nx.draw_networkx_edge_labels(G,pos=nx.spring_layout(G))
    plt.show()
    Z=linkage(All, 'ward')
    #distance calculated
    c , coph_dists = cophenet(Z, pdist(All))
    plt.figure(figsize=(5,5))
    plt.title('Hierarchical Clustering Dendogram')
    plt.xlabel('location')
    plt.ylabel('traffic similarity')
    dendrogram(
        Z,
        leaf_rotation=90.,
        leaf_font_size=8.)
    plt.show()


# In[125]:

graphplot(morn_sim)


# In[126]:

graphplot(noon_sim)


# In[127]:

graphplot(eve_sim)


# In[128]:

graphplot(night_sim)


# In[130]:

#calculate euclidian distance 
def euclidistance(c1, c2, mat):
    """ Calculate the distance between two cluster """
    dist = .0
    n1 = len(c1.points)
    n2 = len(c2.points)
    for i in xrange(n1):
        for j in xrange(n2):
            p1 = c1.points[i]
            p2 = c2.points[j]
            d = 0
            #print p1,p2
            d = d + (mat[int(p1-1)][int(p2-1)])
            dist = dist + d
    return dist


# In[131]:

#calculate UPGMA
class Node:
    def __init__(self, p):
        self.points = p
        self.right = None
        self.left = None
def upgma(idMat,simMat, k):
    """ Cluster based on distance matrix dist using Unweighted Pair Group Method with Arithmetic Mean algorithm up to k cluster"""
 
    # Initialize each cluster with one point
    nodes = []
    n = len(idMat)
    for i in xrange(n):
        node = Node([idMat[i]])
        nodes = nodes + [node]
 
    # Iterate until the number of clusters is k
    nc = n
    while nc > k:
        # Calculate the pairwise distance of each cluster, while searching for pair with least distance
        c1 = 0; c2 = 0; i1 = 0; i2 = 0;
        sdis = 9999999999
        for i in xrange(nc):
            for j in xrange(i+1, nc):
                dis = euclidistance(nodes[i], nodes[j],simMat)
                if dis < sdis:
                    sdis = dis
                    c1 = nodes[i]; c2 = nodes[j];
                    i1 = i; i2 = j;
        # Merge these two nodes into one new node
        node = Node(c1.points + c2.points)
        node.left = c1; node.right = c2;
         
        #Remove the previous nodes, and add the new node
        new_nodes = []
        for i in xrange(nc):
            if i != i1 and i != i2:
                new_nodes = new_nodes + [nodes[i]]
        new_nodes = new_nodes + [node]
        nodes = new_nodes[:]
        nc = nc - 1
 
    return nodes


# In[132]:

# Cluster the data
#nodes = upgma(idMatrix,SIM_MAT, 3)
k=5
morn_clstr = upgma(idMatrix,morn_sim,k)
noon_clstr = upgma(idMatrix,noon_sim,k)
eve_clstr = upgma(idMatrix,eve_sim,k)
night_clstr = upgma(idMatrix,night_sim,k)


# In[133]:

#get Points of the cluster
def getPoints(cluster,pointsList):
    n1 = len(cluster.points)
    for i in xrange(n1):
        p1 = cluster.points[i]
        pointsList.append(p1)
    return pointsList


# In[134]:

#find cluster with max traffic
def max_cluster(cluster , sumofall):
    k=len(cluster)
    totalSum={}
    ms=0
    index=0
    for i in range(0,k):
        l=[]
        l=getPoints(cluster[i],l)
        v=len(l)
        s=0;
        for j in range(0,v):
            s+=sumofall[l[j]]
        if(s>ms):
            ms=s
            index=i
            totalSum[index]=ms
    point= []
    point = getPoints(cluster[index], point)
    return index,point,totalSum[index]


# In[135]:

morn_max,morn_max_ptns,totalMorning=max_cluster(morn_clstr,morningSum)
eve_max,eve_max_ptns, totalEvening=max_cluster(eve_clstr,eveningSum)
noon_max,noon_max_ptns,totalNoon = max_cluster(noon_clstr,noonSum)
night_max,night_max_ptns,totalNight = max_cluster(night_clstr,nightSum)
visibleNode=list(set(morn_max_ptns)|set(eve_max_ptns)|set(noon_max_ptns)|set(night_max_ptns))


# In[136]:

def findpoint(point, clusterpoints):
    for i in range(len(clusterpoints)):
        if(point==clusterpoints[i]):
            return 1
    return 0


# In[137]:

obs = visibleNode
states = ('Morning', 'Noon','Evening','Night')
start_p = [0.25, 0.25,0.25,0.25]
trans_p = [[0.3,0.7,0,0],[0,0.3,0.7,0],[0,0,0.3,0.7],[0.7,0,0,0.3]]
md = [0 for x in range(len(obs))]
nd=[0 for x in range(len(obs))]
nod=[0 for x in range(len(obs))]
ed=[ 0 for x in range(len(obs))]
for i in range(len(obs)):
    h=int(obs[i])
    if(findpoint(h,morn_max_ptns)):
        md[i]=(morningSum[h]/totalMorning)
    if(findpoint(h,night_max_ptns)):
        nd[i]=nightSum[h]/totalNight
    if(findpoint(h,noon_max_ptns)):
        nod[i]=noonSum[h]/totalNoon
    if(findpoint(h,eve_max_ptns)):
        ed[i]=eveningSum[h]/totalEvening
emit_p=[0 for x in range(len(states))]
emit_p[0]=md
emit_p[1]=nod
emit_p[2]=ed
emit_p[3]=nd
emit_p=np.asmatrix(emit_p)
trans_p=np.asmatrix(trans_p)
start_p=np.asmatrix(start_p)


# In[138]:

#create an HMM Class
t=hmm(states, obs, start_p, trans_p, emit_p)


# In[139]:

#calculating alpha
def calc_alpha(a,b):
    x1 = math.floor(a/100)
    x2 = math.floor(b/100)
    y1 = (a%100)
    y2 = (b%100)
    d = math.sqrt((x1-x2)*(x1-x2) + (y1-y2) * (y1-y2))

    if(d==0):
        return 1
    else:
        return 1/math.floor(d)


# In[140]:

#forward algorithm
l=len(obs)
listLoc = [[0 for x in range(0,3)] for y in range(len(obs))]
for i in range(len(obs)):
    ob1=[]
    presetLoc = obs[0]
    ob1.append(presetLoc)
    ob1.append(obs[i])
    listLoc[i][0]=presetLoc
    listLoc[i][1]=obs[i]
    listLoc[i][2]=t.forward_algo(ob1)        

# In[140]:

#forward algorithm
l=len(obs)
listLocD = {}
for i in range(len(obs)):
    ob1=[]
    for j in range(len(obs)):
        presentLoc = obs[i]
        ob1.append(presentLoc)
        ob1.append(obs[j])
        listLocD[int(presentLoc)][int(obs[j])]=t.forward_algo(ob1) 
newList = [0 for x in range (len(obs))]
for i in range(len(obs)):
	for j in range(len(obs)):
		k=(listLocD[obs[i]][obs[j]]+listLocD[obs[j]][obs[i]])/2
		newList[i].append(obs[j])
		newList[i].append(k)
max_loc1
max_loc2
max_value=0
for i in range(len(newList)):
	if(newList[i][1]>max_value):
		max_value = newList[i][1]
		max_loc2 = newList[i][0]
		max_loc1  = obs[i]

print(max_loc1, max_loc2, max_value)
	
# In[141]:

listLoc = sorted(listLoc, key=lambda x:x[2], reverse = True)
listLoc


# In[142]:

#apha multiplication to the probabitlities
listLocNew = listLoc
for i in range(len(listLocNew)):
    alpha = calc_alpha(listLocNew[i][0], listLocNew[i][1])
    listLocNew[i][2] *= alpha


# In[143]:

listLocNew = sorted(listLocNew, key=lambda x:x[2], reverse = True)
listLocNew


# In[144]:

#viterbi algorithms
l=len(obs)
listLoc = [[0 for x in range(0,3)] for y in range(len(obs))]
for i in range(len(obs)):
    ob1=[]
    ob2 = []
    presetLoc = obs[0]
    ob1.append(presetLoc)
    ob1.append(obs[i])
    listLoc[i][0]=presetLoc
    listLoc[i][1]=obs[i]
    listLoc[i][2]=t.viterbi(ob1)


# In[145]:

result=pd.DataFrame(listLoc)
result.columns = ["Present Location", "Predicted Location","Transition"]


# In[146]:

result.reset_index(drop=True, inplace=True)


# In[ ]:

result
