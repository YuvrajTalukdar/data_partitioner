from pickle import FALSE, TRUE
import sys
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import random

colors =[]#{0:'red', 1:'blue',2:'orange',3:'green',4:'pink',5:'black',6:'yellow',7:'violet',8:'purple',9:'chocolate',10:'cyan',11:'brown',12:'indigo',13:'lime',14:'grey',15:'chocolate'}

file_name=sys.argv[1]
print(file_name)

def get_color():
    hexadecimal = '#'+''.join([random.choice('abcdef0123456789') for i in range(6)])
    return hexadecimal

def read_data(file_name):
    df = pd.read_csv(file_name)
    rows = df.values.tolist()

    #color_list=[]
    data_list=[]
    x=[]
    y=[]
    label_list=[]
    for row in rows:
        #color_list.append(colors[row[(len(row)-1)]])
        label_exist=False
        for label in label_list:
            if label==row[(len(row)-1)]:
                label_exist=True
                break
        
        if label_exist==False:
            label_list.append(row[(len(row)-1)])

        attributes=[]
        for element in row:
            attributes.append((round(element,3)))

        data_list.append(attributes)
        x.append(row[0])
        y.append(row[1])
        #print(attributes)

    #plt.scatter(x,y,color=color_list)
    #plt.show()

    data_group_list=[]
    for label in label_list:
        data_group=[]
        a=len(data_list)-1
        while a>=0:
            if data_list[a][len(data_list[a])-1]==label:
                data_list[a].pop()
                data_group.append(data_list[a])
                data_list.pop(a)
            a-=1
        data_group_list.append(data_group)

    return data_group_list,label_list

def partition_data(data_group):
    #finding the no of clusters required
    wcss=[]
    for i in range(1,10):
        kmeans = KMeans(i)
        kmeans.fit(data_group)
        wcss_iter = kmeans.inertia_
        wcss.append(wcss_iter)
    number_clusters = range(1,10)
    '''
    plt.plot(number_clusters,wcss)
    plt.title('The Elbow title')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    '''
    a=1
    prev_wcss=wcss[0]
    no_of_clusters_required=1
    while a<len(wcss):
        #print(wcss[a])
        if prev_wcss-wcss[a]<20:
            no_of_clusters_required=number_clusters[a]
            break
        prev_wcss=wcss[a]
        a+=1
    if no_of_clusters_required==1:
        no_of_clusters_required=9
    a=0
    while a<no_of_clusters_required:
        colors.append((get_color()))
        a+=1
    print("no_of_clusters="+str(no_of_clusters_required))
    kmeans=KMeans(no_of_clusters_required)
    kmeans.fit(data_group)
    identified_clusters=kmeans.fit_predict(data_group)
    #plt.show()

    return identified_clusters,no_of_clusters_required

data_group_list,label_list=read_data(file_name)
labels_to_partition=[1,2]
file1=open("partitioned_"+file_name,"w")
c=0
color_add_by=0
no_of_clusters=0
identified_clusters=[]
for data_group in data_group_list:
    label_found=FALSE
    for label in labels_to_partition:
        if label==label_list[c]:
            label_found=TRUE
            break
    print("len= "+str(len(data_group)))
    if label_found==TRUE:
        identified_clusters,no_of_clusters=partition_data(data_group)
    else:
        no_of_clusters=1
    b=0
    for row in data_group:
        line=""
        for element in row:
            line+=(str(element)+",")
        if label_found==TRUE:
            line+=(str(identified_clusters[b]+color_add_by+1)+"\n")
        else:
            line+=(str(0)+"\n")
        b+=1
        #print(line)
        file1.write(line)
    color_add_by+=no_of_clusters
    c+=1
file1.close()
