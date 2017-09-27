
import os
import pickle as pickle
import csv
import numpy as np 
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import pandas as pd
from pprint import pprint

def read_ftr_data(ftr_path, month):
    if not ftr_path.endswith('/'):
        ftr_path+='/'

    daily_features=os.listdir(ftr_path+str(month))
    feature_dict={}
    
    for i in range(len(daily_features)):
        date=os.path.splitext(daily_features[i])[0]
        
        with open(ftr_path+month+'/'+daily_features[i], 'rb') as f:
            data=pickle.load(f)
        
        feature_dict[date]=data
        
    return feature_dict


def feature_similar(date_to_be_compared):
    rank=20
    ftr_path='D:/Users/heewoong/Desktop/weather_server2/feature'
    month=str(date_to_be_compared)[4:6]
    feature_dict=read_ftr_data(ftr_path, month)

    MSE_result={}
    COS_result={}
    for date in feature_dict.keys():
        if date != str(date_to_be_compared):
            #Mean sqaured error
            MSE_result[date]=np.mean((feature_dict[str(date_to_be_compared)]-feature_dict[date])**2)
            #Cosine Similarity
            COS_result[date]=cosine(feature_dict[str(date_to_be_compared)].reshape(1,-1), feature_dict[date].reshape(1,-1))
            
    sorted_MSE=sorted(MSE_result.values())
    sorted_COS=sorted(COS_result.values(), reverse=True)
    
    #tabulize the result
    date_col=[date_to_be_compared]+['' for i in range(rank-1)]
    MSE_rank=[[date for date,value in MSE_result.items() if value == sorted_MSE[i]] for i in range(rank)]
    MSE_result=[sorted_MSE[i] for i in range(rank)]
    COS_rank=[[date for date,value in COS_result.items() if value == sorted_COS[i]] for i in range(rank)]
    COS_result=[sorted_COS[i] for i in range(rank)]
    
    with open('D:/Users/heewoong/Desktop/weather_server2/cloud_csv/'+str(10)+'.csv', newline='', errors='ignore') as clf:
        date_rows=list(csv.reader(clf))
        index_cloud=['' for _ in range(rank)]
        MSE_cloud=[]
        COS_cloud=[]
        for MSE_date, COS_date in zip(MSE_rank, COS_rank):
            for i in range(len(date_rows)):
                if MSE_date[0] == date_rows[i][0]:
                    MSE_cloud.append(date_rows[i][1:])
                if COS_date[0] == date_rows[i][0]:
                    COS_cloud.append(date_rows[i][1:])
                
                if str(date_to_be_compared) == date_rows[i][0]:
                    index_cloud[0]=date_rows[i][1:]
                
                    
    similarity_data=list(zip(index_cloud, MSE_rank, MSE_result, MSE_cloud, COS_rank, COS_result, COS_cloud))              
    similarity_result=pd.DataFrame(data=similarity_data,
                                   columns=['cloud(18,00,06,12)','MSE_rank','MSE','cloud(18,00,06,12)','COS_rank','COS','cloud(18,00,06,12)'],
                                   index=date_col)
    
    return similarity_result

#execute the fucntion

similarity_result = feature_similar(20161031)
a = similarity_result.iloc[0]['MSE_rank'].values[0]
b = '2016-10-31'
print(b[0:4]+b[5:7]+b[8:10])