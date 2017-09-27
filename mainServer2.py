from flask import Flask, render_template, request, redirect, flash, url_for, session
# from flaskext.mysql import MySQL
import csv, math, io

import json
#from werkzeug.utils import secure_filename

import os
import pickle as pickle
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import pandas as pd
from pprint import pprint

app = Flask(__name__)
app.secret_key = 'secret'

def read_ftr_data(ftr_path, month):
    if not ftr_path.endswith('/'):
        ftr_path += '/'

    daily_features = os.listdir(ftr_path + str(month))
    feature_dict = {}

    for i in range(len(daily_features)):
        date = os.path.splitext(daily_features[i])[0]

        with open(ftr_path + month + '/' + daily_features[i], 'rb') as f:
            data = pickle.load(f)

        feature_dict[date] = data

    return feature_dict


def feature_similar(date_to_be_compared):
    rank = 20
    ftr_path = 'feature'
    month = str(date_to_be_compared)[4:6]
    feature_dict = read_ftr_data(ftr_path, month)

    MSE_result = {}
    COS_result = {}
    for date in feature_dict.keys():
        if date != str(date_to_be_compared):
            # Mean sqaured error
            MSE_result[date] = np.mean((feature_dict[str(date_to_be_compared)] - feature_dict[date]) ** 2)
            # Cosine Similarity
            COS_result[date] = cosine(feature_dict[str(date_to_be_compared)].reshape(1, -1),
                                      feature_dict[date].reshape(1, -1))

    sorted_MSE = sorted(MSE_result.values())
    sorted_COS = sorted(COS_result.values(), reverse=True)

    # tabulize the result
    date_col = [date_to_be_compared] + ['' for i in range(rank - 1)]
    MSE_rank = [[date for date, value in MSE_result.items() if value == sorted_MSE[i]] for i in range(rank)]
    MSE_result = [sorted_MSE[i] for i in range(rank)]
    COS_rank = [[date for date, value in COS_result.items() if value == sorted_COS[i]] for i in range(rank)]
    COS_result = [sorted_COS[i] for i in range(rank)]

    with open('cloud_csv/' + str(10) + '.csv', newline='',
              errors='ignore') as clf:
        date_rows = list(csv.reader(clf))
        index_cloud = ['' for _ in range(rank)]
        MSE_cloud = []
        COS_cloud = []
        for MSE_date, COS_date in zip(MSE_rank, COS_rank):
            for i in range(len(date_rows)):
                if MSE_date[0] == date_rows[i][0]:
                    MSE_cloud.append(date_rows[i][1:])
                if COS_date[0] == date_rows[i][0]:
                    COS_cloud.append(date_rows[i][1:])

                if str(date_to_be_compared) == date_rows[i][0]:
                    index_cloud[0] = date_rows[i][1:]

    similarity_data = list(zip(index_cloud, MSE_rank, MSE_result, MSE_cloud, COS_rank, COS_result, COS_cloud))
    similarity_result = pd.DataFrame(data=similarity_data,
                                     columns=['cloud(18,00,06,12)', 'MSE_rank', 'MSE', 'cloud(18,00,06,12)', 'COS_rank',
                                              'COS', 'cloud(18,00,06,12)'],
                                     index=date_col)

    return similarity_result

@app.route("/")
def showMainPage():
	return render_template("mainPage.html")

@app.route("/table", methods = ['POST'])
def showTablePage():
    userdate = request.form['userdate']
    similarity_result = feature_similar(userdate[0:4] + userdate[5:7] + userdate[8:10])
    sim_date1 = similarity_result.iloc[0]['MSE_rank'].values[0]
    sim_rmse1 = similarity_result.iloc[0]['MSE']
    sim_date2 = similarity_result.iloc[1]['MSE_rank'].values[0]
    sim_rmse2 = similarity_result.iloc[1]['MSE']
    sim_date3 = similarity_result.iloc[2]['MSE_rank'].values[0]
    sim_rmse3 = similarity_result.iloc[2]['MSE']
    sim_date4 = similarity_result.iloc[3]['MSE_rank'].values[0]
    sim_rmse4 = similarity_result.iloc[3]['MSE']
    sim_date5 = similarity_result.iloc[4]['MSE_rank'].values[0]
    sim_rmse5 = similarity_result.iloc[4]['MSE']

    return render_template("tablePage.html", value = {'userdate':userdate,
                                                      'sim_date1':sim_date1, 'sim_rmse1': sim_rmse1,
                                                      'sim_date2':sim_date2, 'sim_rmse2': sim_rmse2,
                                                      'sim_date3':sim_date3, 'sim_rmse3': sim_rmse3,
                                                      'sim_date4':sim_date4, 'sim_rmse4': sim_rmse4,
                                                      'sim_date5':sim_date5, 'sim_rmse5': sim_rmse5})

@app.route("/image/<date>")
def showImagePage(date):
    return render_template("imagesPage.html", value = date)

# 메인 화면
#@app.route("/main")


if __name__=="__main__":
	app.run(debug=True, host = '127.0.0.1', port= 5000)
	#app.run(host='163.152.184.176', port= 5000, debug=True)