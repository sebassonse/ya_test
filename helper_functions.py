import torch
from torch import nn
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd

import seaborn

##################
# Table creation

def create_table_model(cursor, connection):
  """Создает таблицy "model" SQL формата, если ее еще нет в базе данных"""
  try:
    cursor.execute(
        '''
        CREATE TABLE model (
          model_name VARCHAR(15),
          year VARCHAR(15),
          can_be_branded INT,
          car_cnt INT,
          car_sticker_cnt INT,
          trips_success_cnt INT,
          trips_cancel_cnt INT,
          trips_defect_cnt INT,
          trips_rated_cnt INT,
          bad_car_model_tag_cnt INT,
          dissapointed_car_model_tag_cnt INT
          )
        '''
    )
    model = pd.read_csv('/content/drive/MyDrive/Testovye/model.csv')
    model.to_sql('model', connection, if_exists='append', index = False)
    print('Table model created')

  except sqlite3.OperationalError:
    print('Table model already exists')
    
def create_table_partner(cursor, connection):
  """Создает таблицу "partner" SQL формата, если её ещё нет в базе данных"""

  try:
    cursor.execute(
        '''
        CREATE TABLE partner (
          partner_name VARCHAR(15),
          model_name VARCHAR(15),
          year VARCHAR(15),
          car_cnt INT,
          car_sticker_cnt INT
        )
        '''
    )
      
    partner = pd.read_csv('/content/drive/MyDrive/Testovye/partner.csv')
    partner.to_sql('partner', connection, if_exists='append', index = False)
    print('Table partner created')

  except sqlite3.OperationalError:
    print('Table partner already exists')

    
def create_table_model_n_partner(cursor, connection):
  try:
    cursor.execute(
        '''
        CREATE TABLE model_n_partner AS
        SELECT model.model_name,
               model.year,
               can_be_branded AS old_label,
               model.car_cnt AS model_car_cnt,
               model.car_sticker_cnt AS model_car_sticker_cnt,
               trips_success_cnt,
               trips_cancel_cnt,
               trips_defect_cnt,
               trips_rated_cnt,
               bad_car_model_tag_cnt,
               dissapointed_car_model_tag_cnt,

               (CASE
               WHEN trips_cancel_cnt > 0 THEN bad_car_model_tag_cnt*100.0/(trips_cancel_cnt)
               ELSE 0 END) AS bad_car_tag_percent,
               (CASE
               WHEN trips_defect_cnt > 0 THEN dissapointed_car_model_tag_cnt*100.0/trips_defect_cnt
               ELSE 0 END) AS diss_car_tag_percent,
               (CASE
               WHEN trips_success_cnt > 0 THEN trips_rated_cnt*100.0/trips_success_cnt
               ELSE 0 END) AS trips_rated_percent,
               (CASE
               WHEN (trips_rated_cnt - trips_defect_cnt) > 0 THEN (trips_defect_cnt + trips_cancel_cnt)*100.0/(trips_rated_cnt - trips_defect_cnt)
               ELSE 0 END) AS bad_trips_to_good_trips_percent,

               partner.car_cnt AS partner_car_cnt,
               partner.car_sticker_cnt AS partner_car_sticker_cnt
        FROM model INNER JOIN 
        (
          SELECT model_name, year, SUM(car_cnt) AS car_cnt, SUM(car_sticker_cnt) AS car_sticker_cnt
          FROM partner
          GROUP BY model_name, year
        ) AS partner ON model.model_name = partner.model_name AND model.year = partner.year
        WHERE trips_success_cnt > 100
        '''
    )
    print('Table model_n_partner created')

  except sqlite3.OperationalError:
    print('Table model_n_partner already exists')
    data = cursor.execute(
        '''SELECT * FROM model_n_partner'''
    ).fetchall()
    print(len(data))
    print(data)
    
def create_table_model_n_partner_prev(cursor, connection):
  try:
    cursor.execute(
        '''
        CREATE TABLE model_n_partner AS
        SELECT model.model_name,
               model.year,
               can_be_branded AS old_label,
               model.car_cnt AS model_car_cnt,
               model.car_sticker_cnt AS model_car_sticker_cnt,
               trips_success_cnt,
               trips_cancel_cnt,
               trips_defect_cnt,
               trips_rated_cnt,
               bad_car_model_tag_cnt,
               dissapointed_car_model_tag_cnt,

               bad_car_model_tag_cnt*100.0/(trips_success_cnt + trips_cancel_cnt) AS bad_car_tag_percent,
               dissapointed_car_model_tag_cnt*100.0/(trips_success_cnt + trips_cancel_cnt) AS diss_car_tag_percent,
               trips_rated_cnt*100.0/(trips_success_cnt + trips_cancel_cnt) AS trips_rated_percent,
               (dissapointed_car_model_tag_cnt + bad_car_model_tag_cnt)*100.0/(trips_rated_cnt - trips_defect_cnt) AS poor_car_to_good_trips_percent,

               partner.car_cnt AS partner_car_cnt,
               partner.car_sticker_cnt AS partner_car_sticker_cnt
        FROM model INNER JOIN 
        (
          SELECT model_name, year, SUM(car_cnt) AS car_cnt, SUM(car_sticker_cnt) AS car_sticker_cnt
          FROM partner
          GROUP BY model_name, year
        ) AS partner ON model.model_name = partner.model_name AND model.year = partner.year
        WHERE (trips_success_cnt + trips_cancel_cnt) > 100 AND
        (trips_rated_cnt - trips_defect_cnt) > 0
        '''
    )
    print('Table model_n_partner created')
  
  except sqlite3.OperationalError:
    print('Table model_n_partner already exists')
    data = cursor.execute(
        '''SELECT * FROM model_n_partner'''
    ).fetchall()
    print(len(data))
    print(data)

    
###########################
# Data preparation

def get_old_labels(cursor):
  rowlabels = cursor.execute(
      '''
      SELECT old_label
      FROM model_n_partner
      '''
  ).fetchall()

  labels = np.array([i[0] for i in rowlabels])
  return labels


def extract_training_data_prev(cursor):
  rows = cursor.execute(
      '''
      SELECT model_car_cnt,
             trips_success_cnt,
             trips_cancel_cnt,
             trips_defect_cnt,
             trips_rated_cnt,
             bad_car_model_tag_cnt,
             dissapointed_car_model_tag_cnt,
             bad_car_tag_percent,
             diss_car_tag_percent,
             trips_rated_percent,
             poor_car_to_good_trips_percent
      FROM model_n_partner
      '''
  ).fetchall()
  raw_data = list(map(list, rows))
  return raw_data

def extract_training_data(cursor):
  rows = cursor.execute(
      '''
      SELECT trips_success_cnt*1.0/model_car_cnt,
             trips_cancel_cnt*1.0/model_car_cnt,
             trips_defect_cnt*1.0/model_car_cnt,
             trips_rated_cnt/model_car_cnt,
             bad_car_model_tag_cnt/model_car_cnt,
             dissapointed_car_model_tag_cnt/model_car_cnt
      FROM model_n_partner
      '''
  ).fetchall()
  raw_data = list(map(list, rows))
  return raw_data


##################
# k-means model creation and training

def normalize_data(raw_data):
  normalized_data = nn.functional.normalize(torch.tensor(raw_data).type(torch.float32)).numpy()
  return normalized_data


def create_and_train_model(X):
  kmeans = KMeans(n_clusters=2, random_state=0, max_iter=500).fit(X)
  pretrained_labels = np.array(kmeans.labels_)

  return pretrained_labels if sum(pretrained_labels) > sum((pretrained_labels-1)*(-1)) else (pretrained_labels-1)*(-1)


###################
# Data for viz and analysis preparation

def get_data_to_analysis(cursor):
  rows = cursor.execute(
      '''
      SELECT rowid,
             trips_success_cnt*1.0/model_car_cnt,
             trips_cancel_cnt*1.0/model_car_cnt,
             trips_defect_cnt*1.0/model_car_cnt,
             trips_rated_cnt*1.0/model_car_cnt,
             bad_car_model_tag_cnt*1.0/model_car_cnt,
             dissapointed_car_model_tag_cnt*1.0/model_car_cnt,
             bad_car_tag_percent,
             diss_car_tag_percent,
             trips_rated_percent,
             bad_trips_to_good_trips_percent,
             model_car_cnt,
             model_car_sticker_cnt,
             partner_car_cnt,
             partner_car_sticker_cnt

      FROM model_n_partner
      '''
  ).fetchall()
  data = list(map(list, rows))
  return data

def get_data_to_analysis_prev(cursor):
  rows = cursor.execute(
      '''
      SELECT rowid,
             trips_success_cnt,
             trips_cancel_cnt,
             trips_defect_cnt,
             trips_rated_cnt,
             bad_car_model_tag_cnt,
             dissapointed_car_model_tag_cnt,
             bad_car_tag_percent,
             diss_car_tag_percent,
             trips_rated_percent,
             bad_trips_to_good_trips_percent,
             model_car_cnt,
             model_car_sticker_cnt,
             partner_car_cnt,
             partner_car_sticker_cnt

      FROM model_n_partner
      '''
  ).fetchall()
  data = list(map(list, rows))
  return data


####################
# Data visualisation and analysis

def visualize_data_with_labels(data, old_labels, new_labels):
  data = np.array(data)

  new_labels = np.array(new_labels).reshape(len(new_labels),1)
  old_labels = np.array(old_labels).reshape(len(old_labels),1)
  
  dataset = np.hstack((data, old_labels, new_labels))
  
  df = pd.DataFrame(np.around(dataset, 4),  columns= ['rowid',
                                                      'trips_success_cnt',
                                                      'trips_cancel_cnt',
                                                      'trips_defect_cnt',
                                                      'trips_rated_cnt',
                                                      'bad_car_model_tag_cnt',
                                                      'dissapointed_car_model_tag_cnt',
                                                      'bad_car_tag_percent',
                                                      'diss_car_tag_percent',
                                                      'trips_rated_percent',
                                                      'bad_trips_to_good_trips_percent',
                                                      'model_car_cnt',
                                                      'model_car_sticker_cnt',
                                                      'partner_car_cnt',
                                                      'partner_car_sticker_cnt',
                                                      'old_labels',
                                                      'new_labels'])
  

  print(sum(new_labels), sum(old_labels))

  g = seaborn.FacetGrid(df, col="old_labels")
  g.map(seaborn.kdeplot, "bad_car_tag_percent")
  plt.show()
  print(f'mean for "0": {np.mean(dataset[:,7]/100*(dataset[:,15]-1)*(-1)):.3f}, mean for "1": {np.mean(dataset[:,7]/100*dataset[:,15]):.3f}')

  g = seaborn.FacetGrid(df, col="new_labels")
  g.map(seaborn.kdeplot, "bad_car_tag_percent")
  plt.show()
  print(f'mean for "0": {np.mean(dataset[:,7]/100*(dataset[:,16]-1)*(-1)):.3f}, mean for "1": {np.mean(dataset[:,7]/100*dataset[:,16]):.3f}')


  g = seaborn.FacetGrid(df, col="old_labels")
  g.map(seaborn.kdeplot, 'diss_car_tag_percent')
  plt.show()
  print(f'mean for "0": {np.mean(dataset[:,8]/100*(dataset[:,15]-1)*(-1)):.3f}, mean for "1": {np.mean(dataset[:,8]/100*dataset[:,15]):.3f}')

  g = seaborn.FacetGrid(df, col="new_labels")
  g.map(seaborn.kdeplot, 'diss_car_tag_percent')
  plt.show()
  print(f'mean for "0": {np.mean(dataset[:,8]/100*(dataset[:,16]-1)*(-1)):.3f}, mean for "1": {np.mean(dataset[:,8]/100*dataset[:,16]):.3f}')


  g = seaborn.FacetGrid(df, col="old_labels")
  g.map(seaborn.kdeplot, "trips_rated_percent")
  plt.show()
  print(f'mean for "0": {np.mean(dataset[:,9]/100*(dataset[:,15]-1)*(-1)):.3f}, mean for "1": {np.mean(dataset[:,9]/100*dataset[:,15]):.3f}')

  g = seaborn.FacetGrid(df, col="new_labels")
  g.map(seaborn.kdeplot, "trips_rated_percent")
  plt.show()
  print(f'mean for "0": {np.mean(dataset[:,9]/100*(dataset[:,16]-1)*(-1)):.3f}, mean for "1": {np.mean(dataset[:,9]/100*dataset[:,16]):.3f}')


  g = seaborn.FacetGrid(df, col="old_labels")
  g.map(seaborn.kdeplot, "bad_trips_to_good_trips_percent")
  plt.show()
  print(f'mean for "0": {np.mean(dataset[:,10]*(dataset[:,15]-1)*(-1)):.3f}, mean for "1": {np.mean(dataset[:,10]*dataset[:,15]):.3f}')

  g = seaborn.FacetGrid(df, col="new_labels")
  g.map(seaborn.kdeplot, "bad_trips_to_good_trips_percent")
  plt.show()
  print(f'mean for "0": {np.mean(dataset[:,10]*(dataset[:,16]-1)*(-1)):.3f}, mean for "1": {np.mean(dataset[:,10]*dataset[:,16]):.3f}')


def analysis_of_data(data, old_labels, new_labels):
  data = np.array(data)

  new_labels = np.array(new_labels).reshape(len(new_labels),1)
  old_labels = np.array(old_labels).reshape(len(old_labels),1)
  
  dataset = np.hstack((data, old_labels, new_labels))
  
  df = pd.DataFrame(np.around(dataset, 4),  columns= ['rowid',
                                                      'trips_success_cnt_per_car',
                                                      'trips_cancel_cnt_per_car',
                                                      'trips_defect_cnt_per_car',
                                                      'trips_rated_cnt_per_car',
                                                      'bad_car_model_tag_cnt_per_car',
                                                      'dissapointed_car_model_tag_cnt_per_car',
                                                      'bad_car_tag_percent',
                                                      'diss_car_tag_percent',
                                                      'trips_rated_percent',
                                                      'bad_trips_to_good_trips_percent',
                                                      'model_car_cnt',
                                                      'model_car_sticker_cnt',
                                                      'partner_car_cnt',
                                                      'partner_car_sticker_cnt',
                                                      'old_labels',
                                                      'new_labels'])





  # Id модели-года авто, которые стоит брендировать по нынешнему классификатору

  # mean по количественным и процентным метрикам (можно прям на визуализации)

    # Сколько авто было оклеено
  was_branded = np.sum(dataset[:, 12])
  
    # Сколько авто могло быть оклеено
  could_have_been_branded_in_service = np.sum(dataset[:, 15]*dataset[:, 11])
  could_have_been_branded_in_partnership = np.sum(dataset[:, 15]*dataset[:, 13])

    # Сколько авто может быть оклеено теперь
  may_be_branded_in_service = np.sum(dataset[:, 16]*dataset[:, 11])
  may_be_branded_in_partnership = np.sum(dataset[:, 16]*dataset[:, 13])

    # Сколько авто, по новому классификатору, было брендировано зря
  need_to_be_debranded_in_service = np.sum(dataset[:, 12]*((dataset[:,16]-1)*(-1)))
  need_to_be_debranded_in_partnership = np.sum(dataset[:, 14]*((dataset[:,16]-1)*(-1)))


  print(f'was_branded: {was_branded}, \ncould_have_been_branded_in_service: {could_have_been_branded_in_service}, \ncould_have_been_branded_in_service: {could_have_been_branded_in_partnership},\n')
  print(f'may_be_branded_in_service: {may_be_branded_in_service}, \nmay_be_branded_in_partnership: {may_be_branded_in_partnership},\n')
  print(f'need_to_be_debranded_in_service: {need_to_be_debranded_in_service},\nneed_to_be_debranded_in_partnership: {need_to_be_debranded_in_partnership}')

  # Количество
  g = seaborn.FacetGrid(df, col="old_labels")
  g.map(seaborn.kdeplot, "trips_success_cnt_per_car")
  plt.show()
  print(f'mean for "0": {np.mean(dataset[:,1]*(dataset[:,15]-1)*(-1)):.3f}, mean for "1": {np.mean(dataset[:,1]*dataset[:,15]):.3f}')

  g = seaborn.FacetGrid(df, col="new_labels")
  g.map(seaborn.kdeplot, "trips_success_cnt_per_car")
  plt.show()
  print(f'mean for "0": {np.mean(dataset[:,1]*(dataset[:,16]-1)*(-1)):.3f}, mean for "1": {np.mean(dataset[:,1]*dataset[:,16]):.3f}')


  g = seaborn.FacetGrid(df, col="old_labels")
  g.map(seaborn.kdeplot, "dissapointed_car_model_tag_cnt_per_car")
  plt.show()
  print(f'mean for "0": {np.mean(dataset[:,6]*(dataset[:,15]-1)*(-1)):.3f}, mean for "1": {np.mean(dataset[:,6]*dataset[:,15]):.3f}')

  g = seaborn.FacetGrid(df, col="new_labels")
  g.map(seaborn.kdeplot, "dissapointed_car_model_tag_cnt_per_car")
  plt.show()
  print(f'mean for "0": {np.mean(dataset[:,6]*(dataset[:,16]-1)*(-1)):.3f}, mean for "1": {np.mean(dataset[:,6]*dataset[:,16]):.3f}')

  g = seaborn.FacetGrid(df, col="old_labels")
  g.map(seaborn.kdeplot, 'bad_car_model_tag_cnt_per_car')
  plt.show()
  print(f'mean for "0": {np.mean(dataset[:,5]*(dataset[:,15]-1)*(-1)):.3f}, mean for "1": {np.mean(dataset[:,5]*dataset[:,15]):.3f}')

  g = seaborn.FacetGrid(df, col="new_labels")
  g.map(seaborn.kdeplot, 'bad_car_model_tag_cnt_per_car')
  plt.show()
  print(f'mean for "0": {np.mean(dataset[:,5]*(dataset[:,16]-1)*(-1)):.3f}, mean for "1": {np.mean(dataset[:,5]*dataset[:,16]):.3f}')

  g = seaborn.FacetGrid(df, col="old_labels")
  g.map(seaborn.kdeplot, 'bad_car_model_tag_cnt_per_car', 'trips_cancel_cnt_per_car')
  g = seaborn.FacetGrid(df, col="new_labels")
  g.map(seaborn.kdeplot, "dissapointed_car_model_tag_cnt_per_car", "trips_defect_cnt_per_car")

  plt.show()
