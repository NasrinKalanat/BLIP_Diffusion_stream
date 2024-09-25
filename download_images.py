import os
import numpy as np
import pandas as pd
import glob
import xml.etree.ElementTree as ET
import datetime
from datetime import datetime, timedelta, timezone
from skimage.metrics import mean_squared_error, structural_similarity, normalized_root_mse
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MultiLabelBinarizer
import h5py
import requests
from PIL import Image
import torch
from  torchvision import transforms
import joblib
import json
import time

def download_image(url, file_path, file_name=""):
    # download from image url and import it as a numpy array
    full_path = os.path.join(file_path, file_name)
    res = requests.get(url, stream=True) # get full image
    if res.status_code == 200:
      img=res.raw
      img = Image.open(img)
      img = img.resize((360,360))
      # img = ImageOps.grayscale(img) # grayscale
      # img = img.tobytes() # convert to bytes
      # img = bytearray(img) # create byte array
      # img = np.asarray(img, dtype="uint8") # 360x360 array
      # img = img.reshape(360, 360, 3)
      # np.save(full_path,img)

      # with open(full_path, 'rb') as f:
      #   img = np.load(full_path)
      #   img=img.astype('float32') / 255
      # fig,ax = plt.subplots(1)
      # ax.imshow(img)
      img.save(full_path)

    else:
      print('Image Couldn\'t be retrieved '+ res.status_code)
    return img

def half_up_minute(x):
    delta = timedelta(minutes=15)
    ref_time = datetime(1970,1,1, tzinfo=x.tzinfo)
    return ref_time + round((x - ref_time) / delta) * delta

def load_data(path, path_weather, fol, phase="train"):
    imgfile = pd.read_csv(path + fol + '/images.csv',dtype={'station_name':str,'station_id':int,'image_id':int,'timestamp':str,'filename':str,'url':str}, parse_dates=['timestamp'])
    valuesfile = pd.read_csv(path + fol + '/values.csv',dtype={'station_name':str,'station_id':int,'dataset_id':int,'series_id':int,'variable_id':str,'timestamp':str,'value':float}, parse_dates=['timestamp'])
    stationfile = pd.read_csv(path + fol + '/station.csv')

    all_files = glob.glob(os.path.join(path_weather, 'Weather', '*.xlsx'))
    weatherfile=[]
    for w in all_files:
        file = pd.read_excel(os.path.join(path_weather,"Weather",w), skiprows=[0,1,3], parse_dates=["TIMESTAMP", "Time of Daily Temp Max", "Time of Min. Temp", "Time of Max Wind Spd", "Time of Min. Wind Spd."])
    #         file=pd.read_csv(os.path.join(root_data,"Weather",w), skiprows=[0,1,3], parse_dates=["TIMESTAMP", "Time of Daily Temp Max", "Time of Min. Temp", "Time of Max Wind Spd", "Time of Min. Wind Spd."])
    #         file.columns=file.iloc[1]
    #         file.drop([0,1,2], inplace=True)
    #         print("nan", len(file[file.isna().any(axis=1)]))
        file.dropna(inplace=True)
        file.reset_index(drop=True, inplace=True)
    #         file[["TIMESTAMP", "Time of Daily Temp Max", "Time of Min. Temp", "Time of Max Wind Spd", "Time of Min. Wind Spd."]]=file[["TIMESTAMP", "Time of Daily Temp Max", "Time of Min. Temp", "Time of Max Wind Spd", "Time of Min. Wind Spd."]].apply(pd.to_datetime)
        file[["Time of Daily Temp Max", "Time of Min. Temp", "Time of Max Wind Spd", "Time of Min. Wind Spd."]]=file[["Time of Daily Temp Max", "Time of Min. Temp", "Time of Max Wind Spd", "Time of Min. Wind Spd."]].applymap(datetime.timestamp)
        weatherfile.append(file)
    weatherfile=pd.concat(weatherfile, ignore_index=True)
    #       weatherfile.rename(columns={"TIMESTAMP": "timestamp"}, inplace=True)

    #       weatherfile[(weatherfile["Site_Name"]==fol.split("_")[0]) & (weatherfile["Station_No"]==fol.split("_")[1])]
    #       weatherfile=weatherfile[["DateTime_EST", "GageHeight_Hobo_ft", "Discharge_Hobo_cfs", "WaterTemperature_HOBO_DegF"]]
    #       weatherfile["DateTime_EST"]=pd.to_datetime(weatherfile["DateTime_EST"])

      # preprocessing time
    imgfile['timestamp'] = imgfile['timestamp'].map(half_up_minute)
    valuesfile['timestamp'] = valuesfile['timestamp'].map(half_up_minute)
    #       weatherfile["DateTime_EST"]=weatherfile["DateTime_EST"].map(self.half_up_minute)
    #       weatherfile[["TIMESTAMP", "Time of Daily Temp Max", "Time of Min. Temp", "Time of Max Wind Spd", "Time of Min. Wind Spd."]]=weatherfile[["TIMESTAMP", "Time of Daily Temp Max", "Time of Min. Temp", "Time of Max Wind Spd", "Time of Min. Wind Spd."]].applymap(self.half_up_minute)

    if phase=="pretrain":
        data=imgfile.copy()
        data['value']=valuesfile['value'].mean()
    else:
        data = imgfile.merge(valuesfile,on=["station_id","timestamp"])

    def custom_filter(x):
        day_rows=x[(x["timestamp"].dt.hour>9) & (x["timestamp"].dt.hour<18)]
        if not day_rows.empty:
            return day_rows.head(1)
        return x.head(1)
    #       data["date_tmp"]=data["timestamp"].dt.date
    data=data.groupby(data["timestamp"].dt.date, as_index=False).apply(custom_filter).reset_index(drop=True)

    data["date_tmp"]=data["timestamp"].dt.strftime('%Y-%m-%d')
      # print(data["date_tmp"].head(1))
    weatherfile["date_tmp"]=weatherfile["TIMESTAMP"].dt.strftime('%Y-%m-%d')
      # print(weatherfile["date_tmp"].head(1))
    weatherfile["TIMESTAMP"]=weatherfile["TIMESTAMP"].apply(datetime.timestamp)

    weatherfile=weatherfile.drop_duplicates(subset=['date_tmp']).reset_index(drop=True)
      # weatherlabelfile = pd.read_json(path + fol + '/response.jsonl', lines=True)
    l = []
    with open(path_weather+"response.jsonl", 'r') as weatherlabelfile:
        for line in weatherlabelfile.readlines():
            import ast
            t_jdata=json.loads(json.loads(line)[0]["messages"][1]["content"][51:])["Timestamp"]
            w_jdata=json.loads(json.loads(line)[1]["choices"][0]["message"]["content"])["Weather Classified Categories"]
            l.append([t_jdata, w_jdata])
    l_df=pd.DataFrame(data=l,columns=["date_tmp","weather_label"])
    l_df["date_tmp"]=pd.to_datetime(l_df["date_tmp"]).dt.strftime('%Y-%m-%d')
      # print(len(weatherfile),len(l_df))
    weatherfile = weatherfile.merge(l_df, on=["date_tmp"])
      # print(len(weatherfile), len(l_df))
      # weatherfile.drop("Timestamp",axis=1,inplace=True)
    #       exp_d=pd.date_range(pd.to_datetime("2018-01-01"),pd.to_datetime("2023-05-14"))
    #       print(list(set(exp_d)-set(weatherfile["date_tmp"])))
      # print("before", len(data))
    data = data.merge(weatherfile,on=["date_tmp"])
      # print("after", len(data))
    data.drop("date_tmp",axis=1,inplace=True)
    weatherfile.drop("date_tmp",axis=1,inplace=True)
    #       print(data.iloc[10:20])

    min_time=min(imgfile['timestamp'])
    max_time=max(imgfile['timestamp'])
    num_days=(max_time-min_time).days
    times=data['timestamp'].apply(datetime.timestamp)
    #       times=data['timestamp'].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return data
# data.url,data.value,times,data.weather_label

# download images (do just for first time)
images_path = "images"
imgs = load_data(path= "/data/nak168/spatial_temporal/stream_img/data/fpe-westbrook/", path_weather= "/data/nak168/spatial_temporal/stream_img/data/", fol= "Avery Brook_River Left_01171000")
if not os.path.exists(images_path):
  os.mkdir(images_path)
delay=0.001
for i, (id,url,lbl) in enumerate(zip(imgs.image_id,imgs.url,imgs.weather_label)):
  count=0
  while True:
      try:
        download_image(url, images_path, f"id_{id}_lbl_{lbl[0].replace('/', ' and ')}.png")
        # self.imgs.ix[i,"image_path"]=cur_path+fol+"/images/"+ str(id)+".jpg"
#         download_image2(url,cur_path+fol+"/images/"+ str(id)+".npy")
#         self.imgs.loc[:,"image_path"].iloc[i]=cur_path+fol+"/images/"+ str(id)+".npy"
        time.sleep(delay)
        break
      except Exception as e:
        # raise e
        print("error: ", e, "url: ",url)
        if count>5:
            break
        count+=1
        time.sleep(10*count)