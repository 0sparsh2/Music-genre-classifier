import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import uniform, randint
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
from sklearn.linear_model import LogisticRegression
import sklearn.ensemble as ske
import catboost as cb
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier as knn
import pickle
from pprint import pprint
import random
import librosa, IPython
import librosa.display as lplt
import eli5
from eli5.sklearn import PermutationImportance
seed = 12
np.random.seed(seed)

#Load dataset
df = pd.read_csv('mpr/Data/features_3_sec.csv')
df = df.drop(['harmony_mean','harmony_var'], axis = 1)
df.label.value_counts().reset_index()

label_index = dict()
index_label = dict()
for i, x in enumerate(df.label.unique()):
    label_index[x] = i
    index_label[i] = x

df_shuffle = df.sample(frac=1, random_state=seed).reset_index(drop=True)

# remove irrelevant columns
df_shuffle.drop(['filename', 'length'], axis=1, inplace=True)
df_y = df_shuffle.pop('label')
df_X = df_shuffle

# split into train dev and test
X_train, df_test_valid_X, y_train, df_test_valid_y = skms.train_test_split(df_X, df_y, train_size=0.7, random_state=seed, stratify=df_y)
X_dev, X_test, y_dev, y_test = skms.train_test_split(df_test_valid_X, df_test_valid_y, train_size=0.66, random_state=seed, stratify=df_test_valid_y)


#Scale the features
scaler = skp.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_dev = pd.DataFrame(scaler.transform(X_dev), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

pickle.dump(scaler, open('scalar.pkl','wb'))
pickle.dump(X_train, open('xtrain.pkl','wb'))

lr = LogisticRegression(random_state=seed)
lr.fit(X_train,y_train)

# Permutation Importance Feature Selection
perm = PermutationImportance(lr, random_state=seed).fit(X_train, y_train, n_iter=10)

perm_indices = np.argsort(perm.feature_importances_)[::-1]
perm_features = [X_dev.columns.tolist()[xx] for xx in perm_indices]
print(perm_features[:30])

# Model Scoring using Permutation Importances
X_train_perm = X_train[perm_features[:30]]
X_train_rfe = X_train_perm

#Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_rfe,y_train)
pickle.dump(lr, open('lr.pkl','wb'))

#Random Forest
rfc = ske.RandomForestClassifier(random_state=seed, n_jobs=-1)
rfc.fit(X_train_rfe, y_train)
pickle.dump(rfc, open('rfc.pkl','wb'))


#AdaBoost
abc = ske.AdaBoostClassifier(n_estimators=150, random_state=seed)
abc.fit(X_train_rfe, y_train)
pickle.dump(abc, open('abc.pkl','wb'))


#Gradient Boosting
gbc = ske.GradientBoostingClassifier(n_estimators=100, random_state=seed)
gbc.fit(X_train_rfe, y_train)
pickle.dump(gbc, open('gbc.pkl','wb'))


#XGBoost
xgbc = xgb.XGBClassifier(n_estimators=100, random_state=seed)
xgbc.fit(X_train_rfe, y_train)
pickle.dump(xgbc, open('xgbc.pkl','wb'))


#CatBoost
cbc = cb.CatBoostClassifier(random_state=seed, verbose=0, eval_metric='Accuracy', loss_function='MultiClass')
cbc.fit(X_train_rfe, y_train)
pickle.dump(cbc, open('cbc.pkl','wb'))


#KNN
cls = knn() #random_state=seed)
cls.fit(X_train_rfe, y_train)
pickle.dump(cls, open('cls.pkl','wb'))




'''
#Loading Testing Data

audio_fp = 'mpr/Data/genres_original/jazz/jazz.00001.wav'
audio_data, sr = librosa.load(audio_fp, offset=0, duration=3)
audio_data, _ = librosa.effects.trim(audio_data)


d = librosa.feature.mfcc(np.array(audio_data).flatten(),sr=22050 , n_mfcc = 20) #36565
d_var = d.var(axis=1).tolist()
d_mean = d.mean(axis=1).tolist()
test_data = []#[d_var + d_mean]
for i in range(20):
  test_data.append(d_mean[i])
  test_data.append(d_var[i])
mfcc_names=[]
for i in range(1,21):
  mfcc_str = "mfcc"+str(i)+"_mean"
  mfcc_names.append(mfcc_str)
  mfcc_str = "mfcc"+str(i)+"_var"
  mfcc_names.append(mfcc_str)
test_frame = pd.DataFrame([test_data], columns = mfcc_names)
test_data = []
mfcc_names=[]
#chroma
S = np.abs(librosa.stft(audio_data, n_fft=4096))**2
chroma = librosa.feature.chroma_stft(S=S, sr=sr)
#chroma_stft_mean
chroma_mean = round(np.mean(chroma),6)
test_data.append(chroma_mean)
#chrome_stft_var
chroma_var = round(np.var(chroma),6)
test_data.append(chroma_var)
#chroma_label
mfcc_names.append("chroma_stft_mean")
mfcc_names.append("chroma_stft_var")

#rms
rms = librosa.feature.rms(y=audio_data)
#rms_mean
rms_mean = round(np.mean(rms),6)
test_data.append(rms_mean)
#rms_var
rms_var = round(np.var(rms),6)
test_data.append(rms_var)
#rms_label
mfcc_names.append("rms_mean")
mfcc_names.append("rms_var")

#spectral_centroid
cent = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
#spectral_centroid_mean
sc_mean = round(np.mean(cent),6)
test_data.append(sc_mean)
#spectral_centroid_var
sc_var = round(np.var(cent),6)
test_data.append(sc_var)
#sc_label
mfcc_names.append("spectral_centroid_mean")
mfcc_names.append("spectral_centroid_var")

#spectral_bandwidth
spec_bw = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
#spectral_bandwidth_mean
spec_bw_mean = round(np.mean(spec_bw),6)
test_data.append(spec_bw_mean)
#spectral_bandwidth_var
spec_bw_var = round(np.var(spec_bw),6)
test_data.append(spec_bw_var)
#sb_label
mfcc_names.append("spectral_bandwidth_mean")
mfcc_names.append("spectral_bandwidth_var")

#rolloff
rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
#rolloff_mean
rolloff_mean = round(np.mean(rolloff),6)
test_data.append(rolloff_mean)
#rolloff_var
rolloff_var = round(np.var(rolloff),6)
test_data.append(rolloff_var)
#rolloff_label
mfcc_names.append("rolloff_mean")
mfcc_names.append("rolloff_var")

#zero_crossing_rate
zcr = librosa.feature.zero_crossing_rate(audio_data)
#zero_crossing_rate_mean
zcr_mean = round(np.mean(zcr),6)
test_data.append(zcr_mean)
#zero_crossing_rate_var
zcr_var = round(np.var(zcr),6)
test_data.append(zcr_var)
#zero_crossing_rate_label
mfcc_names.append("zero_crossing_rate_mean")
mfcc_names.append("zero_crossing_rate_var")

#harmony
y = librosa.effects.harmonic(audio_data)
harmony = librosa.feature.tonnetz(y=y, sr=sr)
#harmony_mean
harmony_mean = round(np.mean(harmony),6)
test_data.append(harmony_mean) 
#harmony_var
harmony_var = round(np.var(harmony),6)
test_data.append(harmony_var)
#harmony_label
mfcc_names.append("harmony_mean")
mfcc_names.append("harmony_var")

#perceptr_mean
#perceptr_var


#tempo
hop_length = 512
oenv = librosa.onset.onset_strength(y=audio_data, sr=sr, hop_length=hop_length)
tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
                          hop_length=hop_length)[0]

tempo = round(tempo,6)
test_data.append(tempo)
#tempo_label
mfcc_names.append("tempo")
d_var = d.var(axis=1).tolist()
d_mean = d.mean(axis=1).tolist()
#test_data = []#[d_var + d_mean]
for i in range(20):
  test_data.append(d_mean[i])
  test_data.append(d_var[i])
for i in range(1,21):
  mfcc_str = "mfcc"+str(i)+"_mean"
  mfcc_names.append(mfcc_str)
  mfcc_str = "mfcc"+str(i)+"_var"
  mfcc_names.append(mfcc_str)


test_frame = pd.DataFrame([test_data], columns = mfcc_names)
testing_frame = pd.DataFrame(scaler.transform(test_frame), columns=X_train.columns)
shorter_testing_frame = testing_frame[perm_features[:30]]


val=3
while(val<=27):
  audio_fp = 'mpr/Data/genres_original/jazz/jazz.00001.wav'
  audio_data, sr = librosa.load(audio_fp, offset=val, duration=val+3)
  audio_data, _ = librosa.effects.trim(audio_data)
  d = librosa.feature.mfcc(np.array(audio_data).flatten(),sr=22050 , n_mfcc = 20) #36565
  d_var = d.var(axis=1).tolist()
  d_mean = d.mean(axis=1).tolist()
  test_data = []#[d_var + d_mean]
  for i in range(20):
    test_data.append(d_mean[i])
    test_data.append(d_var[i])
  mfcc_names=[]
  for i in range(1,21):
    mfcc_str = "mfcc"+str(i)+"_mean"
    mfcc_names.append(mfcc_str)
    mfcc_str = "mfcc"+str(i)+"_var"
    mfcc_names.append(mfcc_str)
  test_frame = pd.DataFrame([test_data], columns = mfcc_names)
  test_data = []
  mfcc_names=[]
  #chroma
  S = np.abs(librosa.stft(audio_data, n_fft=4096))**2
  chroma = librosa.feature.chroma_stft(S=S, sr=sr)
  #chroma_stft_mean
  chroma_mean = round(np.mean(chroma),6)
  test_data.append(chroma_mean)
  #chrome_stft_var
  chroma_var = round(np.var(chroma),6)
  test_data.append(chroma_var)
  #chroma_label
  mfcc_names.append("chroma_stft_mean")
  mfcc_names.append("chroma_stft_var")

  #rms
  rms = librosa.feature.rms(y=audio_data)
  #rms_mean
  rms_mean = round(np.mean(rms),6)
  test_data.append(rms_mean)
  #rms_var
  rms_var = round(np.var(rms),6)
  test_data.append(rms_var)
  #rms_label
  mfcc_names.append("rms_mean")
  mfcc_names.append("rms_var")

  #spectral_centroid
  cent = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
  #spectral_centroid_mean
  sc_mean = round(np.mean(cent),6)
  test_data.append(sc_mean)
  #spectral_centroid_var
  sc_var = round(np.var(cent),6)
  test_data.append(sc_var)
  #sc_label
  mfcc_names.append("spectral_centroid_mean")
  mfcc_names.append("spectral_centroid_var")

  #spectral_bandwidth
  spec_bw = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
  #spectral_bandwidth_mean
  spec_bw_mean = round(np.mean(spec_bw),6)
  test_data.append(spec_bw_mean)
  #spectral_bandwidth_var
  spec_bw_var = round(np.var(spec_bw),6)
  test_data.append(spec_bw_var)
  #sb_label
  mfcc_names.append("spectral_bandwidth_mean")
  mfcc_names.append("spectral_bandwidth_var")

  #rolloff
  rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
  #rolloff_mean
  rolloff_mean = round(np.mean(rolloff),6)
  test_data.append(rolloff_mean)
  #rolloff_var
  rolloff_var = round(np.var(rolloff),6)
  test_data.append(rolloff_var)
  #rolloff_label
  mfcc_names.append("rolloff_mean")
  mfcc_names.append("rolloff_var")

  #zero_crossing_rate
  zcr = librosa.feature.zero_crossing_rate(audio_data)
  #zero_crossing_rate_mean
  zcr_mean = round(np.mean(zcr),6)
  test_data.append(zcr_mean)
  #zero_crossing_rate_var
  zcr_var = round(np.var(zcr),6)
  test_data.append(zcr_var)
  #zero_crossing_rate_label
  mfcc_names.append("zero_crossing_rate_mean")
  mfcc_names.append("zero_crossing_rate_var")

  #harmony
  y = librosa.effects.harmonic(audio_data)
  harmony = librosa.feature.tonnetz(y=y, sr=sr)
  #harmony_mean
  harmony_mean = round(np.mean(harmony),6)
  test_data.append(harmony_mean) 
  #harmony_var
  harmony_var = round(np.var(harmony),6)
  test_data.append(harmony_var)
  #harmony_label
  mfcc_names.append("harmony_mean")
  mfcc_names.append("harmony_var")

  #perceptr_mean
  #perceptr_var


  #tempo
  hop_length = 512
  oenv = librosa.onset.onset_strength(y=audio_data, sr=sr, hop_length=hop_length)
  tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
                            hop_length=hop_length)[0]

  tempo = round(tempo,6)
  test_data.append(tempo)
  #tempo_label
  mfcc_names.append("tempo")
  d_var = d.var(axis=1).tolist()
  d_mean = d.mean(axis=1).tolist()
  #test_data = []#[d_var + d_mean]
  for i in range(20):
    test_data.append(d_mean[i])
    test_data.append(d_var[i])
  for i in range(1,21):
    mfcc_str = "mfcc"+str(i)+"_mean"
    mfcc_names.append(mfcc_str)
    mfcc_str = "mfcc"+str(i)+"_var"
    mfcc_names.append(mfcc_str)


  test_frame2 = pd.DataFrame([test_data], columns = mfcc_names)
  testing_frame2 = pd.DataFrame(scaler.transform(test_frame2), columns=X_train.columns)
  shorter_testing_frame2 = testing_frame2[perm_features[:30]]
  df_test = pd.concat([shorter_testing_frame, shorter_testing_frame2])
  shorter_testing_frame = df_test
  val+=3



#Testing Input Data
from collections import Counter
result_list=[]
models = {'Catboost':cbc, 'XGBoost':xgbc, 'Gradient Boosting':gbc, 'AdaBoost':abc, 'Random Forest':rfc, 'Linear Regression':lr, 'KNN':cls}
key_list = list(models.keys())
val_list = list(models.values())
 



for model in models.values():
  position = val_list.index(model)
  for i in range(10):
    test = model.predict(df_test[i:(i+1)])
    result_list.append(test)
  t = max(result_list, key = result_list.count)

  
  if t== [[0]] or t ==[['blues']]:
    genre_detected = 'blues'
  elif t== [[1]] or t==[['pop']]:
    genre_detected = 'pop'
  elif t== [[2]] or t==[['jazz']]:
    genre_detected = 'jazz'
  elif t== [[3]] or t==[['reggae']]:
    genre_detected = 'reggae'
  elif t== [[4]] or t==[['metal']]:
    genre_detected = 'metal'
  elif t== [[5]] or t==[['disco']]:
    genre_detected = 'disco'
  elif t== [[6]] or t==[['classical']]:
    genre_detected = 'classical'
  elif t== [[7]] or t==[['hiphop']]:
    genre_detected = 'hiphop'
  elif t== [[8]] or t==[['rock']]:
   genre_detected = 'rock'
  else:
    genre_detected = 'country'
  print("The tested genre with {} model is:  {}".format(key_list[position], genre_detected))

'''

