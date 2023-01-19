import pandas as pd
import sklearn.neural_network as ann
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import sklearn.metrics as met

data = pd.read_csv("datatraining.csv")

# CARA MENGHAPUS ATTRIBUT DARI DATASET
data.drop(["date"], inplace=True, axis=1)
print(data.isnull().sum())
# print(data.head(5))

# MEMILIH ATTRIBUT YANG MAU DI TRAINING
X = data[["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]]
y = data[["Occupancy"]]

# MEMBAGI MENJADI TRAINING DATASET DAN TEST DATASET
X_train, X_test, y_train, y_test = ms.train_test_split(
    X, y, test_size=0.2, random_state=0)

# MENGETEST APAKAH DATASETNYA SUDAH DIBAGI
# print(data.count())
# print(X_train.count())
# print(X_test.count())

scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
scl.fit(X_train)
X_train = scl.transform(X_train)
X_test = scl.transform(X_test)
print(X_train.min())
print(X_test.max())

mlp = ann.MLPClassifier(hidden_layer_sizes=(3), max_iter=(5))
mlp.fit(X_train, y_train)

y_prediksi = mlp.predict(X_test)
print(y_prediksi)

# print(met.classification_report(y_test, y_prediksi))
# print(mlp.coefs_)