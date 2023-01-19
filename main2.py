# =================================================
#            PROGRAMNYA BERHASIL DIBUAT
#          Membuat Neural Network Manual
# ==================================================


from fungsi import sigmoid
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

# Data Sopyan
# Tinggi Badan
x1 = np.array([18 + 1*np.random.randn() for i in range(1,100)])
# Berat Badan
x2 = np.array([5 + 1*np.random.randn() for i in range(1,100)])
# status sopyan
y1 = np.array([1 for i in range(1,100)])
# status syauri
y2 = np.array([0 for i in range(1,100)])
# label
label = ["sopyan" for i in range(1,100)]


# data compiled
data_sopyan = {
    "x1":x1,
    "x2":x2,
    "y1":y1,
    "y2":y2,
    "label":label
}


data_frame1 = pd.DataFrame(data_sopyan)


# Data Syauri
# Tinggi Badan
x1 = np.array([10 + 1*np.random.randn() for i in range(1,100)])
# Berat Badan
x2 = np.array([8 + 1*np.random.randn() for i in range(1,100)])
# status sopyan
y1 = np.array([0 for i in range(1,100)])
# status syauri
y2 = np.array([1 for i in range(1,100)])
# label
label = ["syauri" for i in range(1,100)]


# data compiled
data_syauri = {
    "x1":x1,
    "x2":x2,
    "y1":y1,
    "y2":y2,
    "label":label
}


data_frame2 = pd.DataFrame(data_syauri)
data_frame = pd.concat([data_frame1, data_frame2])
# Mengacak data
data_frame = shuffle(data_frame)
# indexnya supaya berurut
data_frame.reset_index(inplace=True, drop= True)

# print(data_frame.head())


# Visualisasi data
plt.scatter(data_frame1.x1, data_frame1.x2, c="blue")
plt.scatter(data_frame2.x1, data_frame2.x2, c="red")
# plt.show()


# Metode Neural Network
# 1. Matrix Weight
w11 = np.random.uniform(-0.01, 0.01)
w12 = np.random.uniform(-0.01, 0.01)
w21 = np.random.uniform(-0.01, 0.01)
w22 = np.random.uniform(-0.01, 0.01)

W = np.array([[w11, w12], [w21, w22]])
# print(W)

# 2. Iterasi Neural Networknya
learning_rate = 0.1

for index, baris in data_frame.iterrows():
    input = np.array([[baris.x1], [baris.x2]])
    output_hidden = np.dot(W, input)
    output_learn = sigmoid(output_hidden)
    output_actual = np.array([[baris.y1], [baris.y2]])

    error = output_actual - output_learn

    delta_W = np.dot(learning_rate*error*output_learn*(1-output_learn), input.T)

    W = W + delta_W

    tebakan = np.argmax(output_learn)
    jawaban = np.argmax(output_actual)

    label_tebakan = "sopyan" if tebakan == 0 else "syauri"
    print(f"index = {index}, tebakan = {label_tebakan}, jawaban = {baris.label}")