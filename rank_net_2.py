import io
import math
import numpy as np
from keras.layers import Activation, Add, Dense, Input, Lambda
from keras.models import Model

INPUT_DIM = 136

# Model.
h_1 = Dense(68, activation = "sigmoid")
h_2 = Dense(34)
h_3 = Dense(17)
s = Dense(1)

# Relevant document score.
rel_doc = Input(shape = (INPUT_DIM, ), dtype = "float32")
h_1_rel = h_1(rel_doc)
h_2_rel = h_2(h_1_rel)
h_3_rel = h_3(h_2_rel)
rel_score = s(h_3_rel)

# Irrelevant document score.
irr_doc = Input(shape = (INPUT_DIM, ), dtype = "float32")
h_1_irr = h_1(irr_doc)
h_2_irr = h_2(h_1_irr)
h_3_irr = h_3(h_2_irr)
irr_score = s(h_3_irr)

# Subtract scores.
negated_irr_score = Lambda(lambda x: -1 * x, output_shape = (1, ))(irr_score)
diff = Add()([rel_score, negated_irr_score])

# Pass difference through sigmoid function.
prob = Activation("sigmoid")(diff)

# Build model.
model = Model(inputs = [rel_doc, irr_doc], outputs = prob)
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=['mse'])

# extract train data.
with io.open('D:/school/IE_IR/assignment/2/Fold1/train.txt', 'r') as f_train:
    l = f_train.readlines()
    qid_l = [0]
    for i in range(len(l)-1):
        #same qid?
        if(l[i].split(' ')[1] != l[i+1].split(' ')[1]):
            qid_l.append(i+1)
    qid_l.append(len(l))
    L_y = []
    L_x1 = []
    L_x2 = []
    for i in range(len(qid_l)-1):
        for j in range(qid_l[i], qid_l[i+1]-1):
            #analyze relevent value
            if(l[j].split(' ')[0] > l[j+1].split(' ')[0]):
                rel = 1
            elif l[j].split(' ')[0] == l[j+1].split(' ')[0]:
                rel = 0.5
            else:
                rel = 0
            L_y.append([rel])
            L = []
            #get query score
            for k in l[j].split(' ')[2:138]:
                sc = float(k.split(':')[1])
                L.append(sc)
            L_x1.append(L)
            L = []
            for k in l[j+1].split(' ')[2:138]:
                sc = float(k.split(':')[1])
                L.append(sc)
            L_x2.append(L)            
X_1 = np.array(L_x1)
X_2 = np.array(L_x2)
y = np.array(L_y)

# Train model.
NUM_EPOCHS = 10
BATCH_SIZE = 20
history = model.fit([X_1, X_2], y, batch_size = BATCH_SIZE, epochs = NUM_EPOCHS, verbose = 1)

with io.open('D:/school/IE_IR/assignment/2/Fold1/vali.txt', 'r') as f_vali:
    l = f_vali.readlines()
    qid_l = [0]
    for i in range(len(l)-1):
        #same qid?
        if(l[i].split(' ')[1] != l[i+1].split(' ')[1]):
            qid_l.append(i+1)
    qid_l.append(len(l))
    L_y = []
    L_x1 = []
    L_x2 = []
    for i in range(len(qid_l)-1):
        for j in range(qid_l[i], qid_l[i+1]-1):
            #analyze relevent value
            if(l[j].split(' ')[0] > l[j+1].split(' ')[0]):
                rel = 1
            elif l[j].split(' ')[0] == l[j+1].split(' ')[0]:
                rel = 0.5
            else:
                rel = 0
            L_y.append([rel])
            L = []
            #get query score
            for k in l[j].split(' ')[2:138]:
                sc = float(k.split(':')[1])
                L.append(sc)
            L_x1.append(L)
            L = []
            for k in l[j+1].split(' ')[2:138]:
                sc = float(k.split(':')[1])
                L.append(sc)
            L_x2.append(L)            
Xv_1 = np.array(L_x1)
Xv_2 = np.array(L_x2)
yv = np.array(L_y)
loss, accuracy = model.evaluate([Xv_1, Xv_2], yv, batch_size = 10000)
print(' loss: ', loss, ' accuracy: ',  accuracy)

with io.open('D:/school/IE_IR/assignment/2/Fold1/test.txt', 'r') as f_test:
    l = f_test.readlines()
    qid_l = [0]
    for i in range(len(l)-1):
        #same qid?
        if(l[i].split(' ')[1] != l[i+1].split(' ')[1]):
            qid_l.append(i+1)
    qid_l.append(len(l))
    L_x1 = []
    L_x2 = []
    IDCG = 0
    for i in range(len(qid_l)-1):
        #IDCG count
        L_relv = []
        for j in range(qid_l[i], qid_l[i+1]):
            L_relv.append(int(l[j].split(' ')[0]))
        in_s = np.argsort(np.array(L_relv))
        IDCG_l = 0
        for j in range(len(L_relv)-1, -1 ,-1):
            IDCG_l += L_relv[in_s[j]] / math.log(len(L_relv)-1 - j +1 + 1, 2)
        IDCG += IDCG_l
        
        for j in range(qid_l[i], qid_l[i+1]-1):
            L = []
            #get query score
            for k in l[j].split(' ')[2:138]:
                sc = float(k.split(':')[1])
                L.append(sc)
            L_x1.append(L)
            L = []
            for k in l[j+1].split(' ')[2:138]:
                sc = float(k.split(':')[1])
                L.append(sc)
            L_x2.append(L)            
Xt_1 = np.array(L_x1)
Xt_2 = np.array(L_x2)
r = model.predict([Xt_1, Xt_2], batch_size = 10000)
IDCG /= len(qid_l)-1
#sort : 0 = 0 ~ 0.3 , 0.5=0.3 ~ 0.7, 1=0.7~1
#NDCG50
NDCG_50 = 0
DCG = 0
count = 0
for i in range(len(qid_l)-1):
    if qid_l[i+1] - qid_l[i] >=50:
        count += 1
        now = qid_l[i]
        sort_L = []#store index
        for j in range(qid_l[i], qid_l[i+1]-1):#last row don't have next row(irr)(avoid r out of range)
            if r[j - i][0] >= 0.7 or j == qid_l[i+1]-2:
                if j == qid_l[i+1]-2 and r[j - i][0] < 0.7:#add last row   condition
                    sort_L.append(j+1)
                sort_L.append(j)
                for k in range(j-1, now - 1, -1):
                    sort_L.append(k)
                    if(len(sort_L) >=50):
                        break
                now = j + 1
                if(len(sort_L) >=50):
                        break
                if j == qid_l[i+1]-2 and r[j - i][0] >= 0.7:#add last row   condition
                    sort_L.append(j+1)
        DCG_l = 0
        for m in range(50):
            #print(len(l), ' ',len(sort_L),' ', m,' ', sort_L[m])
            DCG_l += int(l[sort_L[m]].split(' ')[0]) / math.log(m+1 + 1, 2)
        DCG += DCG_l
if count != 0:
    DCG /= count
NDCG_50 = DCG / IDCG
print('NDCG_50:', NDCG_50)
#NDCG100
NDCG_100 = 0
DCG = 0
count = 0
for i in range(len(qid_l)-1):
    if qid_l[i+1] - qid_l[i] >=100:
        count += 1
        now = qid_l[i]
        sort_L = []#store index
        for j in range(qid_l[i], qid_l[i+1] - 1):
            if r[j - i][0] >= 0.7 or j == qid_l[i+1]-2:
                if j == qid_l[i+1]-2 and r[j - i][0] < 0.7:#add last row   condition
                    sort_L.append(j+1)
                sort_L.append(j)
                for k in range(j-1, now - 1 , -1):
                    sort_L.append(k)
                    if(len(sort_L) >=100):
                        break
                now = j + 1
                if(len(sort_L) >=100):
                        break
                if j == qid_l[i+1]-2 and r[j - i][0] >= 0.7:#add last row   condition
                    sort_L.append(j+1)
        DCG_l = 0
        for m in range(100):
            DCG_l += int(l[sort_L[m]].split(' ')[0]) / math.log(m+1 + 1, 2)
        DCG += DCG_l
if count != 0:
    DCG /= count
NDCG_100 = DCG / IDCG
print('NDCG_100:', NDCG_100)
