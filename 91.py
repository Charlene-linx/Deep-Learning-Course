import tensorflow as tf 
import numpy as np 
x1 = tf.constant([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
x2 = tf.constant([3,2,2,3,1,2,3,2,2,3,1,1,1,1,2,2],dtype = tf.float32)
y = tf.constant([145.00,110.00,93.00,116.00,35.32,104.00,118.00,91.00,62.00,133.00,51.00,15.00,78.50,69.65,75.69,95.30])
x0 = tf.ones([len(x1)])
X = tf.stack((x0,x1,x2),axis = 1)
Y = tf.reshape(y,[-1,1])
xt = tf.transpose(X)
xtx_1 = tf.linalg.inv(xt@X)
xtx_1_xt = xtx_1@xt
W = xtx_1_xt@Y
W = tf.reshape(W,[-1])
while(1):
    print("请输入房屋面积、房间数量以及预测房屋销售价格：")
    x1_text = float(input("房屋面积："))
    x2_text = int(input("房间数量："))
    if((x1_text<20 or x1_text > 500 ) or ( x2_text < 1 or x2_text ) > 10):
        print("房屋面积在20-500间，房间数量在1-10间。请重新输入房屋面积和房间数量：")
    else:
        break
y_pred = W[1] * x2_text + W[2] * x2_text + W[0]
print("预测房屋价格：",(tf.round(y_pred,2)).numpy(),"万元。")