import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 

boston_housing = tf.keras.datasets.boston_housing
(train_x,train_y),(test_x,test_y) = boston_housing.load_data()
train_x = train_x[:,:12]
test_x = test_x[:,:12]

train_x1=(train_x-train_x.min(axis=0))/(train_x.max(axis=0)-train_x.min(axis=0))
test_x1=(test_x-test_x.min(axis=0))/(test_x.max(axis=0)-test_x.min(axis=0))

x_train = tf.cast(train_x1,dtype = tf.float32)
x_valid = tf.cast(test_x1,dtype=tf.float32)

def model(x,w,b):
    return tf.matmul(x,w)+b

w = tf.Variable(tf.random.normal([12,1],mean = 0.0,stddev = 1.0,dtype = tf.float32))
b = tf.Variable(tf.zeros(1),dtype = tf.float32)

training_epochs = eval(input("请输入您要进行迭代的次数："))       
learning_rate = float(input("请输入你要调整的学习率："))
batch_size = 10

def loss(x,y,w,b):
    err = model(x,w,b) - y
    squared_err = tf.square(err)
    return tf.reduce_mean(squared_err)

def gard(x,y,w,b):
    with tf.GradientTape() as tape:
        loss_ = loss(x,y,w,b)
    return tape.gradient(loss_,[w,b])

optimizer = tf.keras.optimizers.SGD(learning_rate)

loss_list_train = []
loss_list_valid = []
for epoch in range (training_epochs):
    for step in range(40):
        xs = x_train[step * batch_size: (step + 1) * batch_size,:]
        ys = train_y[step * batch_size: (step + 1) * batch_size]
        grads = gard(xs,ys,w,b)
        optimizer.apply_gradients(zip(grads,[w,b]))
    loss_train = loss(x_train,train_y,w,b).numpy()
    loss_valid = loss(x_valid,test_y,w,b).numpy()
    loss_list_train.append(loss_train)
    loss_list_valid.append(loss_valid)
    print("epoch={:3d},train_loss={:4f},valid_loss={:4f}".format(epoch+1,loss_train,loss_valid))
  #  print("epoch={:3d}".format(epoch+1))