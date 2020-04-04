import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(train_x, train_y),(test_x, test_y) = mnist.load_data()

plt.rcParams["font.sans-serif"]="SimHei"

plt.figure(figsize=(10,10))
plt.suptitle("MNIST测试集样本",fontsize=20,color="red")

plt.subplot(221)
for i in range(16):
    num = np.random.randint(1,50000)

    plt.subplot(4,4,i+1)
    plt.axis("off")
    plt.imshow(train_x[num],cmap='gray')
    strtitle="标签值："+str(train_y[num])
    plt.title(strtitle,fontsize=14)

plt.show()