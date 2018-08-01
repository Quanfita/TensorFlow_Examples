#载入TensorFlow并加载MNIST数据集，创建一个InteractiveSession
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
sess = tf.InteractiveSession()

#给隐含层的参数设置Variable并进行初始化
#in_units是输入节点数，h1_units即隐含层的输出节点数（设置在200~1000内）
in_units = 784
h1_units = 300
#w1,b1是隐含层的权重和偏置，将偏置全部赋值为0，并将权重初始化为截断的正态分布
#权重的标准差为0.1，可以通过tf.truncated__normal方便的实现
w1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
w2 = tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))

#输入x的placeholder
#因为在训练和预测时Dropout的比率keep_prob（即保留节点的概率）是不一样的
#通常在训练时小于1，而预测时则等于1，所以把Dropout的比率作为计算图的输入，并定义成一个placeholder
x = tf.placeholder(tf.float32,[None,in_units])
keep_prob = tf.placeholder(tf.float32)

#下面定义模型结构
#首先定义一个隐含层，可以通过tf.nn.relu(tf.matmul(x,w1)+b1)实现一个激活函数为ReLU的隐含层
#这个隐含层的计算公式就是y=relu(w1*x+b1)
#接下来调用tf.nn.dropout实现Dropout的功能
#最后是输出层，Softmax
hidden1 = tf.nn.relu(tf.matmul(x,w1)+b1)
hidden1_drop = tf.nn.dropout(hidden1,keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop,w2)+b2)

#接下来定义损失函数和选择优化器来优化loss，这里还是继续使用交叉信息熵
#优化器选择Adagrad，并把学习速率设为0.3，直接使用tf.train.AdagradOptimizer就可以了
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),
                                             reduction_indices = [1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

#训练
#我们加入了keep_prob作为计算图的输入，并且在训练时设为0.75，即保留75%的节点，其余25%置为0
#对越大规模的神经网络，Dropout的效果越显著
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys,keep_prob:0.75})

#对模型准确率评测
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
