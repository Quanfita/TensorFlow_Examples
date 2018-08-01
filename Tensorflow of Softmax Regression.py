#加载MNIST数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#查看数据集情况55000个样本，测试集有10000个样本，验证集有5000个样本
print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)

#载入TensorFlow库，并创建一个新的InteractiveSession
import tensorflow as tf
sess = tf.InteractiveSession()
#创建一个Placeholder，即输入数据的地方
x = tf.placeholder(tf.float32,[None,784])#第一个参数是数据类型
                              #第二个参数是tensor的shape也就是数据的尺寸
#接下来给Softmax Regression模型中的weights和biases创建Variable对象
#初始化为0
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#实现Softmax Regression算法  y = softmax(wx + b)
y = tf.nn.softmax(tf.matmul(x,w) + b)
#Softmax是tf.nn下的一个函数，tf.nn包含了大量网络神经网络的组件
#tf.matmul是TensorFlow中的矩阵乘法函数

#需要定义一个loss function来描述模型对问题的分类精度
#loss越小，代表模型的分类结果与真实值的偏差越小，也就是模型越精确
#对多分类问题，通常使用cross-entropy作为loss function
#先定义一个placeholder，输入是真实的label，用来计算cross-entropy
#tf.reduce_sum是求和
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                              reduction_indices=[1]))
#采用常见的随机梯度下降SGD
#调用tf.train.GradientDescentOptimizer,并设置学习速率为0.5，优化目标设定为cross-entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#使用TensorFlow的全局参数初始化器tf.global_variables_initializer,并执行run方法
tf.global_variables_initializer().run()
#最后开始迭代的执行训练操作train_step
#每次都随机从训练集中抽取100条样本构成一个mini-batch，并feed给placeholder，然后调用train_step对这些样本进行训练
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})
#完成训练后，就可以对模型的准确率进行验证
#tf.argmax是从一个tensor中寻找最大值的序号
#tf.equal方法是用来判断预测数字类别是否是正确的类别
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#统计全部样本预测的accuracy，需要先用tf.cast将之前的correct_prediction输出的bool值转化为float32，再求平均
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#计算模型在测试集上的准确率，再将结果打印出来
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))
