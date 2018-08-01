#先导入常用库和MNIST数据
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#自编码器中会使用到一种参数初始化方法Xavier initialization
#通过tf.random_uniform创建了一个均匀分布
#fan_in是输入节点的数量，fan_out是输出节点的数量
def xavier_init(fan_in,fan_out,constant = 1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out),
                            minval = low, maxval = high,
                            dtype = tf.float32)

#定义一个去噪自编码的class，方便以后使用，包含一个构建函数__init__(),还有一些常用的成员函数
class AdditiveGaussianNoiseAutoencoder(object):
    #__init__函数包含这样几个输入：
    #n_input（输入变量数）、n_hidden（隐含层节点数）、n_transfer_function（隐含层激活函数，默认为softplus）、
    #optimizer（优化器，默认为Adam）、scale（高斯噪声系数，默认为0.1）
    #其中，class内的scale参数做成一个placeholder，初始化采用了_initialize_weights函数
    def __init__(self,n_input,n_hidden,transfer_function = tf.nn.softplus,
                optimizer = tf.train.AdamOptimizer(),scale = 0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        
        #接下来定义网络结构，我们为输入x创建一个维度为n_input的placeholder
        #然后建立一个能提取特征的隐含层，我们先将输入的x加上噪声，即self.x+scale*tf.random_normal((n_input,))
        #然后用tf.matmul将加了噪声的输入与隐含层的权重w1相乘，并使用tf.add加上隐含层的偏置b1
        #最后使用self.transfer对结果进行激活函数处理
        #经过隐含层后，我们需要在输出层进行数据复原、重建操作（即建立reconstruction层）
        #这里就不需要激活函数了，直接将隐含层的输出self.hidden乘上输出层的权重w2，再加上输出层的偏置b2即可
        self.x = tf.placeholder(tf.float32,[None,self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x+scale*tf.random_normal((n_input,)),
                                                    self.weights['w1']),self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])
        
        #接下来定义自编码器的损失函数，这里直接使用平方误差作为cost
        #即用tf.substract计算输出（self.reconstruction）与输入（self.x）只差
        #在使用tf.pow求差的平方，最后使用tf.reduce_sum求和即可得到平方误差
        #再定义训练操作为优化器self.optimizer对损失self.cost
        #最后创建Session，并初始化自编码器的全部模型参数
        self.cost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,
                                                        self.x),2.0))
        self.optimizer = optimizer.minimize(self.cost)
        
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
    #创建一个名为all_weights的字典dict，然后将w1、b1、w2、b2全部存入其中，最后返回all_weights
    #其中w1需要用前面定义的xavier_init函数初始化，我们直接传入输入节点数和隐含节点数
    #然后Xavier即可返回一个比较适合于softplus等激活函数的权重初始分布
    #而偏置b1只需要使用tf.zeros全部置为0即可
    #对于输出层self.reconstruction，因为没有使用激活函数，这里将w2、b2全部初始化为0即可
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],dtype = tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,
                                                 self.n_input],dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],dtype = tf.float32))
        
        return all_weights
    
    #定义计算损失cost及执行一步训练的函数
    #函数里只需让Session执行两个计算图的节点，分明是损失函数cost和训练过程optimizer
    #输入的feed_dict包括输入数据x，以及噪声的系数scale
    def partial_fit(self,X):
        cost,opt = self.sess.run((self.cost,self.optimizer),
                                feed_dict = {self.x:X,self.scale:self.training_scale})
        return cost
    
    #只求损失cost的函数，评测时会用到
    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict = {self.x:X,self.scale:self.training_scale})
    
    #返回自编码器隐含层的输出结果
    #目的是提供一个接口来获取抽象后的特征
    #自编码器的隐含层的最主要功能就是学习出数据中的高阶特征
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict = {self.x:X,self.scale:eslf.training_scale})
    
    #将隐含层的输出结果作为输入，通过之后重建层将提取到的高阶特征复原为原始数据
    #这个接口和前面的transform正好将整个自编码器拆分为两部分，这里的generate接口是后半部分
    def generate(self,hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights["b1"])
        return self.sess.run(self.reconstruction,
                            feed_dict = {self.hidden:hidden})
    
    #reconstruct函数，它整体运行一遍复原过程，包括提取高阶特征和通过高阶特征复原数据
    #输入数据是原数据，输出数据是复原后的数据
    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict = {self.x:X,
                                                              self.scale:self.training_scale})
    
    #获取隐含层的权重w1
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    
    #获取隐含层的偏置系数b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])

#读取示例数据
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)

#先定义一个对训练、测试数据进行标准化处理的函数
def standard_scale(X_train,X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train,X_test

#再定义一个获取随机block数据的函数：取一个从0到len(data)-batch_size之间的随机整数
#再以这个随机数作为block的起始位置，然后顺序取到一个batch size的数据
def get_random_block_from_data(data,batch_size):
    start_index = np.random.randint(0,len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]

#使用standard_scale函数对训练集、测试集进行标准化交换
X_train,X_test = standard_scale(mnist.train.images,mnist.test.images)
#定义几个常用参数，总训练样本数，最大训练的轮数（epoch）设为20，batch_size设为128
#并设置每隔一轮（epoch）就显示一次损失cost
n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1

#创建一个AGN自编码器的实例
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784,
                                              n_hidden = 200,
                                              transfer_function = tf.nn.softplus,
                                              optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
                                              scale = 0.01)

#下面开始训练过程
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples/batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train,batch_size)
        
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size
        
    if epoch % display_step == 0:
        print("Epoch:",'%04d' % (epoch+1),"cost=","{:.9f}".format(avg_cost))

#对训练完的模型进行性能测试
print("Total cost:"+str(autoencoder.calc_total_cost(X_test)))
