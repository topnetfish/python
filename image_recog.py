# coding:utf-8

# 1.import modules 
from skimage import io,transform    # skimage包主要用于图像数据的处理，在该实验当中，
                                    # io模块主要图像数据的读取（imread）和输出（imshow）操作，transform模块主要用于改变图像的大小（resize函数）
import glob # glob包主要用于查找符合特定规则的文件路径名，跟使用windows下的文件搜索差不多，查找文件只用到三个匹配符：”*”, “?”, “[]”。
            # ”*”匹配0个或多个字符；”?”匹配单个字符；”[]”匹配指定范围内的字符，如：[0-9]匹配数字。
            # 该实验中，glob主要用于返回所有匹配的文件路径列表。它只有一个参数pathname，定义了文件路径匹配规则，这里可以是绝对路径，也可以是相对路径。
import os   # os模块主要用于处理文件和目录，比如：获取当前目录下文件，删除制定文件，改变目录，查看文件大小等。
            # 该案例中os主要用于列举当前目录下所有文件（listdir）和判断某一路径是否为目录（os.path.isdir）。
import tensorflow as tf # tensorflow是目前业界最流行的深度学习框架，在图像，语音，文本，目标检测等领域都有深入的应用。是该实验的核心，主要用于定义占位符，定义变量，创建卷积神经网络模型

import numpy as np  # numpy是一个基于python的科学计算包，在该实验中主要用来处理数值运算，包括创建爱你等差数组，生成随机数组，聚合运算等。
import time  # time模块主要用于处理时间系列的数据，在该实验主要用于返回当前时间戳，计算脚本每个epoch运行所需要的时间。

path='flower_photos/'  # 数据存放路径 
model_path='Model/model.ckpt'  # 模型保存路径 

# 设置图像处理之后的大小（由于是RGB格式数据，长宽高分别是100*100*3）
w=100
h=100
c=3

#2.read images
def read_img(path):                                                    # 定义函数read_img，用于读取图像数据，并且对图像进行resize格式统一处理
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]   # 创建层级列表cate，用于对数据存放目录下面的数据文件夹进行遍历，os.path.isdir用于判断文件是否是目录，然后对是目录文件的文件进行遍历
    imgs=[]                                                            # 创建保存图像的空列表
    labels=[]                                                          # 创建用于保存图像标签的空列表
    for idx,folder in enumerate(cate):                                 # enumerate函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和下标,一般用在for循环当中
        for im in glob.glob(folder+'/*.jpg'):                          # 利用glob.glob函数搜索每个层级文件下面符合特定格式“/*.jpg”进行遍历
            print('reading the images:%s'%(im))                        # 遍历图像的同时，打印每张图片的“路径+名称”信息
            img=io.imread(im)                                          # 利用io.imread函数读取每一张被遍历的图像并将其赋值给img
            img=transform.resize(img,(w,h))                            # 利用transform.resize函数对每张img图像进行大小缩放，统一处理为大小为w*h(即100*100)的图像
            imgs.append(img)                                           # 将每张经过处理的图像数据保存在之前创建的imgs空列表当中
            labels.append(idx)                                         # 将每张经过处理的图像的标签数据保存在之前创建的labels列表当中
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)     # 利用np.asarray函数对生成的imgs和labels列表数据进行转化，之后转化成数组数据（imgs转成浮点数型，labels转成整数型）

data,label=read_img(path)                                              # 将read_img函数处理之后的数据定义为样本数据data和标签数据label
print("shape of data:",data.shape)                                     # 查看样本数据的大小 
print("shape of label:",label.shape)                                   # 查看标签数据的大小 

#3.preprocessing data
# 打乱数据集，防止数据分布对模型结果的影响  
num_example=data.shape[0]           # 利用样本数据data的大小查看数据集的大小num_example
arr=np.arange(num_example)          # 利用numpy中的arange函数生成一个跟样本数据集一样大小的等差数组
np.random.shuffle(arr)              # 利用shuffle函数打乱创建的等差数组的顺序
data=data[arr]                      # 利用打乱的数组对样本数据进行打乱
label=label[arr]                    # 利用打乱的数组对样本标签同流程进行打乱

ratio=0.8                           # 创建比例，用于分割训练集和验证集，80%的数据用于模型训练，20%的数据用于模型验证 
s=np.int(num_example*ratio)         # 定义样本分割数量s，利用int函数去除掉浮点数因素的干扰
x_train=data[:s]                    # 训练样本数据
y_train=label[:s]                   # 训练标签数据
x_val=data[s:]                      # 验证样本数据
y_val=label[s:]                     # 验证标签数据

#4.placeholder
x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')    #创建输入数据的占位符，用于传递样本数据
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')         #创建输出数据的占位符，用于传递标签数据 

# 5 create model 
# 创建卷积神经网络模型,该模型是整个实验的核心,原始的模型只有5层隐藏层,包括两个卷积层,两个池化层,一个全连接层,但由于模型的识别效果较差,所以模型做了一系列的调整,
# 直到现在的稳定版本为止,包括11个隐藏层,包括4个卷积层,4个池化层,3个全连接层。
def model(input_tensor, train, regularizer):    # 定义卷积神经网络模型model，三个变量分别是输入张量input_tensor，用于区分训练过程还是验证过程的train，正则项regularizer。
    with tf.variable_scope('layer1-conv1'):     # 定义一个作用域：layer1-conv1，在该作用域下面可以定义相同名称的变量（用于变量）
        conv1_weights = tf.get_variable("weight",[5,5,3,32],initializer=tf.truncated_normal_initializer(stddev=0.1))  
        # 定义变量权重：weight，名称是weight；5,5代表卷积核的大小，3代表输入的信道数目，32代表输出的信道数目；initializer代表神经网络权重和卷积核的推荐初始值，生成截断正态分布随机数，服从标准差为0.1 
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))    
        # 定义变量偏置：bias，名称bias，[32]代表当前层的深度；initializer代表偏置的初始化，用函数tf.constant_initializer将其初始化为0，也可以初始化为tf.zeros_initializer或者tf.ones_initializer
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')   
        # 上面为定义卷积层：input_tensor为当前层的节点矩阵；conv1_weights代表卷积层的权重；strides为不同方向上面的步长；padding标识填充，有两种方式，SAME表示用0填充，“VALID”表示不填充。
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))  # 定义激活函数：利用bias_add给每个节点都加上偏置项，然后利用relu函数去线性化 

    with tf.name_scope("layer2-pool1"):     #定义一个：layer2-pool1（用于op）
        # 池化层可以优先缩小矩阵的尺寸，从而减小最后全连接层当中的参数；池化层既可以加快计算速度，也可以防止过拟合。
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID") 
        # ksize代表pool窗口的尺寸，首尾两个数必须是1，ksize最常用[1,2,2,1]和[1,3,3,1]；strides代表filter的步长，首尾两个数必须为1；padding代表填充方式；

    with tf.variable_scope("layer3-conv2"):                                                                             # 定义作用域（用于变量）
        conv2_weights = tf.get_variable("weight",[5,5,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))   # 定义权重
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))                          # 定义偏置
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')                                # 定义卷积层
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))                                                         # 定义激活函数

    with tf.name_scope("layer4-pool2"):                                                                                 # 定义命名空间（用于op）
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')                        # 定义池化层 

    with tf.variable_scope("layer5-conv3"):                                                                             # 定义作用域 （用于变量）
        conv3_weights = tf.get_variable("weight",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=0.1))  # 定义权重
        conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))                         # 定义偏置
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')                                # 定义卷积层
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))                                                         # 定义激活函数

    with tf.name_scope("layer6-pool3"):                                                                                 # 定义命名空间（用于op）
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')                        # 定义池化层 

    with tf.variable_scope("layer7-conv4"):                                                                             # 定义作用域（用于变量）
        conv4_weights = tf.get_variable("weight",[3,3,128,128],initializer=tf.truncated_normal_initializer(stddev=0.1)) # 定义权重
        conv4_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))                         # 定义偏置
        conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')                                # 定义卷积层
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))                                                         # 定义激活函数

    with tf.name_scope("layer8-pool4"):                                                                                 # 定义命名空间（用于op）
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')                        # 定义池化层  
        
        # layer8层的输出是矩阵：[6,6,128]，layer9的输入是向量，所以需要把layer8的输出转化成矩阵
        nodes = 6*6*128                                                                                                 # 矩阵的长度：4608
        reshaped = tf.reshape(pool4,[-1,nodes])  
        print("shape of reshaped:",reshaped.shape)                                                                       # reshape函数将pool4的输出转化成向量 

    with tf.variable_scope('layer9-fc1'):                                                                               # 定义作用域：
        fc1_weights = tf.get_variable("weight", [nodes, 1024],initializer=tf.truncated_normal_initializer(stddev=0.1))  # 定义全连接层的权重：
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))                                
        # 给全连接层的权重添加正则项，tf.add_to_collection函数可以把变量放入一个集合，把很多变量变成一个列表
        fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))                          # 定义全连接层的偏置：
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)                                                 # 定义激活函数：
        if train: fc1 = tf.nn.dropout(fc1, 0.5)                                                                         # 针对训练数据，在全连接层添加dropout层，防止过拟合 

    with tf.variable_scope('layer10-fc2'):
        fc2_weights = tf.get_variable("weight", [1024, 512],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer11-fc3'):
        fc3_weights = tf.get_variable("weight", [512, 5],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [5], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logit           

#regularizer
regularizer = tf.contrib.layers.l2_regularizer(0.001)       # 定义正则项：这里主要用L2正则项，用于防止过拟合，提升模型的泛化能力；

logits = model(x,False,regularizer)                         # 模型调用，其中train参数为False，表示不是训练状态;
print("shape of logits:",logits.shape)

b = tf.constant(value=1,dtype=tf.float32)                   # 定义一个常数
logits_eval = tf.multiply(logits,b,name='logits_eval')      # 常数与矩阵相乘 

#6.loss,optimizer and accuracy 
#loss和acc：
loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)  # 定义损失函数：计算预测标签logits和原始标签y_之间的sparse交叉熵

optimizer=tf.train.AdamOptimizer(learning_rate=0.001)                          # 定义优化器：使用adam优化器

train_op=optimizer.minimize(loss)                                              # 定义训练运算，将loss最小化

correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)       # tf.argmax返回最大数值的下标，tf.cast转换数据格式为int32，tf.equal对比两个序列中元素值，相等返回true不等返回false
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))                   # 计算模型准确性：tf.cast转换数据格式为float32，tf.reduce_mean函数用于求平均值

#7.train and test 
#定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):    # 定义批量提取数据函数：inputs为输入数据，targets为标签数据，batch_size为每一个batch数据集的大小，shuffle代表是否打乱
    assert len(inputs) == len(targets)                                         # 用来让程序测试这个condition，如果condition为false，那么raise一个AssertionError出来。
    if shuffle:                                                                # 如果需要对数据集记性打乱，那么记性该条件下面的语句。
        indices = np.arange(len(inputs))                                       # 生成索引
        np.random.shuffle(indices)                                             # 对索引进行打乱 
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):       # 生成以batch_size为等差的等差数列
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]                # 在该等差数列中，如果需要打乱数据集，则提取相应的样本数据索引
        else:
            excerpt = slice(start_idx, start_idx + batch_size)                 # 在该等差数列中，如果不需要打乱数据集，则提取相应的样本数据索引
        yield inputs[excerpt], targets[excerpt]                                # 提取相应的样本数据和标签数据 

n_epoch=10    # 样本训练次数，一个epoch代表用所有的数据训练一次                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
batch_size=64 # batch_size代表使用梯度下降训练模型时候，即每次使用batch_size个数据来更新参数      

saver=tf.train.Saver()  # 创建saver，用于保存模型内容（checkpoint，变量，协议缓存等）

sess=tf.Session()       # 定义一个会话，用于运行定义好的运算
sess.run(tf.global_variables_initializer())     # run之前，首先对模型参数进行全局初始化 

for epoch in range(n_epoch):                    # 总共有5个epoch，开始遍历每一层的epoch
    print("epoch:",epoch+1)                     # 在每层epoch的开始训练之前，打印当前epoch层级
    start_time = time.time()                    # 返回每个epoch开始训练时候的时间，用于计算每个epoch以及总共消耗了多少时间

    #training
    train_loss, train_acc, n_batch = 0, 0, 0                                                # 初始值默认为0 
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):    # 利用minibatches函数从训练数据中以打乱顺序的方式提取批数据进行训练，批大小为64
        _,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})     # 运行train_op，loss，acc三个运算，利用feed_dict给占位符传输数据进行遍历训练 
        train_loss += err; train_acc += ac; n_batch += 1                                    # 将每次迭代的损失值err，准确率ac，批次数n_batch进行累计 
    print("   train loss: %f" % (np.sum(train_loss)/ n_batch))                              # 计算平均loss值
    print("   train acc: %f" % (np.sum(train_acc)/ n_batch))                                # 计算平均acc值 

    #validation
    val_loss, val_acc, n_batch = 0, 0, 0                                                    # 初始值默认为0 
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):           # 利用minibatches函数从验证数据中以不打乱顺序的方式提取批数据进行验证，批大小为64
        err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})                 # 利用feed_dict给占位符传输数据进行遍历验证，运行train_op，loss，acc三个op运算
        val_loss += err; val_acc += ac; n_batch += 1                                        # 将每次迭代的损失值err，准确率ac，批次数n_batch进行累计 
    print("   validation loss: %f" % (np.sum(val_loss)/ n_batch))                           # 计算平均loss值
    print("   validation acc: %f" % (np.sum(val_acc)/ n_batch))                             # 计算平均acc值 
    print("   epoch time: %f" % (time.time()-start_time))                                  # 计算每个epoch所消耗的时间 
    print('-------------------------------------------------------')

#8.save and restore model 
saver.save(sess,model_path)     # 利用saver的save函数对sess运算后的模型进行保存，保存路径为model_path

sess.close()                    # sess运行结束之后，关闭session

# 从原始数据集的每个类别中各自随机抽取一张图像进行模型验证
path1 = "flower_photos/daisy/5547758_eea9edfd54_n.jpg"
path2 = "flower_photos/dandelion/7355522_b66e5d3078_m.jpg"
path3 = "flower_photos/roses/394990940_7af082cf8d_n.jpg"
path4 = "flower_photos/sunflowers/6953297_8576bf4ea3.jpg"
path5 = "flower_photos/tulips/10791227_7168491604.jpg" 

# 定义花类字典，对每种花都赋值一个数值类别
flower_dict = {0:'dasiy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'} 

# 定义转换之后测试花类图像的大小（长宽高分别是100,100,3）
w=100
h=100
c=3

# 定义read_one_image函数，用于将验证图像转换成统一大小的格式（100*100*3）
def read_one_image(path):               # 定义函数read_one_image
    img = io.imread(path)               # 利用io.imread函数读取图像 
    img = transform.resize(img,(w,h))   # 利用transform.resize函数对读取的图像数据进行格式统一化处理 
    return np.asarray(img)              # 对img图像数据进行转化 

with tf.Session() as sess:              # 创建会话，用于执行已经定义的运算
    data = []                           # 定义空白列表，用于保存处理后的验证数据 
    data1 = read_one_image(path1)       # 利用自定义函数read_one_image依次对5张验证图像进行格式标准化处理
    data2 = read_one_image(path2)
    data3 = read_one_image(path3)
    data4 = read_one_image(path4)
    data5 = read_one_image(path5)
    data.append(data1)                  # 将处理过后的验证图像数据保存在前面创建的空白data列表当中
    data.append(data2)
    data.append(data3)    
    data.append(data4)
    data.append(data5)

    saver = tf.train.import_meta_graph('Model/model.ckpt.meta')     # 利用import_meta_graph函数直接加载之前已经持久化了的模型内容
    saver.restore(sess,tf.train.latest_checkpoint('Model'))         # 利用restore函数加载已经训练好的模型，并利用tf.train.latest_checkpoint函数提取最近一次保存的模型

    graph = tf.get_default_graph()              # 获取当前的默认计算图 

    x = graph.get_tensor_by_name("x:0")         # 返回给定名称的tensor
    # print(x)                                  # 返回加载的模型的参数 

    feed_dict = {x:data}                        # 利用feed_dict，给占位符传输数据 

    logits = graph.get_tensor_by_name("logits_eval:0")      # 返回logits_eval对应的tensor
    print(logits)

    classification_result = sess.run(logits,feed_dict)      # 利用feed_dict把数据传输到logits进行验证图像预测

    print(classification_result)                            # 打印预测矩阵
    print(tf.argmax(classification_result,1).eval())        # 打印预测矩阵每一行的最大值的下标 

    output = []                                             # 定义空白列表output
    output = tf.argmax(classification_result,1).eval()      # 选择出预测矩阵每一行最大值的下标，并将字符串str当成有效的表达式来求值并返回计算结果，将其赋值给output
    print(output)
    print(output.shape)

    for i in range(len(output)):                            # 遍历len(output)=5的花的类型
        print("flower",i+1,"prediction:"+flower_dict[output[i]])    # 输出每种花预测值最高的选项 
