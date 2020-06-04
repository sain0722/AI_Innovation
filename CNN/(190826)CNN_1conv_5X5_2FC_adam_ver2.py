# elapsed time =  0:15:30.866821
# Accuracy = 99.07%

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from datetime import datetime      # datetime.now() 를 이용하여 학습 경과 시간 측정

# read_data_sets() 를 통해 데이터를 객체형태로 받아오고
# one_hot 옵션을 통해 정답(label) 을 one-hot 인코딩된 형태로 받아옴

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# mnist 데이터 셋은 train, test, validation 3개의 데이터 셋으로 구성되어 있으며.
# num_examples 값을 통해 데이터의 갯수 확인 가능함

print("\n", mnist.train.num_examples, mnist.test.num_examples, mnist.validation.num_examples)

# 데이터는 784(28x28)개의 픽셀을 가지는 이미지와
# 10(0~9)개 클래스를 가지는 one-hot 인코딩된 레이블(정답)을 가지고 있음

print("\ntrain image shape = ", np.shape(mnist.train.images))
print("train label shape = ", np.shape(mnist.train.labels))
print("test image shape = ", np.shape(mnist.test.images))
print("test label shape = ", np.shape(mnist.test.labels))

#######################################################################################

# Hyper-Parameter
learning_rate = 0.001  # 학습률
epochs = 30            # 반복횟수
batch_size = 100      # 한번에 입력으로 주어지는 MNIST 개수

#######################################################################################

# 입력과 정답을 위한 플레이스홀더 정의
X = tf.placeholder(tf.float32, [None, 784])

A1 = X_img = tf.reshape(X, [-1, 28, 28, 1])   # image 28X28X1 (black/white)

T = tf.placeholder(tf.float32, [None, 10])

#######################################################################################

# 1번째 컨볼루션 층, 5X5X32 필터
W2 = tf.Variable(tf.random_normal([5, 5, 1, 32], stddev=0.01))
b2 = tf.Variable(tf.constant(0.1, shape=[32]))

# 1번째 컨볼루션 연산을 통해 28 X 28 X1  => 28 X 28 X 32
C2 = tf.nn.conv2d(A1, W2, strides=[1, 1, 1, 1], padding='SAME')

# relu
Z2 = tf.nn.relu(C2+b2)

# 1번째 max pooling을 통해 28 X 28 X 32  => 14 X 14 X 32
A2 = P2 = tf.nn.max_pool(Z2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#######################################################################################

# 완전연결층, 14X14X32 개 입력 출력은 256개 의 은닉층 개념
A2_flat = P2_flat = tf.reshape(A2, [-1, 14*14*32])

W3 = tf.Variable(tf.random_normal([14*14*32, 256], stddev=0.01))
b3 = tf.Variable(tf.random_normal([256]))

Z3 = tf.matmul(A2_flat, W3) + b3

A3 = tf.nn.relu(Z3)

#######################################################################################

# 출력층
W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
b4 = tf.Variable(tf.random_normal([10]))

# 출력층 선형회귀  값 Z4, 즉 softmax 에 들어가는 입력 값
Z4 = logits = tf.matmul(A3, W4) + b4

y = A4 = tf.nn.softmax(Z4)

#######################################################################################

loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z4, labels=T) )

optimizer = tf.train.AdamOptimizer(learning_rate)

train = optimizer.minimize(loss)

#######################################################################################

# batch_size X 10 데이터에 대해 argmax를 통해 행단위로 비교함
predicted_val = tf.equal( tf.argmax(A4, 1), tf.argmax(T, 1) )

# batch_size X 10 의 True, False 를 1 또는 0 으로 변환
accuracy = tf.reduce_mean(tf.cast(predicted_val, dtype=tf.float32))

# index list 출력
accuracy_index = tf.cast(predicted_val, dtype=tf.float32)

# 예측값 처리
predicted_list = tf.argmax(A4, 1)

#######################################################################################

index_label_prediction_list = []

with  tf.Session()  as sess:

    sess.run(tf.global_variables_initializer())  # 변수 노드(tf.Variable) 초기화

    start_time = datetime.now()

    for i in range(epochs):    # 50 번 반복수행

        total_batch = int(mnist.train.num_examples / batch_size)  # 55,000 / 100

        for step in range(total_batch):

            batch_x_data, batch_t_data = mnist.train.next_batch(batch_size)

            loss_val, _ = sess.run([loss, train], feed_dict={X: batch_x_data, T: batch_t_data})

            if step % 100 == 0:
                print("epochs = ", i, ", step = ", step, ", loss_val = ", loss_val)

    end_time = datetime.now()

    print("\nelapsed time = ", end_time - start_time)

    # Accuracy 확인
    test_x_data = mnist.test.images    # 10000 X 784
    test_t_data = mnist.test.labels    # 10000 X 10

    accuracy_val, predicted_list_val, index_label = sess.run([accuracy, predicted_list, accuracy_index], feed_dict={X: test_x_data, T: test_t_data})

    print("\nAccuracy = ", accuracy_val)
    print("type(accuracy_val) = ", type(accuracy_val), ', type(predicted_list_val) = ', type(predicted_list_val), ', type(index_label) = ', type(index_label))
    print("index_label.shape = ", index_label.shape)

    index_label_list = list(index_label)
    print("length of index_label_list = ", len(index_label_list))
    print("false label count = ", index_label_list.count([0]))

    # numpy type 으로 디버그
    temp_list = []

    for index in range(len(index_label)):

        if index_label[index] == 0:

            temp_list.append(index)
            temp_list.append(np.argmax(test_t_data[index]))  # one-hot encoding 이므로 argmax 로 정답 추출
            temp_list.append(predicted_list_val[index])

            index_label_prediction_list.append(temp_list)

            temp_list = []

    print("\nlength of index_label_false_list", len(index_label_prediction_list))


