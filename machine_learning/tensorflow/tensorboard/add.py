import tensorflow as tf

with tf.name_scope('graph') as scope:
    matrix1 = tf.constant([[3., 3.]],name ='matrix1')  #1 row by 2 column
    matrix2 = tf.constant([[2.],[2.]],name ='matrix2') # 2 row by 1 column
    matrix3 = tf.constant([[1., 3.]],name ='matrix1')
    product = tf.matmul(matrix1, matrix2,name='product')
    product = tf.matmul(product, matrix3,name='product_new')

sess = tf.Session()

# 定义一个summary, 其中指定相关文件存储路径
writer = tf.summary.FileWriter("logs/add", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)
