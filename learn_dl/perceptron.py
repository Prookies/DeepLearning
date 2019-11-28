from functools import reduce

class Perceptron():
    def __init__(self, input_num, activator):
        super().__init__()
        '''
        初始化感知器，设置输入参数的个数，以及激活函数。
        激活函数的类型为double->double
        '''
        self.activator = activator
        # 权重向量初始化为0 语法：列表解析
        self.weights = [0.0 for i in range(input_num)]
        # 偏置初始化为0
        self.bias = 0.0

    def __str__(self):
        '''
        打印学习到的权重、偏置项
        '''
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)
    
    def predict(self, input_vec):
        '''
        输入向量，输出感知器的计算结果
        '''
        return self.activator(
            reduce(lambda a, b: a+b,
            list(map(lambda x, w: x*w, input_vec, self.weights))) 
            + self.bias)
    
    def train(self, input_vecs, labels, iteration, rate):
        '''
        输入训练数据：一组向量，与每个向量对应的label；
        以及迭代次数，学习率
        '''
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)
    
    def _one_iteration(self, input_vecs, labels, rate):
        '''
        一次迭代，使用所有的数据
        可以使用批量迭代的方法
        '''
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        # 而每个训练样本是(input_vec, label)
        samples = list(zip(input_vecs, labels))
        # 对每个训练样本，按照感知器规则更新权重
        for (input_vec, label) in samples:
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        '''
        按照感知器规则更新权重
        '''
        # 把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]
        # 更新权重
        delta = label - output
        self.weights = list(map(lambda x, w: w + rate*delta*x,
        input_vec, self.weights))
        # 更新bias
        self.bias += rate*delta

def f(x):
    '''
    定义激活函数
    '''
    return 1 if x > 0 else 0

def get_training_dataset():
    '''
    基于and真值表构建训练数据
    '''
    # 构建训练数据
    # 输入向量列表
    input_vecs = [[1,1], [0,0], [1,0], [0,1]]
    # 期望的输出列表
    labels = [1, 0, 0, 0]
    return input_vecs, labels

def train_and_perceptron():
    '''
    使用and真值表训练感知器
    '''
    # 创建感知器，输入参数个数为2，激活函数为f
    p = Perceptron(2, f)
    # 训练，迭代10次，学习速率为0.1
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p

if __name__ == '__main__':
    # 训练and感知器
    and_perception = train_and_perceptron()
    print(and_perception)
    # 测试
    print('1 and 1 = %d' % and_perception.predict([1,1]))






# 学习使用list(),zip(),map(),reduce()函数
# input_vec = [0, 1]
# weights = [0.5, 0.5]
# bias = -0.8
# pair_input_weight = list(zip(input_vec, weights))
# print(pair_input_weight)
# products = list(map(lambda x: x[0]*x[1], list(zip(input_vec, weights))))
# print(products)
# sum_product = reduce(lambda a, b: a+b, products) + bias
# print(sum_product)
# print(f(sum_product))
