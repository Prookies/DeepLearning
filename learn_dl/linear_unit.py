from perceptron import Perceptron
import matplotlib.pyplot as plt


class LinearUnit(Perceptron):
    def __init__(self, input_num, activator):
        '''
        初始化线性单元，设置输入参数的个数
        '''
        super(LinearUnit, self).__init__(input_num, activator)
        # super().__init__(input_num, activator)也可以


f = lambda x: x

def get_training_dataset():
    '''
    捏造5个人的收入数据
    '''
    # 构建训练数据
    # 输入向量列表
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels

def train_linear_unit():
    '''
    使用数据训练线性单元
    '''
    # 创建感知器，输入参数的特征数为1（工作年限）
    lu = LinearUnit(1, f)
    # 训练，迭代10次，学习速率为0.01
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    # 返回训练好的线性单元
    return lu

def plot(linear_uint):
    input_vecs, labels = get_training_dataset()
    fig = plt.figure() # 新建figure对象
    ax = fig.add_subplot(111) # 新增子图
    ax.scatter(list(map(lambda x: x[0], input_vecs)), labels) # 画散点图
    weights = linear_uint.weights
    bias = linear_uint.bias
    x = range(0,12,1)
    y = list(map(lambda x: weights[0]*x+bias, x))
    ax.plot(x,y) # 曲线图
    plt.show()


if __name__ == '__main__':
    '''
    训练线性单元
    '''
    linear_uint = train_linear_unit()
    # 打印训练获得的权重
    print(linear_uint)
    # 测试
    print('Work 3.4 years, monthly salary = %.2f' % linear_uint.predict([3.4]))
    plot(linear_uint)

