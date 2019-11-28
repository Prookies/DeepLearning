import struct
import os
from blob.bp import *
from datetime import datetime

# 数据加载器基类
class Loader():
    def __init__(self, path, count):
        '''
        初始化加载器
        path: 数据文件路径
        count: 文件中的样本个数
        '''
        up_path = os.path.abspath('..')
        print(up_path+"\\mnist\\"+path)
        self.path = os.path.join(up_path + '\\mnist\\' + path)
        self.count = count
    
    def get_file_content(self):
        '''
        读取文件内容
        '''
        f = open(self.path, 'rb')
        # read()函数可以读取整个文件中的内容
        content = f.read()
        f.close()
        return content

    def to_int(self, byte):
        '''
        将unsigned byte字符转换为整数
        '''
        print(struct.unpack('B', byte))
        return struct.unpack('B', byte)

class ImageLoader(Loader):
    def get_picture(self, content, index):
        '''
        内部函数，从文件中获取图像
        '''
        # 为什么+16，因为数据集头部有4个integer类型，
        start = index*28*28 + 16
        # 使用了一种非常传统的方法来读取数据
        # 其把picture当作二维数组，然后遍历28*28的数据
        picture = []
        for i in range(28):
            # 添加一行
            picture.append([])
            for j in range(28):
                # print(type(content[start + i*28 + j]))
                picture[i].append(
                    content[start + i*28 +j]
                )
        return picture
    
    def get_one_sample(self, picture):
        '''
        内部函数，将图像转化为样本的输入向量
        该函数可以与get_picture()函数结合，直接得到一个样本
        '''
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample
    
    def load(self):
        '''
        加载数据文件，获得全部样本的输入向量
        '''
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):
            data_set.append(
                self.get_one_sample(
                    self.get_picture(content, index)
                )
            )
        return data_set
    

# 标签数据加载器
class LabelLoader(Loader):
    def load(self):
        '''
        加载数据文件，获得全部样本的标签向量
        '''
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index + 8]))
        return labels
    
    def norm(self, label):
        '''
        内部函数，将一个值转换为10维标签向量
        '''
        # print("label=" + str(label))
        label_vec = []
        # label_value = self.to_int(label)
        for i in range(10):
            if i == label:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec
    
def get_training_data_set():
    '''
    获得训练数据集
    '''
    image_loader = ImageLoader('train-images.idx3-ubyte', 60000)
    label_loader = LabelLoader('train-labels.idx1-ubyte', 60000)
    return image_loader.load(), label_loader.load()

def get_test_data_set():
    '''
    获得测试数据集
    '''
    image_loader = ImageLoader('t10k-images.idx3-ubyte', 10000)
    label_loader = LabelLoader('t10k-labels.idx1-ubyte', 10000)
    return image_loader.load(), label_loader.load()

def get_result(vec):
    '''
    获得实际的输出结果
    '''
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index

def evaluate(network, test_data_set, test_labels):
    '''
    使用错误率对网络进行评估
    '''
    error = 0
    total = len(test_data_set)

    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i]))
        if label != predict:
            error += 1
    return float(error) / float(total)

def now():
    return datetime.now().strftime('%c')

def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    train_data_set, train_labels = get_training_data_set()
    test_data_set, test_labels = get_test_data_set()
    network = NetWork([784, 300, 10])
    print("开始训练")
    while True:
        epoch += 1
        network.train(train_labels, train_data_set, 0.01, 1)
        print('%s epoch %d finished' % (now(), epoch))
        if epoch%10 == 0:
            error_ratio = evaluate(network, test_data_set, test_labels)
            print('%s after epoch %d, error ratio is %f' % (now(), epoch, error_ratio))
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio

if __name__ == '__main__':
    train_and_evaluate()