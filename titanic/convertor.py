import numpy as np
import argparse

from sklearn.linear_model import LogisticRegression   
from sklearn.svm import SVC


def filter(splits):
    return splits[2] == '' or splits[5] == '' or splits[6] == '' or splits[7] == '' or splits[8] == ''

'''
PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S

PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
'''
def train(train):
    # Safely read the input filename using 'with'
    samples = []
    targets = []
    with open(train) as iFile:
        lines = iFile.readlines()
        for i in range(len(lines)):
            line = lines[i]
            splits = line.split(',')
            vector = []
            if i == 0 or filter(splits):
                continue
            vector.append(float(splits[2]))
            if splits[5] == 'female':
                vector.append(0)
            else:
                vector.append(1)
            vector.append(float(splits[6]))
            vector.append(float(splits[7]))
            vector.append(float(splits[8]))
            samples.append(vector)
            targets.append(int(splits[1]))

            # oFile.write('\t'.join(vector) + '\n')

    # classifier = LogisticRegression()  # 使用类，参数全是默认的  
    classifier = SVC()
    classifier.fit(samples, targets)  # 训练数据来学习，不需要返回值 
    error = 0
    pred = classifier.predict(samples)
    # print(pred)
    for i in range(len(samples)):
        # print('pred: ', pred[i], ', train: ', targets[i])
        if pred[i] != targets[i]:
            error += 1
    print('train error: ', error, '/', len(samples))
    return classifier

def test(test, classifier):
    inputs = []
    ids = []
    with open(test) as tFile:
        lines = tFile.readlines()
        for i in range(len(lines)):
            line = lines[i]
            splits = line.split(',')
            vector = []
            if i == 0:
                continue
            vector.append(float(splits[1]))
            if splits[4] == 'female':
                vector.append(0)
            else:
                vector.append(1)
            if splits[5] == '':
                vector.append(30)
            else:
                vector.append(float(splits[5]))
            if splits[6] == '':
                vector.append(0)
            else:
                vector.append(float(splits[6]))
            if splits[7] == '':
                vector.append(0)
            else:
                vector.append(float(splits[7]))
            inputs.append(vector)
            ids.append(splits[0])
    result = classifier.predict(inputs)
    rtn = [['PassengerId','Survived']] + [[str(ids[i]),str(result[i])] for i in range(len(result))]
    return rtn;

def main():
    parser = argparse.ArgumentParser(description='The args to convert raw data.')

    parser.add_argument('--train', type=str, help='file of train data')
    parser.add_argument('--test', type=str, help='file of test data')

    args = parser.parse_args()
    if args.train is None:
        parser.print_help()
        exit(1)
    classifier = train(args.train)
    # result = test(args.test, classifier)
    # for item in result:
        print(','.join(item))

if __name__ == '__main__':
  main()
