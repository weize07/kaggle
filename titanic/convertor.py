from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

import numpy as np
import argparse

'''
PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
'''
def convert(input, output):
    # Safely read the input filename using 'with'
    with open(input) as iFile, open(output, 'w') as oFile:
        lines = iFile.readlines()
        for i in range(len(lines)):
            line = lines[i]
            splits = line.split(',')
            vector = []
            vector.append(splits[2])
            if splits[5] == 'male':
                vector.append('1')
            else:
                vector.append('0')
            vector.append(splits[6])
            vector.append(splits[7])
            vector.append(splits[8])
            oFile.write('\t'.join(vector) + '\n')


def main():
    parser = argparse.ArgumentParser(description='The args to convert raw data.')

    parser.add_argument('--input', type=str, help='file of raw data')
    parser.add_argument('--output', type=str, help='output to tensors')

    args = parser.parse_args()
    if args.input is None or args.output is None:
        parser.print_help()
        exit(1)
    print(args.input, ' to ', args.output)
    convert(args.input, args.output)

if __name__ == '__main__':
  main()
