import sys, math
import numpy as np

def load_labels(filename):
    labels = {}
    for line in open(filename).readlines():
        items = filter(None, line.split(','))
        labels[items[0]] = int(items[1])
    return labels

if __name__ == '__main__':
    filename = sys.argv[1]
    # print(filename)
    #
    # errors = []
    # for line in open(filename).readlines():
    #     if line.startswith('original evaluation errors'):
    #         items = line[len('original evaluation errors: ['):-2].split(', ')
    #         errors.append([float(x) for x in items])
    #
    # errors = np.asarray(errors)
    # channel_num = errors.shape[1]
    #
    # for i in range(channel_num):
    #     nums = errors[:,i]
    #     mean = np.mean(nums)
    #     std = np.std(nums)
    #     print('%g +- %g' % (mean, std/math.sqrt(len(nums))))

    label_true = load_labels('/scr/zhangyuc/randomized-defense/datasets/labels.csv')
    label_predict = load_labels(filename)

    count = 0
    correct = 0
    for item in label_predict.items():
        count += 1
        if label_true[item[0][:-4]] == item[1]:
            correct += 1

    print (float(correct)/count)