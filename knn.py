import numpy as np

import sys, os
sys.path.append(os.pardir)
#import at the parent directory

import numpy as np
from dataset.mnist import load_mnist
#import _ load mnist data

from PIL import Image
#python image processing library

class KNN:
    dist_list = []

#   make a TestDataList, TrainDataList by loading MNIST
#   x_train : input(image)    t_train : real_label      all size : 60000
#   x_test : test(image)      t_test : real_label       all size : 10000
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=False)
    label_name=[0,1,2,3,4,5,6,7,8,9]

#   convert data_type uint->int
    x_train = np.int_(x_train)
    x_test = np.int_(x_test)

#   init hand_craft array
    h_train = np.array([])
    h_test = np.array([])


    #   init K, the number of test data
    def __init__(self, Knum, Num):
        self.K = Knum
        self.N = Num
#       get N random_index
        self.rand_idx = np.random.randint(0, self.x_test.shape[0], Num)
        tmp_x_test=[]
        tmp_t_test=[]
#       get the rand_idx test data
        for i in range(Num):
            tmp_x_test.append(self.x_test[self.rand_idx[i]])
            tmp_t_test.append(self.t_test[self.rand_idx[i]])
        self.x_test=np.array(tmp_x_test)
        self.t_test=np.array(tmp_t_test)
#       convert list to numpy array
        self.tmp_train = np.array(self.x_train)
        self.tmp_test = np.array(self.x_test)
#       hand craft MNIST data
        self.handcraft()


    def handcraft(self):
#       hand craft the test data : h_test
        main_arr=[]
        for img in self.tmp_test:
            img = img.reshape(28, 28)
            sub_arr = []
            for i in range(0, 28, 4):
                for j in range(0, 28, 4):
                    tmp = img[i:i + 4, j:j + 4]
                    sub_arr.append(np.sum(tmp))
            main_arr.append(sub_arr)
        self.h_test=np.array(main_arr)

#       hand craft the train data : h_train
        main_arr = []
        for img in self.tmp_train:
            img = img.reshape(28, 28)
            sub_arr = []
            for i in range(0, 28, 4):
                for j in range(0, 28, 4):
                    tmp = img[i:i + 4, j:j + 4]
                    sub_arr.append(np.sum(tmp))
            main_arr.append(sub_arr)
        self.h_train=np.array(main_arr)

#   calculate between each test data and each train data by using euclidean distance
    def Euc_dist(self,a,b):
        return np.sqrt(np.sum((a-b) ** 2))

#   nearer one have more influence : calculate weight
    def D_weight(self,d):
        return 1/(1+d)

#   calculate a distance between every x_test and x_train
    def Cal_dist(self):
        # initialize dist_list
        self.dist_list = []
        for i in self.x_test:
            # initialize dist
            dist = []
            for j in self.x_train:
                dist.append(self.Euc_dist(i,j))
            self.dist_list.append(dist)

#   weighted majority vote
    def Weight_vote(self):
        self.Cal_dist()
        perc=0
        for i in range(self.x_test.shape[0]):
            result = {x: y for x, y in zip(self.dist_list[i], self.t_train)}
            # key:거리를 기준으로 sort
            sort_result = sorted(result.items(), key=lambda item: item[0])[:self.K]
            w_cnt = [0,0,0,0,0,0,0,0,0,0]
            for j in range(self.K):
                w_cnt[sort_result[j][1]] += self.D_weight(sort_result[j][0])
            if(self.t_test[i]==self.label_name[w_cnt.index(max(w_cnt))]):
                perc+=1
        return perc / self.h_test.shape[0]

#   calculate a distance between every h_test and h_train
    def H_Cal_dist(self):
        # initialize dist_list
        self.dist_list = []
        for i in self.h_test:
            # initialize dist
            dist = []
            for j in self.h_train:
                dist.append(self.Euc_dist(i, j))
            self.dist_list.append(dist)

#   weighted majority vote
    def H_Weight_vote(self):
        self.H_Cal_dist()
        perc=0
        for i in range(self.h_test.shape[0]):
            result = {x: y for x, y in zip(self.dist_list[i], self.t_train)}
            # key:거리를 기준으로 sort
            sort_result = sorted(result.items(), key=lambda item: item[0])[:self.K]
            w_cnt = [0,0,0,0,0,0,0,0,0,0]
            for j in range(self.K):
               w_cnt[sort_result[j][1]] += self.D_weight(sort_result[j][0])
            print(self.rand_idx[i],"th data","\t result:", w_cnt.index(max(w_cnt)),"\t \t label: ", self.t_test[i])
            if(self.t_test[i]==self.label_name[w_cnt.index(max(w_cnt))]):
                perc+=1
        return perc/self.h_test.shape[0]