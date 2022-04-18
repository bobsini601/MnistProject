import knn
import time

# using time lib for calculating KNN time

# set K, N
K=[3,5,10]
N=[100,500,1000]

for i in range(3):
    for j in range(3):
        result=knn.KNN(K[i],N[j])
        print("K=",K[i],"N=",N[j])
        print("===========Origin=============")
        O_start=time.time()
        O_acc=result.Weight_vote()
        O_time=time.time()-O_start
        print("accuracy : ", O_acc)
        print("time : ",O_time)

        print("===========HandCraft==========")
        H_start=time.time()
        H_acc=result.H_Weight_vote()
        H_time=time.time()-H_start
        print("accuracy : ", H_acc)
        print("time : ",H_time)
        print("\n")
