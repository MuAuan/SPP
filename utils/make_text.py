import re
import os
import glob
from PIL import Image
import time
            
files = glob.glob('./history/history/*.txt')
print(files)
ptn = re.compile(r'([0-9]+)	([0-9.]+)	([0-9.]+)	([0-9.]+)	([0-9.]+)')

iteration_list = []
loss_list = []
acc_list = []
val_loss_list = []
val_acc_list = []
s=0
result_file='./history/result.txt'
for i in range(len(files)):
    for line in open(files[i], 'r'):   #files:
        print(line)
        m = ptn.search(line)
        print('m=',m)
        if m:
            iteration_list.append(int(m.group(1)))
            #iteration_list.append(int(s))
            loss_list.append(float(m.group(2)))
            acc_list.append(float(m.group(3)))
            val_loss_list.append(float(m.group(4)))
            val_acc_list.append(float(m.group(5)))
            print(len(val_acc_list))
with open(result_file, "w") as fp:
    fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
    for i in range(len(iteration_list)):
        fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss_list[i], acc_list[i], val_loss_list[i], val_acc_list[i]))    
