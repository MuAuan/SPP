import argparse
import re
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('log', type=str)
args = parser.parse_args()

#ptn = re.compile(r'iteration = ([0-9]+), loss = ([0-9.]+), accuracy = ([0-9.]+)')
#ptn = re.compile(r'epoch = ([0-9]+), loss = ([0-9.]+), acc = ([0-9.]+), val_loss = ([0-9.]+), val_acc = ([0-9.]+)')
#ptn = re.compile(r'loss:([0-9.]+)- acc:([0-9.]+) - val_loss:([0-9.]+)- val_acc: ([0-9.]+)')
ptn = re.compile(r'([0-9]+)	([0-9.]+)	([0-9.]+)	([0-9.]+)	([0-9.]+)')
# loss: 1.8175 - acc: 0.3631 - val_loss: 1.1386 - val_acc: 0.6179
#epoch	loss	acc	val_loss	val_acc

iteration_list = []
loss_list = []
acc_list = []
val_loss_list = []
val_acc_list = []
s=0
for line in open(args.log, 'r'):
    m = ptn.search(line)
    if m:
        iteration_list.append(int(m.group(1)))
        #iteration_list.append(int(s))
        loss_list.append(float(m.group(2)))
        acc_list.append(float(m.group(3)))
        val_loss_list.append(float(m.group(4)))
        val_acc_list.append(float(m.group(5)))
    #print(iteration_list)
    s += 1
fig, ax1 = plt.subplots()
p1, = ax1.plot(iteration_list, loss_list, 'b', label='loss')
ax1.set_xlabel('iterations')
#ax2=ax1.twinx()
p2, = ax1.plot(iteration_list, val_loss_list, 'g', label='val_loss')
ax1.legend(handles=[p1, p2])

fig, ax3 = plt.subplots()
p3, = ax3.plot(iteration_list, acc_list, 'b', label='acc')
ax3.set_xlabel('iterations')
#ax4=ax3.twinx()
p4, = ax3.plot(iteration_list, val_acc_list, 'g', label='val_acc')
ax3.legend(handles=[p3, p4])

plt.show()
