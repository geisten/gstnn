import matplotlib.pyplot as plt
import csv

x = []
y = []
i=0
with open('error.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(i)
        y.append(float(row[2]))
        i+=1

plt.plot(x,y, label='Error per training input')
#plt.yscale('log')
plt.xscale('log')
plt.xlabel('test run')
plt.ylabel('error')
plt.title('Error during training of the weights\nError rate should decrease with increasing number of training data')
plt.legend()
plt.show()