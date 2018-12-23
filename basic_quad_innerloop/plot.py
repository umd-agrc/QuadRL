import os 
import glob
import numpy as np
import matplotlib.pyplot as plt

def main():
    fig_pos=[]
    ax_pos=[]
    fig_eul=[]
    ax_eul=[]
    for file in glob.glob('csv/pos_error*.csv'):
        if file == 'csv/pos_error1.csv':
            data = np.genfromtxt(file,delimiter=',')
            f,a = plt.subplots(1,3)
            x = np.arange(data.shape[0])
            a[0].scatter(x,data[:,0])
            a[1].scatter(x,data[:,1])
            a[2].scatter(x,data[:,2])
            fig_pos.append(f)
            ax_pos.append(a)

    for file in glob.glob('csv/euler_error*.csv'):
        if file == 'csv/euler_error1.csv':
            data = np.genfromtxt(file,delimiter=',')
            f,a = plt.subplots(1,3)
            x = np.arange(data.shape[0])
            a[0].scatter(x,data[:,0])
            a[1].scatter(x,data[:,1])
            a[2].scatter(x,data[:,2])
            fig_eul.append(f)
            ax_eul.append(a)

    plt.show()

if __name__ == '__main__':
    main()
