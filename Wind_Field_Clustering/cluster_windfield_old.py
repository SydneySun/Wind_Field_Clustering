
# coding: utf-8



# load libraries toimport and handle data
import pickle
import sys
import scipy.io as sio # to import matlab files
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale # to normalize variables
from sklearn.cross_validation import train_test_split # to split data in random groups
from sklearn import cluster # set of algorithms to cluster
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from numpy import mean,nanmin,nanmax,nanmean
import time



# In[1]:

def cluster_windfield(scannumber,
                      data,
                      mat_filename="",
                      cluster_method="meanshift",
                      plot_original=False,
                      train=0.5,
                      test=0.5,
                      r_state=13):
    datatime = data['time']
    xx = data['xx']
    yy = data['yy']
    uu_val = data['uu_val']
    vel_array = uu_val[:,:,scannumber]
    
    x = list()
    y = list()
    v = list()
    # create masked arrays to mark invalid 'nan' values in list "v"
    # first create array with original list, then a masked version of it, using a boolean detection of 'nan' to indicate masked values
    x_array = np.array(x)
    [rows, cols] = xx.shape
    for r in range(rows):
        for c in range(cols):
            x.append(xx[r, c])
            y.append(yy[r, c])
            v.append(vel_array[r, c])
    x_array = np.array(x)        
    x_mask = np.ma.masked_array(x_array, np.isnan(x_array))
    y_array = np.array(y)
    y_mask = np.ma.masked_array(y_array, np.isnan(y_array))
    v_array = np.array(v)
    v_mask = np.ma.masked_array(v_array, np.isnan(v_array))

    # create new "clean" arrays with non-masked version of masked array (x_mask, y_mask, v_mask) for its later use
    v_clean = v_mask[~v_mask.mask]
    x_clean = x_mask[~v_mask.mask]
    y_clean = y_mask[~v_mask.mask]

    # Data Mining
    ## Pre-processing

    # define features (transpose): coordinates and velocities
    # define targets: velocities
    # use "clean" arrays (x_mask, y_mask, v_mask), without "nan"
    coordinates = np.vstack((x_clean, y_clean))
    coordinates = coordinates.T
    targets = v_clean
    features = np.vstack((x_clean, y_clean, v_clean))
    features = features.T

    # normalize features around their individual mean, with unit variance
    features = scale(features)
    
    if plot_original:
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.scatter(features[:, 0], features[:, 1], c=features[:, 2], marker='.', s=1, linewidths=0,cmap='viridis')
        plt.grid()
        plt.title('Original {i}'.format(i=scannumber))
        plt.savefig('./plot/Velocity/cluster_{i}.png'.format(i=scannumber), format='png')

    # divide data into equal and random training and test data sets
    X_train, X_test, cord_train, cord_test, y_train, y_test = train_test_split(features, coordinates, targets,
                                                        test_size= test, train_size = train, random_state=r_state)
    
    cluster_number=[];
    metric=0;
    if cluster_method=="meanshift":
        bandwidth = cluster.estimate_bandwidth(X_train, quantile=0.15, n_samples=300)
        ms = cluster.MeanShift(bandwidth = bandwidth,bin_seeding=True)
        ms.fit(X_train)
        cluster_number= ms.predict(X_test)
        labels_ms = ms.labels_
        _,label_test,_,data_test = train_test_split(labels_ms,X_train,test_size= 0.1, random_state=r_state)
        print(label_test.shape)
        print(data_test.shape)
        metric = metrics.silhouette_score(data_test,label_test,metric='euclidean')
    

    if cluster_method=="twomeans":
        two_means = cluster.MiniBatchKMeans(n_clusters=3)
        two_means.fit(X_train)
        cluster_number = two_means.predict(X_test)
        labels_ms = two_means.labels_
        _,label_test,_,data_test = train_test_split(labels_ms,X_train,test_size= 0.1, random_state=r_state)
        metric = metrics.silhouette_score(data_test,label_test,metric='euclidean')


    if cluster_method=="kmeans":
        kmeans = cluster.KMeans(init='k-means++', n_clusters=3, random_state=r_state)
        kmeans.fit(X_train)
        cluster_number = kmeans.predict(X_test)
        labels_ms = kmeans.labels_
        _,label_test,_,data_test = train_test_split(labels_ms,X_train,test_size= 0.1, random_state=r_state)
        metric = metrics.silhouette_score(data_test,label_test,metric='euclidean')
    
    if mat_filename:
        strtime=time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        mat_path = "{}_{}_{}.mat".format(mat_filename, scannumber,strtime)
        sio.savemat(mat_path,{'test_data':X_test,'cluster_number':cluster_number,'sil_score':metric})
    return (X_test,cluster_number,metric)


# In[2]:


if __name__ == '__main__':
	# In[3]:
	
    filename = './WakeIdentification/Data/singlescan_ppi_data20150228T040944.mat'
    data = sio.loadmat(filename)
    scanfrom=int(sys.argv[1])
    scanto=scanfrom+1
    (scannum,meanshiftscore,twomeansscore,kmeansscore,meanshifttime,twomeanstime,kmeanstime)=pickle.load(open("windresult.txt", "rb"))
    print('Begining processing layer',scanfrom)
    for scannumber in np.arange(scanfrom,scanto):
        scannum.append(scannumber)
        
        Tstart=time.clock()
        [X_test,cluster_number,metric]=cluster_windfield(scannumber,data,"","meanshift")
        elapsed = (time.clock() - Tstart)
        meanshifttime.append(elapsed)
        meanshiftscore.append(metric)
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.scatter(X_test[:, 0], X_test[:, 1], c=cluster_number, marker='.', s=1, linewidths=0,cmap='viridis')
        plt.grid()
        plt.title('Meanshift {i}'.format(i=scannumber))
        plt.savefig('./plot/Meanshift/cluster_{i}.png'.format(i=scannumber), format='png')
        
        Tstart=time.clock()
        [X_test,cluster_number,metric]=cluster_windfield(scannumber,data,"","twomeans")
        elapsed = (time.clock() - Tstart)
        twomeanstime.append(elapsed)
        twomeansscore.append(metric)
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.scatter(X_test[:, 0], X_test[:, 1], c=cluster_number, marker='.', s=1, linewidths=0,cmap='viridis')
        plt.grid()
        plt.title('twomeans {i}'.format(i=scannumber))
        plt.savefig('./plot/Twomeans/cluster_{i}.png'.format(i=scannumber), format='png')
        
        Tstart=time.clock()
        [X_test,cluster_number,metric]=cluster_windfield(scannumber,data,"","kmeans",True)
        elapsed = (time.clock() - Tstart)
        kmeanstime.append(elapsed)
        kmeansscore.append(metric)
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.scatter(X_test[:, 0], X_test[:, 1], c=cluster_number, marker='.', s=1, linewidths=0,cmap='viridis')
        plt.grid()
        plt.title('kmeans {i}'.format(i=scannumber))
        plt.savefig('./plot/Kmeans/cluster_{i}.png'.format(i=scannumber), format='png')
    resultout=(scannum,meanshiftscore,twomeansscore,kmeansscore,meanshifttime,twomeanstime,kmeanstime)
    pickle.dump(resultout, open("windresult.txt", "wb"))
	


