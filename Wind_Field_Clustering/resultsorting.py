
# coding: utf-8



# load libraries toimport and handle data
import pickle
import numpy as np





if __name__ == '__main__':
	
    (scannum,meanshiftscore,twomeansscore,kmeansscore,meanshifttime,twomeanstime,kmeanstime)=pickle.load(open("windresult.txt", "rb"))
    print(scannum) 
    print(meanshiftscore) 
    print(twomeansscore) 
    print(kmeansscore) 
    print(meanshifttime) 
    print(twomeanstime) 
    print(kmeanstime) 
	
#	(meanshiftout,twomeansout,kmeansout)=pickle.load(open("windresult.txt", "rb"))
#	meanshiftarray = np.asarray(meanshiftout)
#	twomeansarray = np.asarray(twomeansout)
#	kmeansarray = np.asarray(kmeansout)
#	
#
#
#	meanshiftsorted=np.sort(meanshiftarray.view('float,float,float'),order=['f1'],axis=0)
#	twomeanssorted=np.sort(twomeansarray.view('float,float,float'),order=['f1'],axis=0)
#	kmeanssorted=np.sort(kmeansarray.view('float,float,float'),order=['f1'],axis=0)
#	
#	print(meanshiftsorted)
#	print(twomeanssorted)
#	print(kmeanssorted)

	


	