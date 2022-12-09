
# coding: utf-8



# load libraries toimport and handle data
import pickle





if __name__ == '__main__':
	
	
    meanshiftscore=[]
    twomeansscore=[]
    kmeansscore=[]
    meanshifttime=[]
    twomeanstime=[]
    kmeanstime=[]
    scannum=[]
	
    resultout=(scannum,meanshiftscore,twomeansscore,kmeansscore,meanshifttime,twomeanstime,kmeanstime)
    pickle.dump(resultout, open("windresult.txt", "wb"))
	