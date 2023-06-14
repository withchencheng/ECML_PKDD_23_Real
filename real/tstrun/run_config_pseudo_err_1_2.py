from runbg_sst import HyperParam
import os
import sys
home_dir= os.path.expanduser( '~/' ) 
sys.path.append(home_dir+'epro/intent_flow')
device='cuda:0'


# Save sample info更全的 pseudo_er1.2 放在 tst18 spi18
logdir =home_dir+'data/AL/experiment/'+'tst18/'
spidir =home_dir+'data/AL/experiment/'+'spi18/' # dir for sample info analysis

hp_agnews=HyperParam(task='agnews',seeds=[0,1], total_ntrain=1300, warm_size=100, rounds=8, ncentroids=[30,44,55,56,58,62,64,66,70] ,unctth=[]) #37, 22, 24, 40, 50, 60
hp_dbpedia=HyperParam(task='dbpedia',seeds=[0,1], total_ntrain=600, warm_size=70, rounds=8, ncentroids=[14,15, 18, 20, 25, 30, 35], unctth=[])
hp_pubmed=HyperParam(task='pubmed',seeds=[0,1], total_ntrain=1500, warm_size=200, rounds=8, ncentroids=[20,25,30,35] , unctth=[])
hp_qqp=HyperParam(task='qqp',seeds=[0,1], total_ntrain=6000, warm_size=400, rounds=8, ncentroids=[30,50,40,60,80] ,unctth=[])
hp_sst=HyperParam(task='sst-2',seeds=[0,1], total_ntrain=800, warm_size=70, rounds=8, ncentroids=[48, 51, 46, 49, 40, 10], unctth=[])

