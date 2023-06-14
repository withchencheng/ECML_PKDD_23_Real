from runbg_sst import HyperParam
import os
import sys
home_dir= os.path.expanduser( '~/' ) 
sys.path.append(home_dir+'epro/intent_flow')
device='cuda:0'


logdir =home_dir+'data/AL/experiment/'+'tst18/'
spidir =home_dir+'data/AL/experiment/'+'spi18/' # dir for sample info analysis

K=2
hp_agnews=HyperParam(task='agnews',seeds=[0,1], total_ntrain=1300, warm_size=100, rounds=8, ncentroids=[60,40,22,50, 100*K, 150*K, 200*K, 350*K, 500*K ] ,unctth=[])
K=2
hp_pubmed=HyperParam(task='pubmed',seeds=[0,1], total_ntrain=1500, warm_size=200, rounds=8, ncentroids=[20,25,30,35,100*K, 200*K, 350*K, 500*K] , unctth=[])
K=25
hp_dbpedia=HyperParam(task='dbpedia',seeds=[0,1], total_ntrain=600, warm_size=70, rounds=8, ncentroids=[14, 18, 20, 25, 30, 35,  100*K, 150*K, 200*K, 350*K, 500*K], unctth=[])
hp_qqp=HyperParam(task='qqp',seeds=[0,1], total_ntrain=6000, warm_size=400, rounds=8, ncentroids=[30,50,40,60,80, 100, 150, 200, 350, 500] ,unctth=[])
hp_sst=HyperParam(task='sst-2',seeds=[0,1], total_ntrain=800, warm_size=70, rounds=8, ncentroids=[49, 48, 350, 500, 200, 8, 100, 40], unctth=[]) # 超大的ncentroid 仅适用于pseudo_err_1.3


