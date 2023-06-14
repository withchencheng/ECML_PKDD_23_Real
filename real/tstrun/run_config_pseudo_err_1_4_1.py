from runbg_sst import HyperParam
import os
import sys
home_dir= os.path.expanduser( '~/' ) 
sys.path.append(home_dir+'epro/intent_flow')
device='cuda:0'


logdir =home_dir+'data/AL/experiment/'+'tst20/'
spidir =home_dir+'data/AL/experiment/'+'spi20/' # dir for sample info analysis

K=2
hp_agnews=HyperParam(task='agnews',seeds=[0,1], total_ntrain=1300, warm_size=100, rounds=8, ncentroids=[60,40,22,50, 100*K, 150*K, 200*K, 350*K, 500*K ] ,unctth=[])
hp_agnews1=HyperParam(task='agnews',seeds=[0,1], total_ntrain=1300, warm_size=100, rounds=8, ncentroids=[300, 500, 600,  800, 900] ,unctth=[])# 100的整数倍
K=2
hp_pubmed=HyperParam(task='pubmed',seeds=[0,1], total_ntrain=1500, warm_size=200, rounds=8, ncentroids=[20,25,30,35,100*K, 200*K, 350*K, 500*K] , unctth=[])
K=25
hp_dbpedia=HyperParam(task='dbpedia',seeds=[0,1], total_ntrain=600, warm_size=70, rounds=8, ncentroids=[14, 18, 20, 25, 30, 35,  100*K, 150*K, 200*K, 350*K, 500*K], unctth=[])
hp_qqp=HyperParam(task='qqp',seeds=[0,1], total_ntrain=6000, warm_size=400, rounds=8, ncentroids=[30,40,80, 100, 200, 350, 500] ,unctth=[])
hp_qqp1=HyperParam(task='qqp',seeds=[0,1], total_ntrain=8400, warm_size=400, rounds=8, ncentroids=[30, 80, 100, 500, 800, 1000] ,unctth=[]) # for unlabeled pool=100k
hp_sst=HyperParam(task='sst-2',seeds=[0,1], total_ntrain=800, warm_size=70, rounds=8, ncentroids=[49, 48, 350, 500], unctth=[]) # 超大的ncentroid 仅适用于pseudo_err_1.3

hp_snips=HyperParam(task='snips',seeds=[0,1], total_ntrain=700, warm_size=80, rounds=8, ncentroids=[50, 60, 80] ,unctth=[]) #70, 140, 200, 280, 560, 600

hp_atis=HyperParam(task='atis',seeds=[0,1], total_ntrain=700, warm_size=80, rounds=8, ncentroids=[50, 55, 60, 65, 70,75,80] ,unctth=[]) #16, 32, 48, 64

# hp_trec=HyperParam(task='trec',seeds=[0,1], total_ntrain=500, warm_size=100, rounds=8, ncentroids=[12, 24, 30, 36, 42, 48] ,unctth=[])
hp_trec = HyperParam(task='trec',seeds=[0,1], total_ntrain=400, warm_size=100, rounds=8, ncentroids=[12, 24, 30, 36, 42, 48] ,unctth=[])

# dynamically add ncentroids
hp_stov = HyperParam(task='stov',seeds=[0,1], total_ntrain=900, warm_size=200, rounds=8, ncentroids=[15, 20, 30, 40, 50, 60] ,unctth=[])