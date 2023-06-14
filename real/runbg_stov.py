"""
Author: Chen Cheng, 20221107


20221118: for AL roberta
20221222: Prob suitable range of {warm up size, AL step size, #AL rounds} for AL methods
"""

from dataclasses import dataclass
import psutil
import subprocess
from time import sleep
from datetime import datetime
import os
from runbg_sst import Config, LaunchedProcs, all_finished, task2seqlen, HyperParam
from tstrun.run_config_pseudo_err_1_4_1 import hp_stov

    
def main():
    

    # oldpids=[772264]
    # print('monitoring .....', oldpids) #!!!!
    # while(True):
    #     sleep(5)
    #     if all_finished(oldpids):
    #         break


    pid = os.getpid()
    print('runbg pid:', pid)
    home_dir= os.path.expanduser( '~/' ) 
    # 占用一块GPU的实验

    logdir =home_dir+'data/AL/experiment/'+'tst20/'
    spidir =home_dir+'data/AL/experiment/'+'spi20/' # dir for sample info analysis
    config_pool=[]

    device='cuda:1'
    gpu_budget=2 # sst2: 最大用量 sst-2: 4521MiB   agnews: 9109MiB  pubmed: 8839


    # hp_stov1 = HyperParam(task='stov',seeds=[0,1], total_ntrain=1000, warm_size=200, rounds=8, ncentroids=[] ,unctth=[])
    # hp_stov1 = HyperParam(task='stov',seeds=[0,1], total_ntrain=7979, warm_size=7979, rounds=8, ncentroids=[] ,unctth=[])
    # hp_stov2 = HyperParam(task='stov',seeds=[0,1], total_ntrain=800, warm_size=180, rounds=8, ncentroids=[] ,unctth=[])
    # hp_stov3 = HyperParam(task='stov',seeds=[0,1], total_ntrain=900, warm_size=180, rounds=8, ncentroids=[] ,unctth=[])
    hp = hp_stov
    # ---------------------------------------------- tst17 ---------------------------------------------------
    al_methods = ['entropy',  'emnlp21cal', 'naacl22actune', 'iclr20badge', 'plm_KM', 'random',  'icml17bald',] #   'emnlp20alps'
    for al_method in al_methods:
        for seed in hp.seeds:
            config=Config(hp.warm_size, hp.total_ntrain, hp.rounds, home_dir, logdir, spidir, hp.task, seed+2, device, al_method, ncentroids=None, unctth=-1)
            config_pool.append(config)
    al_methods = ['pseudo_err_1.4', 'pseudo_err_1.4.1', 'pseudo_err_1.4.2'] #
    for al_method in al_methods:
        for nc in hp.ncentroids:
            for seed in hp.seeds:
                config=Config(hp.warm_size, hp.total_ntrain, hp.rounds, home_dir, logdir, spidir, hp.task, seed+2, device, al_method, ncentroids=nc, unctth=-1)
                config_pool.append(config)
    # ---------------------------------------------- tst17 ---------------------------------------------------

    cursor=0 # try to move  cursor in config_pool is there's more space
    launchedp=LaunchedProcs()
    all_launched=False
    while True:
        # try to keep running procs in [1,4]
        # 并且尽量接近4

        if cursor<len(config_pool) and launchedp.nrunning()<gpu_budget:
            #launch 1 and move cursor
            config=config_pool[cursor]
            p=launchedp.run1(config)
            dt_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            print(f'{dt_string}: Launching {cursor+1}-th/{len(config_pool)} config, Pid={p.pid}: {config.logfname()}')
            cursor +=1

        if cursor>=len(config_pool) and not all_launched: # 只打印一次
            print(f"All {len(config_pool)} processes launched!!")
            print('Keep monitoring...')
            all_launched=True
        # then keep running for monitoring
        
        if launchedp.nfinished()>=len(config_pool):
            print(f"All {len(config_pool)} processes finished!!")
            return

        sleep(4) # 每隔4秒 check是不是有空位，有空位就挪一下 cursor
    

    

if __name__ == '__main__':
    main()

