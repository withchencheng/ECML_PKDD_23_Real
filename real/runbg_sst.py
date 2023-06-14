"""
Author: Chen Cheng, 20221107


20221118: for AL roberta
20221222: Prob suitable range of {warm up size, AL step size, #AL rounds} for AL methods
20230319 : Add macro F1 metric
          The macro-averaged F1 score (or macro F1 score) is computed using the arithmetic mean (aka unweighted mean) of all the per-class F1 scores.
          
"""


from dataclasses import dataclass
from typing import List
import psutil
import subprocess
from time import sleep
from datetime import datetime
import os
from real.conferr.utils import PairInputTasks

task2seqlen={
    "sst-2": 32,
    "qqp": 32,
    "snips": 32,
    "trec": 32,
    "atis": 32,
    "stov": 32,
    "agnews":96,
    "pubmed":96,
}

NoSaveMethod=['icml17bald']


class Config:
    def __init__(self, warm_size, total_ntrain, rounds,
     home_dir, logdir, spidir, task, seed, device, al_method,
      ncentroids = None, unctth=-1.0 ) -> None:
        self.task = task
        self.al_method = al_method 
        self.data_dir = f'{home_dir}data/AL/data3/{task}/'
        self.seed = seed
        self.device = device
        if unctth>0:
            self.output_dir= f'{spidir}{task}_w{warm_size}_ttr{total_ntrain}_r{rounds}_{al_method}_s{seed}_unctth{unctth}/' # 无用的unctth已经带上了，就不管了吧
        else:
            self.output_dir= f'{spidir}{task}_w{warm_size}_ttr{total_ntrain}_r{rounds}_{al_method}_s{seed}/' # 无用的unctth已经带上了，就不管了吧
    
        self.max_seq_len= task2seqlen[task] #'sst-2' !!!BUG!!
        self.unctth = unctth
        if ncentroids:
            self.ncentroids = ncentroids
            self.output_dir= f'{spidir}{task}_w{warm_size}_ttr{total_ntrain}_r{rounds}_{al_method}_nc{ncentroids}_s{seed}/'

        self.tsb_dir =  self.output_dir + 'tsb/' # output_dir最终确定后才能谈 tsb_dir
        # if al_method not in NoSaveMethod:
            # self.save_sample = '' # store_true  # 效果调好了再c存中间结果

        # ------ constant params

        if task in PairInputTasks:
            self.add_sep_token = ''
        self.warm_size= warm_size #100
        self.rounds = rounds #10
        self.sample_labels = int((total_ntrain-warm_size)/rounds)

        self.dev_labels = 3000 
        self.num_train_epochs = 10 #10 
        self.train_batch_size =8
        self.logging_steps = 20 
        self.train_file = "train.json"
        self.dev_file = "valid.json" 
        self.test_file = "test.json" 
        self.unlabel_file = "unlabeled.json" 
        self.weight_decay = 1e-8   # double check
        self.learning_rate = 2e-5 
        self.model_type = "roberta-base" 
        if 'alps' in al_method:
            self.eval_batch_size = 320 # 256; 12255MiB
        else:
            self.eval_batch_size = 512 
        
        # prefix with _ , not args for running program
        self._logdir = logdir
        self._spidir = spidir
        self._total_ntrain = total_ntrain


    def argli(self) -> list:
        agli = []
        for k,v in self.__dict__.items():
            if k[0]=='_':continue
            agli.append(f'--{k}')
            v=str(v)
            if len(v)>0:
                agli.append(v)
        return agli

    def argdict(self) -> dict:
        # 用于中间结果分析，jupyter好用
        # 没做action store true的
        argd = {}
        for k,v in self.__dict__.items():
            if k[0]=='_':continue
            argd[k]=v
            # if self.task not in NoSaveTask:
            #     argdsave_sample = '' # store_true
        return argd

    
    def logfname(self) ->str:
        # 不是参数，不能放在class attribute里面
        if self.unctth>0:
            fname =f'{self._logdir}log_{self.task}_w{self.warm_size}_ttr{self._total_ntrain}_r{self.rounds}_{self.al_method}_u{self.unctth}_s{self.seed}'
        elif hasattr(self, 'ncentroids'):
            fname =f'{self._logdir}log_{self.task}_w{self.warm_size}_ttr{self._total_ntrain}_r{self.rounds}_{self.al_method}_nc{self.ncentroids}_s{self.seed}'
        else:
            fname =f'{self._logdir}log_{self.task}_w{self.warm_size}_ttr{self._total_ntrain}_r{self.rounds}_{self.al_method}_s{self.seed}'

        return fname
    
    def pklfname(self) ->str:
        # 不是参数，不能放在class attribute里面
        fname = f'{self._spidir}/log_tst{self.version}_ALwm{self.ALwarmup}_s{self.seed}.pkl'
        return fname
    
    def summary(self) ->str:
        """Summarize important args"""
        ret = f'seed={self.seed}, alzb={self.albz}, version={self.version}, ALwarmup={self.ALwarmup}, device={self.device}'
        return ret
    

class LaunchedProcs():
    def __init__(self) -> None:
        self.launched=[]
        self.finishedpid=set()

    def run1(self, config):
        process = subprocess.Popen(['python', '-u', 'AL/conferr/main.py']+config.argli(),
            stdout=open(config.logfname(), 'w'),
            stderr=subprocess.STDOUT
            )
        self.launched.append(process)
        return process

    def nrunning(self):
        """get number of running procs"""
        return len(self.launched) - self.nfinished()
    
    def nfinished(self):
        """get number of finished procs"""
        for i,p in enumerate(self.launched):
            if p.pid in self.finishedpid: continue # 已经结束 不用poll
            rc = p.poll() # return code
            if rc is not None:
                self.finishedpid.add(p.pid)
                dt_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                print(f'{dt_string}: Finished Pid={p.pid} rc={rc}. finished {len(self.finishedpid)}/{len(self.launched)}.')
        n= len(self.finishedpid)
        return n

def all_finished(pids):
    for pid in pids:
        if psutil.pid_exists(pid):
            return  False
    return True


@dataclass
class HyperParam:
    task: str
    seeds:list
    total_ntrain: int
    warm_size: list
    rounds: list 
    ncentroids: List[int]
    unctth: list

    
def main():
    
    # sleep(80*60)

    # oldpids=[2858697]
    # print('monitoring .....', oldpids) #!!!!
    # while(True):
    #     sleep(5)
    #     if all_finished(oldpids):
    #         break
    pid = os.getpid()
    print('runbg pid:', pid)
    home_dir= os.path.expanduser( '~/' ) 
    # 占用一块GPU的实验
    device='cuda:0'
    gpu_budget=4 # sst2: 最大用量 5197MiB

    logdir =home_dir+'data/AL/experiment/'+'tst20/' #
    spidir =home_dir+'data/AL/experiment/'+'spi20/' # dir for sample info analysis
    config_pool=[]

    hp_sst=HyperParam(task='sst-2',seeds=[0,1], total_ntrain=800, warm_size=70, rounds=8, ncentroids=[49, 48, 350, 500, 200, 8, 100, 40], unctth=[]) # 超大的ncentroid 仅适用于pseudo_err_1.3
    hp_sst=HyperParam(task='sst-2',seeds=[0,1], total_ntrain=800, warm_size=70, rounds=8, ncentroids=[49, 48, 350, 500], unctth=[]) # 超大的ncentroid 仅适用于pseudo_err_1.3
    hp=hp_sst

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

