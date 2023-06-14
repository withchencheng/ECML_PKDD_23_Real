# ECML_PKDD_23_Real
Code for our ECML/PKDD 2023 Paper:
  Real: A Representative Error-Driven Approach for Active Learning

Please cite our work if you find the code useful.
```
@inproceedings{chencheng2023Real,
	Author = {Cheng Chen, Yong Wang, Lizi Liao, Yueguo Chen, Xiaoyong Du},
	Booktitle = {ECML/PKDD},
	Title = {Real: A Representative Error-Driven Approach for Active Learning},
	Year = {2023}
}	
```
Contact: Cheng Chen (chchen [at] ruc.edu.cn)

## Data Preparation 

Specify data dir at `runbg.py` in `class Config`
```
   self.data_dir = f'{home_dir}data/AL/data3/{task}/'
```

Except for pubmed, the classes for all the other datasets are close to be balanced.



## Run the AL Process
Change to `real` dir under this repository, then execute
```bash
python runbg_agnews.py
```


