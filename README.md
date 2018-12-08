# Nim-Cifar10

A test place for cifar-10 classification using [Arraymancer](https://github.com/mratsim/Arraymancer).

## How to use   
```
bash cifar10_downloader.sh # This downloads cifar-10 to cifar10 folder.
nim c -r -d:release demoCifar10.nim # compile and start to train
```

While training, csv files which include accuracy, val-loss, and train-loss are stored in ``` log ``` folder. For visualization, ``` visualize.py ``` is placed here.   
```
python3 visualize.py ./log/23481250.csv # example
# csv-name = [start-time(hour,minute,secound) + epoch].csv
```
This fig shows the performance of the current model. The accuracy can be improved of course.
<img src="https://github.com/cashiwamochi/nim_dl/blob/master/img/fig.png" width="500">

This implementation works on my MBP(2013-Late, HighSierra) and Ubuntu18.   
Nim -> 0.19.0   
Arraymancer -> 0.4.0   
If you are interested in this, please try to get better models.
