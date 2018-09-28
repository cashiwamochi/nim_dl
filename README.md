# Nim-Cifar10

A test place for cifar-10 classification using [Arraymancer](https://github.com/mratsim/Arraymancer).

## How to use   
```
bash cifar10_downloader.sh # This downloads cifar-10 to cifar10 folder.
nim c -r -d:release nimCifar10.nim # start to train
```

While training, csv files which include accuracy, val-loss, and train-loss are stored in ``` log ``` folder. For visualization, ``` visualize.py ``` is placed here.   
```
python3 visualize.py ./log/23481250.csv # example
# csv-name = [start-time(hour,minute,secound) + epoch].csv
```
