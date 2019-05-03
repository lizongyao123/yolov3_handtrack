# Hand tracking with PyTorch YOLO v3 + EgoHand dataset
<img src="https://user-images.githubusercontent.com/13127230/57129982-10dda900-6dd3-11e9-913b-f8292d5f616a.png" width="600">

# Note
This project is based on PyTorch YOLOv3 software developed by Ultralytics LLC which has **GPL-3.0 license**.<br>
I added explanations about how to train hand tracking and some patches for EgoHand dataset.<br>
For more information, please visit https://www.ultralytics.com and https://github.com/ultralytics/yolov3.<br>

# YOLO v3 software preparation
## Install
```bash
$ git clone https://github.com/speardutch/yolov3_handtrack.git && cd yolov3_handtrack
$ conda create --name yolov3_handtrack python=3.7
$ conda activate yolov3_handtrack
(yolov3_handtrack)$ conda install numpy opencv matplotlib tqdm
(yolov3_handtrack)$ conda install pytorch torchvision -c pytorch
(yolov3_handtrack)$ pip install opencv-python
```
From now, we consider we are in (yolov3_handtrack) virtual environment.

## Download COCO weights
Download pre-trainded weights and locate them to weights/ <br>
(I downloaded YOLO-spp weights and darknet53.conv.74)
- PyTorch `*.pt` format: https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI
Note if you don't download darknet53.conv.74 in weights folder, it will download automatically when train.

## run COCO demo

```bash
$ python detect.py --cfg cfg/yolov3-spp.cfg --weights weights/yolov3-spp.weights --webcam
```
<img src="https://user-images.githubusercontent.com/13127230/57130206-bdb82600-6dd3-11e9-805e-b0fac18125e4.png" width="600">

# Egohand dataset preparation
## Download dataset & convert
```bash
$ git clone https://github.com/victordibia/handtracking.git && cd handtracking
$ conda install scipy
$ python egohands_dataset_clean.py # download and convert m file
```
## Convert label file from m files to csv
```bash
$ git clone https://github.com/victordibia/handtracking.git && cd handtracking
$ conda install scipy
# Download EgoHand dataset, split them into train/test, convert label file m to csv, copy to images/ folder
$ python egohands_dataset_clean.py
```
After visualization of converting, check test_labels.csv and train_labels.csv file in images/test and inages/train folder with jpg files.

## Copy images folder to your dataset folder
Copy images/ folder to your EgoHands dataset folder
```bash
$ mkdir [EgoHands dataset] &&  cp [handtracking project folder]/images/* [EgoHands dataset]/ -rf
```

## Prepare label files and other files to train PyTorch YOLOv3 software.
```bash
$ cd yolov3_handtrack
$ python js_create_labels_from_csv.py  --csv [EgoHands dataset]/test/test_labels.csv 
399 files label created
$ python js_create_labels_from_csv.py  --csv [EgoHands dataset]/train/train_labels.csv 
4383 files label created
```
Note jpg files are 400 and 4400 files but there are files with no labels, so label files are lesser.<br>

See visualization if labels are created well(ESC for quit).
```bash
$ python js_visualize_dataset.py --data [EgoHands dataset]/test
```

Go to [EgoHands dataset]
```bash
$ cd [EgoHands dataset]
```

create train.txt and test.txt, list of test and train files(jpg)
```bash
$ find test/*.jpg -type f | xargs realpath > test.txt
$ find train/*.jpg -type f | xargs realpath > train.txt
```

create class file, we only have one class, "hand"
```bash
$ echo "hand" > classes.txt
```

# Train & Run Demo with Egohands
## Train
Go to project folder
```bash
$ cd yolov3_handtrack
```

Edit data cfg file, write down train.txt, test.txt, classes.txt file path
```bash
$ cd yolov3_handtrack
$ vi cfg/egohands-dataset.cfg
#
#classes=1   <== Do not Change
#train=[EgoHands dataset]/train.txt    <== Change to your EgoHands Dataset folder
#valid=[EgoHands dataset]/test.txt     <== Change to your EgoHands Dataset folder
#names=[EgoHands dataset]/classes.txt   <== Change to your EgoHands Dataset folder
# ...
```

Now, train!
```bash
$ python train.py --cfg cfg/yolov3-spp-egohands.cfg --data-cfg cfg/egohands-dataset.cfg --batch-size 8
```
Note if you don't downloaded darknet53.conv.74 in weights folder, it will download automatically when train.<br>
Change Hyper-parameters by option. I trained with above option with RTX 2070 GPU, 273 epochs 14 hours and mAP was 0.962<br>
You can resume train with `--resume` option, if terminated unfinished. <br>
You can plot training status with below commands.
```bash
$ python
>> from utils import utils
>> utils.plot_results()
>> exit()
$ eog results.png
```
<img src="https://user-images.githubusercontent.com/13127230/57130058-4c787300-6dd3-11e9-80db-07f269ec64d0.png" width="600">

Results will be saved in weights/ folder, best.pt file is the best weights. <br>

## Run demo
```bash
python detect.py --cfg cfg/yolov3-spp-egohands.cfg --data-cfg cfg/egohands-dataset.cfg --weights weights/best.pt --webcam
```
I got about **15~18ms** inference time, approximately **50fps >**, with RTX 2070 GPU.

<img src="https://user-images.githubusercontent.com/13127230/57129982-10dda900-6dd3-11e9-913b-f8292d5f616a.png" width="600">

# To Do
- Hyper parameters tuning for better result
- Use other dataset for more robust inferrence
- Please post issues if you found better way!!!