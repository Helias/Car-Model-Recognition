# Car Model Recognition

This is a university project for the course "Computer Vision".
This project consists in a classifier of car model.

## Requirements
- Python3
- numpy
- pytorch
- torchvision
- scikit-learn
- matplotlib
- pillow

## Usage

First, if you have no resnet152 model trained and you need from scratch to do it you need to:

- download dataset
- preprocess the dataset
- train the model

After you can try a new sample.

### Download dataset

I suggest to use [VMMRdb](http://vmmrdb.cecsresearch.org/) as dataset, it's free and full of labelled images for car model recognition instead of detection (the most dataset is for this).

So download the dataset, select some models and put the directory model in the dataset folder, any directory in "dataset" will be considered a new class.

### Handle CSV training, testing, validation and dataset structure

The dataset structure should be like this:
```
dataset / classes / file.jpg
```

For example, we have 3 classes: **honda_civic, nissan and ford**:
```
dataset_dir / honda_civic / file1.jpg
dataset_dir / honda_civic / file2.jpg
....
dataset_dir / nissan / file1.jpg
dataset_dir / nissan / file2.jpg
....
dataset_dir / ford / file1.jpg
dataset_dir / ford / file2.jpg
...
and so on.
```

The **"dataset_dir"** is the **IMAGES_PATH** in config.py.
The python script will save the classes in a  dict() named **num_classes**, like this:
```
num_classes = {
  "honda_civic": 1,
  "nissan": 2,
  "ford": 3
}
```

This conversion happens automatically when you just add a directory inside the IMAGES_PATH, if you add tomorrow a new car, like, FIAT, the program will add automatically that to the classes, just **pay attention to the order of the classes inside num_classes and generate the related trainin,testing and validation CSV files**.

The file **training, testing and validation (CSV)** should contain only two columns:
**FILE_NAME, NUM_CLASS**

Example of CSV file:
```
file1.jpg, 1
file2.jpg, 1
file1.jpg, 2
file2.jpg, 2
file1.jpg, 3
file2.jpg, 3
```

### Preprocess the dataset
You have to calculate the mean and standard deviation to apply a normalization, just use the -p parameter to process your dataset so type:

```
$ python3 main.py -p
```

### Train the model

To train a new model resnet152 model you can run the main.py with the -t parameter, so type:

```
$ python3 main.py -t
```

The results will be saved in the results/ directory with the F1 score, accuracy, confusion matrix and the accuracy/loss graph difference between training and testing.

## Try new sample

To try predict a new sample you can just type:
```
python3 main.py -i path/file.jpg
```

---

I used this project predicting 3 models:
- Nissan Altima
- Honda Civic
- Ford Explorer

I selected all 2000-2007 images from VMMRdb, so I downloaded the full dataset and choose the 2000-2007 images and put them into one directory per class (so I had 3 directory named "Ford Explorer", "Nissan Altima", "Honda Civic" in dataset folder).

## Credits
- [Helias](https://github.com/Helias)
