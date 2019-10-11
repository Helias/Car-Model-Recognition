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
- torch (pytorch)
- torchvision

You can install the requirements using:
```
pip3 install -r requirements.txt
```

**Troubleshooting**: if you get some errors about pytorch or torchvision install use `sudo` to install it.

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

This conversion happens automatically when you just add a directory inside the IMAGES_PATH, if you add tomorrow a new car, like, FIAT, the program will add automatically to the classes, just **pay attention to the order of the classes inside num_classes and the related trainin,testing and validation CSV files**.

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

Anyway, this paragraph is only for your info, the CSV files are automatically genrated by the preprocessing phase explained in the follow paragraph.


### Preprocess the dataset
You have to generate the CSV files and calculate the mean and standard deviation to apply a normalization, just use the -p parameter to process your dataset so type:

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

## Troubleshooting

### Size mismatch

Error:
```python
RuntimeError: Error(s) in loading state_dict for ResNet:
size mismatch for fc.weight: copying a param with shape torch.Size([1000, 2048]) from checkpoint, the shape in current model is torch.Size([3, 2048]).
size mismatch for fc.bias: copying a param with shape torch.Size([1000]) from checkpoint, the shape in current model is torch.Size([3]).
```

**Solution**: probably you need to re-train your neural network model because you are using a wrong model for your data and classes, so don't use some **pretrained** model but train a new neural network with your data/classes.

### CUDA out of memory

Error:
```python
######### ERROR #######
CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 1.96 GiB total capacity; 967.98 MiB already allocated; 25.94 MiB free; 48.02 MiB cached)


######### batch #######
[images.png, files_path.png, ....]

Traceback (most recent call last):
  File "main.py", line 227, in <module>
    train_model_iter("resnet152", resnet152_model)
  File "main.py", line 215, in train_model_iter
    model, loss_acc, y_testing, preds = train_model(model_name=model_name, model=model, weight_decay=weight_decay)
  File "main.py", line 124, in train_model
    epoch_loss /= samples
ZeroDivisionError: division by zero
```

**Solution**: you're using CUDA, probably the memory of your GPU is too low for the batch size that you're giving in input, try to reduce the `BATCH_SIZE` from **config.py** or use your RAM instead of GPU memory if you have more, so put `USE_CUDA=false` in **config.py**.


## Credits
- [Helias](https://github.com/Helias)
