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

If you need more data for your project you can also add the followings dataset:
- [Stanford Cars Dataset from jkrause](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) (low images quantity)
- [Comprehensive Cars Database](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/), here the module to get this dataset [MODULE](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/agreement.pdf)

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

**Little introduction**

Before the training process, modify the `EPOCHS` parameter in `config.py`, usually with 3 classes 30-50 epochs should be enough, but you have to see the results_graph.pn file (when you finish your training with the default epochs parameter) and check if the blue curve is stable.

An example of the graph could be the follow:
![graph results - Car Model Recognition](https://user-images.githubusercontent.com/519778/67412403-81fe5c00-f5bf-11e9-9bd1-e86251bb9a0c.png)

After 45-50 epochs (number bottom of the graph), the blue curve is stable and does not have peaks down.
Moreover, the testing curve (the orange one) is pretty "stable", even with some peaks, for the testing is normal that the peaks are frequently.

**Train the model**

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

### - Size mismatch

Error:
```python
RuntimeError: Error(s) in loading state_dict for ResNet:
size mismatch for fc.weight: copying a param with shape torch.Size([1000, 2048]) from checkpoint, the shape in current model is torch.Size([3, 2048]).
size mismatch for fc.bias: copying a param with shape torch.Size([1000]) from checkpoint, the shape in current model is torch.Size([3]).
```

**Solution**: probably you need to re-train your neural network model because you are using a wrong model for your data and classes, so don't use some **pretrained** model but train a new neural network with your data/classes.

### - CUDA out of memory

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

### - "My model does not recognize exactly the class"
Probably you have to increase the **DATA PER CLASS** in your dataset, a good number of images per class could be 10k (10 000 items), but with only 3 classes you can even use 2k-5k items per class.
Another parameter that affect hugely the training is the **EPOCHS**, try to at least 50 epochs if you are not satisfied about the results.


**You are not the only one to get this troubles, check the issue [#3](https://github.com/Helias/Car-Model-Recognition/issues/3) to get a full conversation of this solutions/troubleshooting.**


## Credits
- [Helias](https://github.com/Helias)

### Contribute
You can help us [opening a new issue](https://github.com/Helias/Car-Model-Recognition/issues/new) to report a bug or a suggestion  

or you can donate to support us

[![Donate PayPal](https://camo.githubusercontent.com/ed44813b2a0ca01f80a00cca116f04208c127a80/68747470733a2f2f7777772e70617970616c2e636f6d2f656e5f47422f692f62746e2f62746e5f646f6e61746543435f4c472e676966)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WE5LZM2D4WPBC&source=url)
