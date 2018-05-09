# cylindrical-histogram-AAE
An implementation of the paper "Applying Adversarial Auto-encoder for Estimating Human Walking Gait Index"

## Requirements
* Python
* Numpy
* TensorFlow
* Scikit-learn
* Matplotlib

## Notice
* The code was implemented to directly work on [DIRO gait dataset](http://www-labs.iro.umontreal.ca/~labimage/GaitDataset/)
* Please download the [histogram data](http://www.iro.umontreal.ca/~labimage/GaitDataset) and put the npz file into the folder **dataset**

## Usage
Process default training and test sets with suggested parameters
```
$ python3 main.py
```
Specify test subject for leave-one-out cross-validation, store sampled histograms and save AUC results
```
$ python3 main.py -l 0 -s 1 -f results.csv
```
Param | Description
---: | :---
 -l | index of test subject (0 to 8 for 9 subjects in DIRO gait dataset)
 -e1 | first epoch for evaluation
 -e2 | last epoch for evaluation
 -o | overlapping segments (i.e. using sliding window)
 -s | sampling histogram (GAN) and save images
 -f | file for saving AUC results

Saved file structure

Test subject | Segment length | AUCs of partial and combined measures
:---: | ---: | :---:
 0 |    1 | AUC values
 0 |   10 | AUC values
 ...|.....|.....
 0 | 1200 | AUC values

## Example of output
Default training and test sets
```
(512, 1)
(?, 1)
training subjects: [0 2 4 5 8]
data shape:
(6000, 256)
(4800, 256)
(38400, 256)

Epoch   1: D_loss 9.980, G_loss 0.032, Recon_loss: 0.641
Epoch   2: D_loss 7.510, G_loss 0.025, Recon_loss: 0.499
.....
Epoch 309: D_loss 1.545, G_loss 0.616, Recon_loss: 0.328
Epoch 310: D_loss 1.550, G_loss 0.612, Recon_loss: 0.327

FINAL RESULTS (AVERAGE)

Results probability
(   1) AUC = 0.5952 (+0.0204)
(  10) AUC = 0.6319 (+0.0310)
.....
(1200) AUC = 0.6766 (+0.0404)

Results discriminator
(   1) AUC = 0.4238 (+0.0468)
(  10) AUC = 0.3965 (+0.0591)
.....
(1200) AUC = 0.3516 (+0.0872)

Results reconstruction
(   1) AUC = 0.8151 (+0.0074)
(  10) AUC = 0.8732 (+0.0079)
.....
(1200) AUC = 0.9524 (+0.0092)

Results dist + prob
(   1) AUC = 0.8142 (+0.0074)
(  10) AUC = 0.8724 (+0.0078)
.....
(1200) AUC = 0.9513 (+0.0085)

Results dist + disc
(   1) AUC = 0.8085 (+0.0104)
(  10) AUC = 0.8700 (+0.0111)
.....
(1200) AUC = 0.9594 (+0.0113)

Results combination
(   1) AUC = 0.8073 (+0.0103)
(  10) AUC = 0.8687 (+0.0110)
.....
(1200) AUC = 0.9583 (+0.0112)
```

Portions of the work employed codes from [Agustinus Kristiadi](https://github.com/wiseodd) and [kevinroth](https://github.com/rothk).
