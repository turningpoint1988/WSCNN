## Weakly-supervised convolutional neural network for predicting protein-DNA binding (WSCNN)
RC-CNN and WSCNN models are both based on this work [SIL-CNN](https://github.com/gifford-lab/mri-wrapper), and implemented on the [Caffe](http://caffe.berkeleyvision.org/installation.html) ploatform.

## Data preparation for RC-CNN
In the RC-CNN file, the script rc-cnn.py, which incorparates the reverse-complement mode, is to transform DNA sequences to the data format that Caffe can take.

+ Usage example:
	```
	python rc-cnn.py example/train.tsv example/train_target.tsv example/data/train.h5 -k 24
	python rc-cnn.py example/test.tsv example/test_target.tsv example/data/test.h5 -k 24
	```
+ Type the following for details on other optional arguments:
	```
	python rc-cnn.py -h
	```
The trainval.prototxt is for training RC-CNN, and the deploy.prototxt is for testing RC-CNN.

## Data preparation for WSCNN
In the WSCNN file, the script wscnn.py, which also incorparates the reverse-complement mode, is for tranforming DNA seqences to the weakly-supervised data format whose data shape is N*C*H*W (N denotes the number of bags, and C=4 denotes four channels, and H denotes the number of instances per bag, and W denotes the length of instnces).

+ Usage example:
	```
	python wscnn.py example/train.tsv example/train_target.tsv example/data/train.h5 -c 79 -s 10 -k 24
	python wscnn.py example/test.tsv example/test_target.tsv example/data/test.h5 -c 79 -s 10 -k 24
	```
+ Type the following for details on other optional arguments:
	```
	python wscnn.py -h
	```
The four files (Max, Average, Linear-Regression, and Top-Bottom) correspond to the four fusion methods, where the trainval.prototxt is for training WSCNN, and the deploy.prototxt is for testing WSCNN. 
In the Top-Bottom file, all scripts in the 
