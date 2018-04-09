## Weakly-supervised convolutional neural network for predicting protein-DNA binding (WSCNN)
RC-CNN and WSCNN models are both based on this work [SIL-CNN](https://github.com/gifford-lab/mri-wrapper), and implemented on the [Caffe](http://caffe.berkeleyvision.org/installation.html) ploatform.

## Data preparation for RC-CNN
In the RC-CNN/ directory, the script rc-cnn.py, which incorporates the reverse-complement mode, is for transforming DNA sequences to the data format that Caffe can take, whose size is N * C * H * W (N denotes the number of sequences, and C (=4) denotes the four channels {A, C, G, T}, and H (=1) denotes the height of sequences, and W denotes the length of sequences).

+ Usage example:
	```
	python rccnn.py example/train.tsv example/train_target.tsv example/data/train.h5 -k 24
	python rccnn.py example/test.tsv example/test_target.tsv example/data/test.h5 -k 24
	```
+ Type the following for details on other optional arguments:
	```
	python rccnn.py -h
	```
The trainval.prototxt is for training RC-CNN, and the deploy.prototxt is for testing RC-CNN.

## Data preparation for WSCNN
In the WSCNN/ directory, the script wscnn.py, which also incorporates the reverse-complement mode, is for transforming DNA sequences to the weakly-supervised data, whose size is N * C * H * W (N denotes the number of bags (sequences), and C (=4) denotes the four channels {A, C, G, T}, and H denotes the number of instances per bag, and W denotes the length of instnces).

+ Usage example:
	```
	python wscnn.py example/train.tsv example/train_target.tsv example/data/train.h5 -c 79 -s 10 -k 24
	python wscnn.py example/test.tsv example/test_target.tsv example/data/test.h5 -c 79 -s 10 -k 24
	```
+ Type the following for details on other optional arguments:
	```
	python wscnn.py -h
	```
The four files (Max, Average, Linear-Regression, and Top-Bottom) correspond to the four fusion methods in this paper, where the trainval.prototxt is for training WSCNN, and the deploy.prototxt is for testing WSCNN. 
In the WSCNN/Top-Bottom/caffe-scripts/ directory, the scripts are the implementation of Top-Bottom Instances method on the Caffe platform.
