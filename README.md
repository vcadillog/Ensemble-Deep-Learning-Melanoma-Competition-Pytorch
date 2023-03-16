# Ensemble-Deep-Learning-Melanoma-Competition-Pytorch

Adaptation for learning of Ensemble Neural Network of the Kaggle competition winner code and solution in a Jupyter Notebook.

Has been added:

- Weight & Biases to plot the model metrics.

- Result comparison between Ensemble Neural Networks using the metadata of the images and a Convolutional Neural Network.

The model was trained only on 2 Kfolds, 15 epochs in the preprocessed data of 256x256 size without a resize for the data augmentation to fit the Kaggle limits, modify below that consideration.

The ensemble model result has a score in the kaggle competition of:

- Private: 0.9208  Public : 0.9241

The CNN model result has a score in the kaggle competition of:

- Private: 0.9078  Public : 0.9158

W&B plots:

- Accuracy
![Alt text](https://github.com/vcadillog/Ensemble-Deep-Learning-Melanoma-Competition-Pytorch/blob/main/images/Accuracy.png)

- Ensemble Network loss
![Alt text](https://github.com/vcadillog/Ensemble-Deep-Learning-Melanoma-Competition-Pytorch/blob/main/images/Ensemble%20Loss.png)

- CNN loss
![Alt text](https://github.com/vcadillog/Ensemble-Deep-Learning-Melanoma-Competition-Pytorch/blob/main/images/NoMeta%20Loss.png)

- Epochs
![Alt text](https://github.com/vcadillog/Ensemble-Deep-Learning-Melanoma-Competition-Pytorch/blob/main/images/Epochs.png)

- Folds
![Alt text](https://github.com/vcadillog/Ensemble-Deep-Learning-Melanoma-Competition-Pytorch/blob/main/images/Folds.png)


[1] This work was inspired by 1st place Melanoma winners public repository : 

https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution
 
