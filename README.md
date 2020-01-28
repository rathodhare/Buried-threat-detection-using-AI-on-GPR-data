# GPRS-using-AI
We, Achin and Harekrissna worked as a team to complete the project given to us on Buried threat detection using ground penetrating radar. 

We applied Machine Learning techniques specifically CNN to identify the threats hidden underground by analysing the radar data. We implemented many techniques given in the research paper (Some Good Practices for Applying Convolutional Neural Networks to Buried Threat Detection in Ground Penetrating Radar, by Daniël Reichman, Leslie M. Collins, Jordan M)

The implementation goes as follows: -
  1. We used grayscale imagery from the Cifar10 dataset to pre-train initial layers of our GPR CNN.
  
  2. The parameters for all layers of a CNN are learned jointly using adam optimizer
  
  3. In training, the dataset is provided to the network in mini-batches (i.e., small, randomly selected
subsets) and the whole dataset is traversed times (epochs). To update the weights, the network
is evaluated on a mini-batch of data, the prediction is compared to the training labels, and the
error is back-propagated through the network. At the end of an epoch, the network is typically
evaluated on a validation set to measure whether the network is learning general patterns or
overfitting to the training data The final network chosen for testing is the one with the highest
validation accuracy.

  4. Network pretraining: Pretraining here was done by transferring at most 4 convolutional layers
from the network trained on the Cifar10 dataset to the GPR network.

  5. Training data augmentation: In the vertical dimension, patches at every other pixel above and
below the maximum keypoint location for threats are chosen, up to 4 pixels away. In the
horizontal dimension, one patch is chosen 2 pixels away, in both directions from the maximum
energy keypoint for threats. For non-threats, 33% of the data is randomly sampled from 3
A-scans: the central A-scan, one A-scan 2 pixels to the right and to the left of the central A-scan.

  6. The CNN network is same as that proposed in the research paper
