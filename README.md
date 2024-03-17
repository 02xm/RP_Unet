# RP_Unet
Rand Augmentation and Pixel Shuffle Unet  
  
You should follow the folder format above. Place your dataset in inputs folder. The dataset_name is the same as you can set in the train.py. 
Then, you should begin to train to check whether it could work. The models and outputs could generate automatically by your set.  
  
arch.py include the main model of experiments, you also can modify by yourself.  
convert.py allows you to convert the size of picture to all you like.(Warning:It should match the input_h and input_w in train.py)  
fps.py allows you to calcate the fps of your model.  
draw.py, summary_accuracy.py, summary_loss.py and summary_accuracy.py can draw some necessary picture of training process.  
metrics.py stores some evalution index, you can add with your need.  
pixelShuffle.py has the pixel shuffle algorithm.  
rand_augmentation.py has the method to rand augment the picture.  
val.py can valid the dataset and save as to outputs folder, dataset is divided as training and validing.  
other .py is utilized as utils.  
