# AI-Image-Detection

---Before trying either of these files 2 seperate folders need to be downloaded that I sent as a Google Drvie since the files are too big for github. 
-train folder - dataset containing a lot of images to train our model 
-dataset folder - contains 2 files that will be used in the gld_test, those files are train.csv, and train_label_to_category.csv.

gld_test.py - This file tests the Google Landmarks Dataset, in order to run this file you need to download the dataset of images in the train folder, as well as download the following .csv files - train.csv, and train_label_to_category.csv. 

Correct ouput should look like this
________________________________________________________________________
Image found at: train\0\0\2\002a196a7a4d4e48.jpg
Landmark ID: 90864
Category: http://commons.wikimedia.org/wiki/Category:Basilica_of_the_Assumption_(Prague)
Landmark Name: Category:Basilica_of_the_Assumption_(Prague)
Coordinates not found

--So far it only accepts images within the dataset, but it's a good starting point.


google_vision_test.py - This file uses the Google vision api using an api key. I've tested this by creating my own api key on my google cloud account, but everyone should be able to use it in testing. Also need to make sure the training set is downloaded in order to use this as well. 

Correct output should look like this
_______________________________________________________________________
Detected Landmark: Frederiksborg Castle
Location: 55.93496520000001, 12.3012724

In the future if we get both of these working for everyone, these resources can be combined to make a very useful dataset. 
