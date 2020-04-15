#Satellite_Imagery_Detection
Convolutional Neural Network and Support Vector Machine for binary classification task involving statellite imagery.

#A Comparative Study of Support Vector Machines and Convolutional Neural Networks for Object Detection Tasks Involving Satellite Imagery
Tudor Dorobantu
Tudor.Dorobantu@city.ac.uk

Abstract

This paper is motivated by the success of AI practices in generating Alternative Data strategies. Two Machine Learning algorithms are considered for a binary classification task involving detection of ships from satellite imagery. The models investigated are Support Vector Machine (SVM) and Convolutional Neural Networks (CNN). Model parameter choice are based on optimizing for accuracy scores while the F1 Score metric is used to compare the performance of competing algorithms. The Convolutional Neural Network proved to be slightly better in the context of the proposed classification task.

Implementation Details

• run runModels_script.m for trained algorithms
• To download the original data please go to the link in the LINK_TO_ORIGINAL_DATA.txt. Alternatively, the data can be downloaded by following the link here: https://www.kaggle.com/rhammell/ships-in-satellite-imagery
• To run the entire experiment pipeline please run runExperiments_livescript.mlx. Make sure that the data is in the augmented_data folder. The livescript will function with both the augmented and original data set without requiring any modifications.
• To augment the image data (e.g. add noise, gaussian blur, etc.) please use the ImageAugment.py class. Instructions on how to use the class are available here: https://github.com/tdorobantu/ImageAugment
