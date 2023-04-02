# Phytoplankton Classifier

## Background

Phytoplanktons are autotrophic free floating algae which form the base of aquatic food chains. It is important to understand the diversity and community structure of phytoplanktons in water bodies due to several reasons. First, any changes in their community structure or in the periodicity of their blooms can radiate up the food web causing trophic cascade. Second, phytoplankton are sensitive to good or poor water quality and thus can be used as bioindicators of water quality. Finally, certain species of phytoplankton are toxic and early detection can . Current established methods of surveying phytoplankton use manual sorting, microscopy and labelling by professional taxonomists.

This creates several bottlenecks in the process of surveying phytoplanktons. Current statistics show that errors in identification occur with experts up to 10% for common species and 60% for rarer specimens (Dunker et al., 2018). Machine learning has the capabilities to automate the identification process. This helps in increasing speed and accuracy, and reducing the amount of skilled labour required. 

## Data Collection & Extraction

### Description of the raw data

The Imaging FlowCytobot (IFCB) (Olsen and Sosik, 2007) is an instrument that uses a combination of flow cytometry and video technology to capture high resolution images of phytoplankton. The IFCB can generate up to 30,000 images per hour. This allows a large amount of phytoplankton data to be collected. 

The dataset contains 63000 annotated images belonging to 50 different classes mainly of phytoplankton. The images were collected between 2016 - 2019 using an Imaging FlowCytobot from different locations in the Baltic Sea, and were manually-annotated by expert taxonomists. The distribution of samples across the classes is seen in Fig 1 of the Appendix and the distribution of samples across genera, displaying the highest represented genera in the sample is seen in Fig 2 of the Appendix. 

### Data Pre-processing

#### Preliminary Models - KNN & DT

We need to convert our input data into shape (n_samples, n_features) for KNN and DT. However, the original dataset is too large and causes memory problems. Hence, we decided to convert the images to grayscale, reshape the images to 100x100 pixels and remove one of the classes (Cyanophyceae) for the purpose of this experiment. This leaves us with 32646 instances and 40 classes. This is because Cyanophyceae takes up 48% of our data but we are more concerned with Dinophyceae and Diatomophyceae as indicators of water quality. In the improved models, we will use the original dataset if possible. The amended dataset is shown in Fig 3 of the Appendix. Following that we do a 70/30 stratified train test split. We normalise the data first and run the models.

#### Improved Models - CNN

The images in the data set will be converted to floating point tensors for feeding to CNN. The process is as follows:
Read and decode the images to RGB grids of pixels with channels
Convert these to floating point tensors for input to neural nets
Rescale pixel values to the interval [0, 1] for efficiency 
