# Phytoplankton_Classifier
Classification of phytoplankton using ml using images

## Description of files
- Prelimary models: `evidence_knn_dt.ipynb`
- Preprocessing: `preprocessing.ipynb`
- Improved models (Naive Bayes): `gaussian-nb-removed-cyanophycae.ipynb`
- Improved models (SVM): `svm.py`
- Improved models (CNN): `CNN Final.ipynb`
- SOTA models (ResNet): `pretrained_resnet.ipynb`
- SOTA models (VGG): `pretrained_vgg.ipynb`

## Background

Phytoplanktons are autotrophic free floating algae which form the base of aquatic food chains. It is important to understand the diversity and community structure of phytoplanktons in water bodies due to several reasons. First, any changes in their community structure or in the periodicity of their blooms can radiate up the food web causing trophic cascade. Second, phytoplankton are sensitive to good or poor water quality and thus can be used as bioindicators of water quality. Finally, certain species of phytoplankton are toxic and early detection can . Current established methods of surveying phytoplankton use manual sorting, microscopy and labelling by professional taxonomists.

This creates several bottlenecks in the process of surveying phytoplanktons. Current statistics show that errors in identification occur with experts up to 10% for common species and 60% for rarer specimens (Dunker et al., 2018). Machine learning has the capabilities to automate the identification process. This helps in increasing speed and accuracy, and reducing the amount of skilled labour required. 

## Data Collection & Extraction

### Description of the raw data

The Imaging FlowCytobot (IFCB) (Olsen and Sosik, 2007) is an instrument that uses a combination of flow cytometry and video technology to capture high resolution images of phytoplankton. The IFCB can generate up to 30,000 images per hour. This allows a large amount of phytoplankton data to be collected. 

The dataset contains 63000 annotated images belonging to 50 different classes mainly of phytoplankton. The images were collected between 2016 - 2019 using an Imaging FlowCytobot from different locations in the Baltic Sea, and were manually-annotated by expert taxonomists. The distribution of samples across the classes is seen in Fig 1 of the Appendix and the distribution of samples across genera, displaying the highest represented genera in the sample is seen in Fig 2 of the Appendix. 

### Data Extraction - PCA
PCA is a statistical technique that can be used to find the components that maximise the variance and minimise the projection error in a dataset. For our improved models which require a 2D input, we need to convert our input data into shape (n_samples, n_features). Thereafter we carry out PCA with 100 components to reduce the dimensionality of the data. We ended up with an explained variance ratio of around 0.3.

### Data Extraction - Edge detection
We also do edge detection to get the shape of the cell. This makes sense for this image classification since all images are grayscale and different classes seem to have distinctively different shapes. We tried with both OpenCV and Pillow. It seems that OpenCV’s edge detection may be too aggressive, and it distorts the original shape too much. Hence we went with Pillow.

## Data Pre-processing

### Preliminary Models - KNN & DT

We need to convert our input data into shape (n_samples, n_features) for KNN and DT. However, the original dataset is too large and causes memory problems. Hence, we decided to convert the images to grayscale, reshape the images to 100x100 pixels and remove one of the classes (Cyanophyceae) for the purpose of this experiment. This leaves us with 32646 instances and 40 classes. This is because Cyanophyceae takes up 48% of our data but we are more concerned with Dinophyceae and Diatomophyceae as indicators of water quality. In the improved models, we will use the original dataset if possible. The amended dataset is shown in Fig 3 of the Appendix. Following that we do a 70/30 stratified train test split. We normalise the data first and run the models.

### Improved Models
We need to convert our input data into shape (n_samples, n_features) for the improved models which require a 2D input. In this case, we reshaped the images to 100x100 pixels. Following that we do a 70/30 stratified train test split. We normalise the data first, run PCA and run the models.

### SOTA Models
We preprocess the images in 4 ways, resizing, padding, denoising and augmenting the data. 
1. Firstly, the size of the images varied which makes it difficult for machine learning models to deal with. Hence, we resized them to 100x100 pixels. However, we wanted to maintain the aspect ratio of the images, since we felt this may be an important feature. Hence we decided to resize so that the maximum width/height of an image was 100 pixels first. Then we would pad the images so the result would be 100x100.
2. There is a lot of background in the images and it is usually grey. If we did a normal padding with white or black, it would not be very useful and may end up being an unnecessary feature since it may be interpreted as an edge. Hence, we padded the images with the mode of the image, that is, the most common pixel value in the image.
3. Marine snow is a shower of organic material falling from upper waters to the deep ocean. IFCB images tend to have some noise due to this marine snow (Zheng et al., 2017). Hence we want to eliminate unnecessary noise, and we used OpenCV’s non local means denoising function (Bradski, G., 2000). The fundamental principle of the nonlocal means algorithm is to average the colours of many image sub-windows that are similar to the one that makes up the pixel neighbourhood to replace the colour of a pixel.
4. We decided to do some data augmentation only for the minority classes, this was due to the large dataset, we did not want to add too much data. However, all augmentations were done sparingly since images generated by the IFCB are quite homogeneous. Hence, we only did horizontal flipping, vertical flipping and changes in brightness, where we halved the brightness and doubled it.

## Setup environment for Python
1. Run `conda env create --file environment.yml`
2. Activate with `conda activate cs3244_env`
3. If needed, update the env with `conda env update --file environment.yml`

## Classes
- 0: 'Amylax_triacantha'
- 1: 'Aphanizomenon_flosaquae'
- 2: 'Aphanothece_paralleliformis'
- 3: 'Beads'
- 4: 'Centrales_sp'
- 5: 'Ceratoneis_closterium'
- 6: 'Chaetoceros_sp'
- 7: 'Chaetoceros_sp_single'
- 8: 'Chlorococcales'
- 9: 'Chroococcales'
- 10: 'Chroococcus_small'
- 11: 'Ciliata'
- 12: 'Cryptomonadales'
- 13: 'Cryptophyceae-Teleaulax'
- 14: 'Cyclotella_choctawhatcheeana'
- 15: 'Cymbomonas_tetramitiformis'
- 16: 'Dinophyceae'
- 17: 'Dinophysis_acuminata'
- 18: 'Dolichospermum-Anabaenopsis'
- 19: 'Dolichospermum-Anabaenopsis-coiled'
- 20: 'Euglenophyceae'
- 21: 'Eutreptiella_sp'
- 22: 'Gonyaulax_verior'
- 23: 'Gymnodiniales'
- 24: 'Gymnodinium_like'
- 25: 'Heterocapsa_rotundata'
- 26: 'Heterocapsa_triquetra'
- 27: 'Heterocyte'
- 28: 'Katablepharis_remigera'
- 29: 'Licmophora_sp'
- 30: 'Melosira_arctica'
- 31: 'Merismopedia_sp'
- 32: 'Mesodinium_rubrum'
- 33: 'Monoraphidium_contortum'
- 34: 'Nitzschia_paleacea'
- 35: 'Nodularia_spumigena'
- 36: 'Oocystis_sp'
- 37: 'Oscillatoriales'
- 38: 'Pauliella_taeniata'
- 39: 'Pennales_sp_thick'
- 40: 'Pennales_sp_thin'
- 41: 'Peridiniella_catenata_chain'
- 42: 'Peridiniella_catenata_single'
- 43: 'Prorocentrum_cordatum'
- 44: 'Pseudopedinella_sp'
- 45: 'Pyramimonas_sp'
- 46: 'Skeletonema_marinoi'
- 47: 'Snowella-Woronichinia'
- 48: 'Thalassiosira_levanderi'
- 49: 'Uroglenopsis_sp'

