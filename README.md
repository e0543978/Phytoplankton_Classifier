# Phytoplankton Classifier

## Background

Phytoplanktons are autotrophic free floating algae which form the base of aquatic food chains. It is important to understand the diversity and community structure of phytoplanktons in water bodies due to several reasons. First, any changes in their community structure or in the periodicity of their blooms can radiate up the food web causing trophic cascade. Second, phytoplankton are sensitive to good or poor water quality and thus can be used as bioindicators of water quality. Finally, certain species of phytoplankton are toxic and early detection can . Current established methods of surveying phytoplankton use manual sorting, microscopy and labelling by professional taxonomists.

This creates several bottlenecks in the process of surveying phytoplanktons. Current statistics show that errors in identification occur with experts up to 10% for common species and 60% for rarer specimens (Dunker et al., 2018). Machine learning has the capabilities to automate the identification process. This helps in increasing speed and accuracy, and reducing the amount of skilled labour required. 

## Data Collection & Extraction

### Description of the raw data

The Imaging FlowCytobot (IFCB) (Olsen and Sosik, 2007) is an instrument that uses a combination of flow cytometry and video technology to capture high resolution images of phytoplankton. The IFCB can generate up to 30,000 images per hour. This allows a large amount of phytoplankton data to be collected. 

The dataset contains 63000 annotated images belonging to 50 different classes mainly of phytoplankton. The images were collected between 2016 - 2019 using an Imaging FlowCytobot from different locations in the Baltic Sea, and were manually-annotated by expert taxonomists. The distribution of samples across the classes is seen in Fig 1 of the Appendix and the distribution of samples across genera, displaying the highest represented genera in the sample is seen in Fig 2 of the Appendix. 
