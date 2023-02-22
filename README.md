# Automated Blood Cell count from Blood Smear Images for COVID-19 Diagnosis

## 1 Problem Statement & Aims

The COVID-19 pandemia has to date reached more than 156 million confirmed cases (probably a much higher number of infected), and almost 3.27 million deaths. We all know that the recent outbreak of the COVID-19 is an urgent global concern. To deal with it, healthcare professionals need to make rapid and accurate decisions on the COVID-19 diagnosis, treatment, and isolation needs. Therefore, many researchers
have been recently attracted to provide techniques that can improve the mentioned decisions.The current gold standard test for COVID-19 diagnosis is amplification of viral RNA by (real time) reverse transcription polymerase chain reaction
(rRT-PCR). However, it presents known shortcomings such as long turnaround times (3-4 hours to generate results), potential shortage of reagents, false-negative rates as large as % 15-20 , the need for certified laboratories, expensive equipment and trained personnel. Thus there is a need for alternative, faster, less expensive and more accessible tests. A recent work has shown that a simple blood test might help to reduce the false-negative rRT-PCR tests [1]. Blood tests also can be used in developing countries and in those countries suffering from a shortage of rRT-PCR reagents and/or specialized laboratories as an inexpensive and available alternative to identify potential COVID-19 patients. However, performing regular blood tests also requires medical experts. To leverage the advantage of blood analysis and reduce the cost of performing blood tests, researchers are recently working on the automated blood-based COVID-19 diagnosis tests.
In this study, the blood sample image of each patient will be considered as a new testing sample, and the probability of being diagnosed with positive COVID-19 ( diagnosis label ) will be the class predicted for that test sample. Figure 1 shows a graphical representation of the pipeline of this project. Based on this figure, it can be observed that we will utilize two data sets (𝑖) data set of blood cell smear images with annotations (rectangular box and labels) for blood cells (WBC, RBC, and Platelets) and labels for the WBC subtypes (Neutrophil, Monocyte, Lymphocyte, Eosinophil); and (𝑖𝑖) data set of COVID binary classification having features of blood cell and sub-types count. In this project, we have three main tasks as follows:
(1) Detection of blood cell objects (WBC, RBC, Platelets) in the collected smear image;
(2) Classification of WBC sub-types based on the cropped subimages from detected box for each WBC; and
(3) COVID-19 Diagnosis based on the features collected from task (1) and task (2).

<figure>
<img src="images/pipeline.jpg" alt="Trulli" style="width:100%">
<figcaption align = "center"><b>Fig.1 - Graphical Representation of pipeline of this project</b></figcaption>
</figure>

## 2 DATASET

In this project, we have exploited two data sets. The data set is the combined version of two data sets: https://github.com/Shenggan/BCCD_Dataset and https://www.kaggle.com/paultimothymooney/blood-cells that contain a total of 17,092 annotated and labeled images of blood cells. In this data set, each blood cell object (WBC, RBC, and Platelets) are annotated with a rectangular ground truth box and corresponding labels. Moreover, in this data set, the cropped pictures of each WBC object is also labeled as following four groups: Neutrophil, Monocyte, Lymphocyte,
Eosinophil. The size of the images is 540×960 pixels, in format of jpg. The second dataset http://zenodo.org/record/3886927#.YFqiLkhKjVp is consisted of routine blood-test results performed on 1,925 patients on admission to the ED at the San Raffaele Hospital (OSR) from February 19, 2020, to May 31, 2020. For each sample, COVID-19 diagnosis was determined based on the result of the molecular test for SARS-CoV-2 performed by RT-PCR on nasopharyngeal swabs. The response of each COVID-19 test data sample takes a binary value {0, 1} in case the COVID-19 test result is {𝑛𝑒𝑔𝑎𝑡𝑖𝑣𝑒, 𝑝𝑜𝑠𝑖𝑡𝑖𝑣𝑒}, respectively. Table 1 represents the available features in this data set.

| Table 1. Feature Descriptions |
| ----------------------------- | --------------------- |
| Feature                       | Type                  |
| -------                       | -----                 |
| Gender                        | Categorical           |
| Age                           | Categorical           |
| Leukocytes (WBC)              | Numerical(continuous) |
| Red Blood Cells (RBC)         | Numerical(continuous) |
| Platelets                     | Numerical(continuous) |
| Neutrophils                   | Numerical(continuous) |
| Lymphocytes                   | Numerical(continuous) |
| Monocytes                     | Numerical(continuous) |
| Eosinophils                   | Numerical(continuous) |
| Basophils                     | Numerical(continuous) |

## 3 DATA PRE-PROCESSING

• Since we are using a 300 variant of single shot multibox detection for blood dection (SSD300) model for cell detection in images of task 1, input images to the model have to be (1) transformed into float tensors with size 3𝑥300𝑥300; and (2) normalized to ImageNet images’ RGB channels.
• For task 2 (classification of WBC sub-types), the data was highly imbalanced, i.e., some sub-types have large number of samples while others don’t. Therefore, to make the data set balanced, we needed to augment it by adding rotation, mirroring, random cropping, and shearing so that all the WBC sub-types will have the same number of image samples. The data augmentation will increase the number of all data samples being imported to the CNN model, and make the train and validation data more balanced.
• In task 2, before importing images to the deep neural networks (DNNs), we also need to normalize the blood cell images by scale of 255.

## Task 1: Detection of blood cell objects

The objective of task 1 is to detect the blood cell objects (WBC, RBC, and Platelets) in the blood smear images. Among all objects, the detection of WBC is more important since it will then be imported into task 2. We performed one unsupervised and two supervised object detection techniques. Figure 2 will represent all the steps that we performed in this task.

<figure>
<img src="images/Task1.jpg" alt="Trulli" style="width:100%">
<figcaption align = "center"><b>Fig.2 - Graphical Representation of models utilized in Task 1</b></figcaption>
</figure>

## Task 2: Classification of WBCs

The main objective of this task is to classify the cropped smear images of detected WBC into four sub-types (Neutrophil, Monocyte, Lymphocyte, Eosinophil). To do so, we utilize eight CNN models:

(1) customized architecture 1 (Figure 3);
(2) customized architecture 2 (Figure 4);
(3) pre-trained ResNet50 [5];
(4) pre-trained DenseNet121 [6];
(5) pre-trained VGG16 [15];
(6) pretrained MobileNet-v2 [13];
(7) pre-trained Xception [3]; and
(8) pre-trained Inception [16].

<figure>
<img src="images/architecture1.jpg" alt="Trulli" style="width:100%">
<figcaption align = "center"><b>Fig.3 - Graphical Representation of customized architecture 1</b></figcaption>
</figure>

<figure>
<img src="images/architecture2.jpg" alt="Trulli" style="width:100%">
<figcaption align = "center"><b>Fig.4 - Graphical Representation of customized architecture 2</b></figcaption>
</figure>

Task 3: The main objective of this task is to classify COVID-19 diagnosis based on the collected
features collected from task 1 and task 2. The training of this task is based on the second data set in which the sample size is too small (201 samples). Therefore, we decided to perform this classification task by the classical machine learning classification models instead of deep learning models. To do so, we are using nine classifiers as follows:

(1) Logistic Regression [12];
(2) K-Nearest Neighbor [9];
(3) Linear SVM [4];
(4) Non-Linear SVM [4];
(5) Gaussian Process [11];
(6) Decision Tree[8];
(7) Random Forest [2];
(8) Multi-Layer Perceptron (MLP) [10] with three layers containing (20,40,20)
neurons, respectively;
(9) AdaBoost [18].

## Blood Cell Detection Output

<figure>
<img src="images/Detection_Examples.jpg" alt="Trulli" style="width:100%">
<figcaption align = "center"><b>Fig.5 - Detection examples on BCCD test dataset with VGG16(top) and Alexnet(bottom) based SSD models.</b></figcaption>
</figure>

## White Blood Cell identification

Figure 6 and 7 illustrate the layer-1 feature maps generated from DenseNet121 and architecture 2,respectively, for three randomly selected samples.

<figure>
<img src="images/Detection_Examples.jpg" alt="Trulli" style="width:100%">
<figcaption align = "center"><b>Fig.6 - Layer-1 feature maps generated by pretrained DenseNet121.</b></figcaption>
</figure>

<figure>
<img src="images/Detection_Examples.jpg" alt="Trulli" style="width:100%">
<figcaption align = "center"><b>Fig.7 - Layer-1 feature maps generated by customized architecture 2.</b></figcaption>
</figure>
