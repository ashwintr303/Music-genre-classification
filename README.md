## 1. Introduction 

This project aims to classify music genres using Multinomial logistic classification and Multilayer Perceptron(MLP). We have extracted features from time domain and domain. We have successfully reduced the number of features of MLP and logistic classifier by half while still maintaining a high accuracy.

## 2. Dataset  

We have used the GTZAN dataset which can be found [here](http://marsyas.info/downloads/datasets.html).The dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. We have used the four most distinct genres for this project; classical, jazz, metal and pop. Hence our data set has 400 tracks that are all 22050Hz Mono 16-bit audio files in .wav format. 

## 3. Feature Extraction
The time domain and frequency domain features are extracted from the audio files using [librosa](https://librosa.github.io/), a python package for music and audio analysis. 

### 3.1 Time Domain Features

**3.1.1 Root Mean Square (RMSE)**
RMSE corresponds to the root mean square energy of a signal. RMSE can be calculated as:  
<p align="center">
    <img src="Music-genre-classification/misc/figures/centroid.jpg"/>
    <figcaption>Fig1. Formula to calculate RMSE of a signal</figcaption>
</p>

**3.1.2 Zero Crossing Rate (ZCR)**
A zero-crossing is a point where the sign of a mathematical function changes (positive to negative or vice versa). Zero crossing rate is calculated as the total number zero crossings divided by the length of the frame. It usually has higher values for highly percussive sounds like those in metal and rock. It is calculated as:  
<p align="center">
    <img src="Music-genre-classification/misc/figures/zcr.jpg"/>
</p>


**3.1.3 Temo**
Tempo is the speed or pace at which a passage of music is played. It is measured in beats per minute.                     

### 3.2 Frequency Domain Features

**3.2.1 Spectral Centroid**
Spectral Centroid is a measure which indicates the centre of mass of the spectrum i.e., the frequency around which most of the spectral energy is centered.  Spectral centroid can be calculated as:  
<p align="center">
    <img src="Music-genre-classification/misc/figures/centroid.jpg"/>
</p>


**3.2.2 Spectral Contrast**
Spectral contrast is the difference between the maximum and minimum magnitudes in the frequency domain. Higher values represent narrow-band signals and lower values represent broad-band noise.   

**3.2.3 Spectral Rolloff**   
Spectral rolloff is the frequency value below which a given percentage (85% by default) of the total energy in the spectrum lies. Spectral rolloff is given by:  
<p align="center">
    <img src="Music-genre-classification/misc/figures/rolloff.jpg"/>
</p>

  
**3.2.4 Mel Frequency Cepstrum Coefficients (MFCC)**  
MFCC features are based on Short Time Fourier Transforms (STFT). First, STFT is calculated with n_fft window size (2048 by default), hop length (512 by default) and a Hann window. Next power spectrum is computed and applied to triangular MEL filter banks. Finally, the discrete cosine transform of the logarithm of all filter energies is calculated to obtain MFCCs. 
Here we take the first 5 coefficients averaged over all frames.    

**Note:** For this project, all the features are calculated framewise for the default frame length of 2048 and hop length of 512 and the average of the values across all frames is used as the representative feature of the audio clip.  

## 4. Evaluation  
We implement different models for time domain features, frequency domain features and all the features combined. We experiment with interaction terms to check their performances. We demonstrate the performance of neural networks for different train-test splits. All our analysis is carried out on [IBM SPSS® Statistics software](https://www.ibm.com/analytics/spss-statistics-software). Our analysis considers the relative importance of features, multicollinearity, pseudo R-squared values and accuracy of predictions.

### 4.1 Multinomial Logistic Classification  
We have evaluated four multinomial logistic regression models, a time domain model, a frequency domain model, combined features model and a significant features model.     
Table 1 shows that all three of our models perform better than their respective null models (which has intercept only value and considers all other weights to be zero). We observed that the -2\*Log Likelihood values of all the models are lower than that of the null model and a value of 0.000 (p<0.001) in the significance column indicates that the final model performs better than the null model.
<p align="center">
    <img src="Music-genre-classification/misc/tables/table1.jpg"/>
</p>


**4.1.1 Time Domain Model**  
The time domain model considers the three time domain features root mean square energy (RMSE), zero crossing rate (ZCR) and tempo.  We observed that the collinearity between the time domain features is minimal. The time domain model achieves a top accuracy of 66.5%.  
<p align="center">
    <img src="Music-genre-classification/misc/tables/table2.jpg"/>
</p>


**4.1.2 Frequency Domain Model**
Frequency domain model takes into account 5 MFCC features, spectral centroid, spectral contrast and spectral rolloff. The frequency domain model suffers from multicollinearity problems. Spectral centroid and spectral rolloff have very high collinearity of -0.965 and MFCC1 & MFCC2 have high collinearity of 0.732.  However, it still achieves a high accuracy of 95%.  
<p align="center">
    <img src="Music-genre-classification/misc/tables/table3.jpg"/>
</p>


**4.1.3 Combined Features Model**
Combined features model includes all the features from both time and frequency domains. We found high collinearity between spectral centroid & spectral rolloff, MFCC1 & MFCC2, spectral rolloff and ZCR and spectral centroid and ZCR.  
<p align="center">
    <img src="Music-genre-classification/misc/figures/correlations.jpg"/>
</p>
In terms of accuracy and R-squared values, the combined features model performs the best among the 3 models. 
<p align="center">
    <img src="Music-genre-classification/misc/figures/r_squares.jpg"/>
</p>
The confusion matrix in table 4 for the combined model proves that it achieves the highest accuracy of 97.8%.
<p align="center">
    <img src="Music-genre-classification/misc/tables/table4.jpg"/>
</p>

**4.1.4 Best Features Model**
The likelihood ratio test of the combined features model shows that not all features are significant. Only the first 4 MFCC features, ZCR, RMSE and tempo have a significance value of 0.000 (less than a p value of 0.001). However, due to the high collinearity between MFCC1 and MFCC2, we have only considered MFCC1 for our best model. Hence, our best features model has MFCC1, MFCC3, MFCC4, ZCR, RMSE and tempo as independent variables.   
<p align="center">
    <img src="Music-genre-classification/misc/tables/table5.jpg"/>
</p>
The best features model which only uses about half the number of total features achieves a comparable accuracy of 91.5%. 
<p align="center">
    <img src="Music-genre-classification/misc/tables/table6.jpg"/>
</p>

**4.1.5 Forward and Backward Stepwise Regression using Interaction terms** 
We performed forward and backward stepwise regression using interactions terms with top 4 features MFCC 4, MFCC 3, MFCC 1 and ZCR. We got an accuracy of 94% using forward step regression and 91.1% using backward step regression.  

## 4.2 Multilayer Perceptron (MLP)
We have evaluated the same four MLP models, a time domain model, a frequency domain model, combined features model and a significant features model.   

**4.2.1 Time Domain Model**
We chose the minimum number of neurons to be 1 and maximum number of neurons to be 500. We used gradient descent optimization technique for our MLP. We tried multiple train-test splits such as 60-40, 70-30, 80-20 and 90-10 and multiple learning rates such as 0.4, 0.1 and 0.01 to optimise the model accuracy. We ran each of the variations five times as SPSS split the data randomly, and calculated the average accuracy for each train-test split.  
From the above experiments, we found that a train-test split of 60-40 and learning rate of 0.4 gave the best accuracy of 71.5%.   
<p align="center">
    <img src="Music-genre-classification/misc/figures/mlp_td.jpg"/>
</p>
<p align="center">
    <img src="Music-genre-classification/misc/tables/table7.jpg"/>
</p>

**4.2.2 Frequency Domain Model**
With an same experimental setup as the time domain model, we found that a train-test split of 90-10 and learning rate of 0.4 gave the best accuracy of 96.4%.   
<p align="center">
    <img src="Music-genre-classification/misc/figures/mlp_fd.jpg"/>
</p>
<p align="center">
    <img src="Music-genre-classification/misc/tables/table8.jpg"/>
</p>

**4.2.3 Combined Features Model**
With the same experimental steup again for the combined features model, we found that a train-test split of 90-10 and learning rate of 0.4 gave the best accuracy of 97.9%.  
<p align="center">
    <img src="Music-genre-classification/misc/figures/mlp_c.jpg"/>
</p>
<p align="center">
    <img src="Music-genre-classification/misc/tables/table9.jpg"/>
</p>
 
**4.2.4 Best Features Model**
We have taken the best six features from the combined model. The normalized importance graph shows an example for one of the iterations for the best train-test split and learning rate from the combined feature model. We used this graph to extract the most occuring six features and they were MFCC 4, MFCC 3, MFCC 1, Zero Crossing Rate, RMSE and Rolloff.    
<p align="center">
    <img src="Music-genre-classification/misc/figures/importance.jpg"/>
</p>

In order to predict the number of minimal features required for good accuracy, we experimented by running models with top 3, top 4, top 5 and top 6 features with a learning rate of 0.4 using gradient descent optimization. We ran each of the variations five times and averaged the accuracy for each train-test split.  
From the above experiments, we found that using Top five features gave comparable accuracy of 95.3% to the combined feature model with accuracy of 97.9%.  
<p align="center">
    <img src="Music-genre-classification/misc/figures/mlp_reduced_features.jpg"/>
</p>
<p align="center">
    <img src="Music-genre-classification/misc/tables/table10.jpg"/>
</p>

## 5. Results
Table 11 shows the percentage accuracy of prediction for all the models we have analysed. As we can see, the frequency domain features give better results than time domain features. The highest accuracy is achieved by the combination of both frequency and time domain features. 
<p align="center">
    <img src="Music-genre-classification/misc/tables/table11.jpg"/>
</p>
 
## 6. Conclusion
We have used Multinomial Logistic Classification and Multi-layer perceptron to classify music genres. Frequency domain features performed than time domain features, and the combined features model performed the best. We have reduced the number of features by half while still maintaining high accuracy. 
 
### References
[1] http://marsyas.info/downloads/datasets.html GTZAN dataset  
[2] G. Tzanetakis and P. Cook, "Musical genre classification of audio signals," in IEEE Transactions on Speech and Audio Processing, vol. 10, no. 5, pp. 293-302, July 2002.   
[3] https://librosa.github.io/ Librosa python package   
[4] Davis, Stan and Paul Mermelstein. “Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Se.” (1980).  
[5] https://arxiv.org/pdf/1804.01149.pdf Hareesh Bahuleyan, “Music Genre Classification using Machine Learning Techniques”   
[6] http://cs229.stanford.edu/proj2018/report/21.pdf  
 
### Contributors
1. Ashwin Telagimathada Ravi - [https://www.linkedin.com/in/ashwin-tr/] (https://www.linkedin.com/in/ashwin-tr/)  
2. Vignesh Muthuramalingam - [https://www.linkedin.com/in/vignesh-muthuramalingam/] (https://www.linkedin.com/in/vignesh-muthuramalingam/)  
3. Vineeth Rajesh Ellore - [https://www.linkedin.com/in/vineethellore/](https://www.linkedin.com/in/vineethellore/)  

