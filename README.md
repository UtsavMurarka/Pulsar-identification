# Pulsar-identification

The [dataset](https://archive.ics.uci.edu/ml/datasets/HTRU2) contains statistical features of signals of pulsar candidates collected by the High Time Resolution Universe Survey (South). The data has a positive:negative class imbalance of 1:10, which makes training models to correctly identify the positives difficult. Therefore, the aim to to decrease the number of false negatives predicted by the models.
The number of false negatives were decreased by 50% by using oversampling and undersampling.

The Data:

The dataset contains statistical features of the integrated pulse profile and the DM-SNR curve.
The statistical features include:
1. Mean
2. Standard Deviation
3. Skewness
4. Kurtosis

Integrated Pulse Profile:
Each pulsar produces a unique pattern of pulse emission known as its pulse profile. It is like a fingerprint of the pulsar. It is possible to identify pulsars from their pulse profile alone. But the pulse profile varies slightly in every period. This makes the pulsar hard to detect. This is because their signals are non-uniform and not entirely stable overtime. However, these profiles do become stable, when averaged over many thousands of rotations.

DM-SNR Curve:
Radio waves emitted from pulsars reach earth after travelling long distances in space which is filled with free electrons. Since radio waves are electromagnetic in nature, they interact with these electrons, this interaction results in slowing down of the wave. The important point is that pulsars emit a wide range of frequencies, and the amount by which the electrons slow down the wave depends on the frequency. Waves with higher frequency are sowed down less as compared to waves higher frequency. i.e. lower frequencies reach the telescope later than higher frequencies. This is called dispersion.

Data Plots:
![Mean of the Integrated Pulse Profile](/Visualization/img/mean_ipp.png)
![Standard Deviation the Integrated Pulse Profile](/Visualization/img/stddev-ipp.png)
![Excess Kurtosis of the Integrated Pulse Profile](/Visualization/img/ek-ipp.png)
![Skewness of the Integrated Pulse Profile](/Visualization/img/sk-ipp.png)

![Mean of the DMSNR Curve](/Visualization/img/mean-dmsnr.png)
![Standard Deviation the DMSNR Curve](/Visualization/img/stddev-dmsnr.png)
![Excess Kurtosis of the DMSNR Curve](/Visualization/img/ek-dmsnr.png)
![Skewness of the DMSNR Curve](/Visualization/img/sk-dmsnr.png)
