# Impact of Noise over EINNs and Feature Importance over EINNs

## Abstract
Machine Learning and Deep Learning models are increasingly being used to forecast epidemic trends and help in efforts in containing
them. These models are extremely accurate in the short term but fail to be accurate in forecasting long term trends. However, re-
cently some researchers embedded the knowledge of physical laws that govern a given data-set in the learning process and came up
with Physics-Informed Neural Networks (PINNs), these have been successful in predicting long term trends. Taking inspiration from
this, Epidemiologically-Informed Neural Networks (EINNs) were designed for epidemics. EINNs use time series data from SEIRM
model which is a famous ODE based model. For epidemics the reported data is usually lesser than the actual
number of infected people due to asymptomatic cases or negligence in testing or human errors. This calls for a model that can predict
robustly even with inaccurate data, hence we evaluated impact of noise on synthetic SEIRM components. We find that EINNs are
robust against the noise. Also, we identify ’Negative increment’ and ’change in retail and recreation’ as the two most important
features to focus more efforts on maintaining quality of those.


## Requirements
Use the package manager [conda](https://docs.conda.io/en/latest/) to install required Python dependencies. Note: We used Python 3.7.

```bash
conda env create -f requirements.yml
```


## Training

The following command will train and predict for all regions from epidemic week 202036 to 202109:

```bash
python main.py --region AL --dev cpu --exp 400 --start_ew 202036 --end_ew 202109 --step 2 --noise D --stdev 0.5
```

You can set up your own model hyperparameter values (e.g. learning rate, loss weights) in the file ```./setup/EINN-params.json```.

## Pre-print paper of Original Paper

Implementation of the paper "EINNs: Epidemiologically-Informed Neural Networks."

Authors: Alexander Rodríguez, Jiaming Cui, Naren Ramakrishnan, Bijaya Adhikari, B. Aditya Prakash

Pre-print: [https://arxiv.org/abs/2202.10446](https://arxiv.org/abs/2202.10446)


## Contact:

If you have any questions about the code, please contact Priyal Chhatrapati, Shubham Agarwal and Divya Umapathy at pchhatrapati3[at]gatech[dot]edu


