# Introduction

Like software systems also machine learning can suffer from security weaknesses. This repository contains some resources to provide an overview.

## Possible Security Issues in Machine Learning

### Poisoning

In a poisoning attack an adversary can insert carefully crafted examples into the training data. Hence, this attack happens at training time. First, such an attack could degrade the performance of a machine learning model (the adversary targets the availability of the model). Second, the adversary could use this technique to inject a backdoor.

#### Resources

* [Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning](https://arxiv.org/pdf/1712.05526), 2017
* [Manipulating Machine Learning: Poisoning Attacks and Countermeasures for Regression Learning](https://arxiv.org/pdf/1804.00308.pdf), 2018
* [Data poisoning attacks against online learning](https://arxiv.org/abs/1808.08994), 2018
* [Why do adversarial attacks transfer? explaining transferability of evasion and poisoning attacks](https://arxiv.org/abs/1809.02861), 2018

### Evasion

An evasion attack happens at test time. Here, an instance that would be classified correctly without modification, will be misclassified when small modifications are added by the adversary. A well know example of evasion attacks are adversarial examples. An adversary adds small perturbations to an image which are invisible to a human but will fool the image classifier which will misclassify them into a category that can be chosen by the adversary.

#### Resources 

* [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199), 2014
* [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), 2014
* [Evasion Attacks against Machine Learning at Test Time](https://arxiv.org/abs/1708.06131), 2017
* [Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods](https://arxiv.org/abs/1705.07263), 2017

### Inversion Attacks

In inversion attacks an adversary tries to extract useful information from a machine learning model like training data that was used for the training. Due to the fact that machine learning is used in more and more privacy sensitive applications an adversary could learn very sensitive data of individuals.

#### Resources

* [Privacy in Pharmacogenetics: An End-to-End Case Study of Personalized Warfarin Dosing](https://www.usenix.org/system/files/conference/usenixsecurity14/sec14-paper-fredrikson-privacy.pdf), 2014
* [Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures](https://www.cs.cmu.edu/~mfredrik/papers/fjr2015ccs.pdf), 2015
* [Membership Model Inversion Attacks for Deep Networks](https://arxiv.org/abs/1910.04257), 2019
* [The Secret Revealer: Generative Model-Inversion Attacks Against Deep Neural Networks](https://arxiv.org/abs/1911.07135), 2019

# Examples

This repositories contains some examples of attacks in the folder `example`. Examples are:

* Create adversarial examples
* Model stealing
* Model inversion attacks

To run the examples it is recommended to create a virtual environment first and install all required packages in that environment:

    virtualenv -p python3 venv
    source venv/bin/activate
    pip3 install -r requirements.txt

# Resources

* [TensorFlow Privacy](https://github.com/tensorflow/privacy)
* [PySyft](https://github.com/OpenMined/PySyft)
* [Encrypted Training Demo on MNIST](https://blog.openmined.org/encrypted-training-on-mnist/)

