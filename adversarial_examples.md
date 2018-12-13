# TL;DR
* Early standard techniques (section "Standard techniques") to create adversarial examples can often be bypassed via simple image transformations (i.e. change in contrast, lighting conditions, noise reduction, etc).
* Defenses and detection techniques targeting noise (introduced by methods mentioned in section "Standard techniques") can probably bypassed via adversarial deformations and harmonic adversarial attacks.
* Sometimes we see a bypass technique which renders a whole class of defenses useless at once.
  * "Synthesizing Robust Adversarial Examples" creates adversarial examples for the physical world. Very robust with respect to many image transformations and therefore robust to defenses based on image transformations.
  * "Obfuscated gradients give a false sense of security: circumventing defenses to adversarial examples" bypasses many defenses (those which based on obfuscated gradients) presented at ICLR 2018. 
  * "Adversarial examples are not easily detected: Bypassing ten detection methods" bypasses another bunch of defenses.
* The adversarial patch is an interesting development. It does not try to hide but is quiet robust.

# Techniques to create adversarial examples 

## Standard techniques

The following techniques produce adversarial examples by introducing noise.

| Method | Paper | Year | Notes |
|-----|-----|---|---|         
| L-BFGS | Intriguing properties of neural networks ([pdf](https://arxiv.org/pdf/1312.6199)])| 2013 | Szegedy, Goodfellow, First paper about adversarial examples. <br><br> "The same perturbation can cause a different network, that was trained on a different subset of the dataset, to misclassify the same input." <br><br> "The existence of the adversarial negatives appears to be in contradiction with the network’s ability to achieve high generalization performance. Indeed, if the network can generalize well, how can it be confused by these adversarial negatives, which are indistinguishable from the regular examples?" | 
| FGSM | Explaining and harnessing adversarial examples ([pdf](https://arxiv.org/pdf/1412.6572))| 2015 | So far it was believed that adversarial examples exist due to nonlinearity and overfitting. Explanation here: it's due to their linear nature. <br><br> Faster than L-BFGS.  |
| DeepFool | DeepFool: a simple and accurate method to fool deep neural networks ([pdf](https://arxiv.org/pdf/1511.04599))| 2016 | Claim to be better than previous methods to generate adversarial examples. "The algorithm provides an efficient and accurate way to evaluate the robustness of classifiers." |
| C&W | Towards evaluating the robustness of neural networks ([pdf](https://arxiv.org/pdf/1608.04644))| 2017 | Claim better performance than FGSM. |

## Adversarial deformations 

The following papers create adversarial examples via deformations instead of noise.

| Paper | Year | Notes |
|----|---|---|
| Spatially transformed adversarial examples ([pdf](https://arxiv.org/pdf/1801.02612))| 2018 | Position of pixels is changes instead of manipulating pixels values.|
| ADef: an Iterative Algorithm to Construct Adversarial Deformations ([pdf](https://arxiv.org/pdf/1804.07729))| 2018 | Apply small deformations to the image. | 

## Other

| Paper | Year | Notes |
|----|---|---|
| Harmonic Adversarial Attack Method ([pdf](https://arxiv.org/pdf/1807.10590))| 2018 | Noise produces a lot of edges. Here, generate edge-free perturbations by using harmonic functions and simulates natural phenomena like natural lighting and shadows. Laplacian edge detector cannot detect edges.<br><br> Hence, bypassing detectors based on noise analysis. |   

## Physical world 

| Paper | Year | Notes |
|----|---|---|
| Adversarial examples in the physical world ([pdf](https://arxiv.org/pdf/1607.02533))| 2017 | "Up to now, all previous work has assumed a threat model in which the adversary can feed data directly into the machine learning classifier" <br><br> "This paper shows that even in such physical world scenarios, machine learning systems are vulnerable to adversarial examples. We demonstrate this by feeding adversarial images obtained from a cell-phone camera to an ImageNet Inception classifier and measuring the classification accuracy of the system."|
| Synthesizing Robust Adversarial Examples ([pdf](https://arxiv.org/pdf/1707.07397))| 2018 | (see section "Bypassing / Defenses") |
| Robust Physical-World Attacks on Deep Learning Models ([pdf](https://arxiv.org/pdf/1707.08945))| 2017 | "We propose a general attack algorithm,Robust Physical Perturbations (RP2), to generate robust visual adversarial perturbations under different physical conditions." |
| Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition ([pdf](https://www.ece.cmu.edu/~lbauer/papers/2016/ccs2016-face-recognition.pdf))| 2016 | (see section "Similar methods to fool image classifiers") |

# Defenses 

Most of the defenses mentioned in the table below have been bypassed by one of these 
* Synthesizing Robust Adversarial Examples
* Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples 

See also:
* Are adversarial examples inevitable? (section 1.1)
* Synthesizing Robust Adversarial Examples (section 4.2)

| Paper | Year | Notes |
|----|---|---|
| Distillation as a defense to adversarial perturbations against deep neural networks ([pdf](https://arxiv.org/pdf/1511.04508))| 2015 | "we introduce a defensive mechanism called defensive distillation to reduce the effectiveness of adversarial samples" |
| Efficient Defenses Against Adversarial Attacks ([pdf](https://arxiv.org/pdf/1707.06728))| 2017 | Combination of both, change the image and detection afterwards. <br><br> "When the model uses the proposed defense, the perturbation necessary for misclassification is much larger, making the attack detectable and, in some cases, turning the images into nonsense" |
| Thermometer Encoding: One Hot Way To Resist Adversarial Examples ([pdf](https://openreview.net/pdf?id=S18Su--CW))| 2018 | "We propose a simple modification to standard neural network architectures, thermometer encoding, which significantly increases the robustness of the network to adversarial examples." |
| Countering Adversarial Images using Input Transformations ([pdf](https://arxiv.org/pdf/1711.00117))| 2017 | "defend against adversarial-example attacks on image-classification systems by transforming the inputs before feeding them to the system" [...] "The strength of those defenses lies in their non-differentiable nature and their inherent randomness" <br><br> (can this be bypassed via "Obfuscated gradients give a false sense of security: circumventing defenses to adversarial examples"?) |
| Stochastic Activation Pruning for Robust Adversarial Defense ([pdf](https://arxiv.org/pdf/1803.01442))| 2018 | "we propose Stochastic Activation Pruning (SAP), a mixed strategy for adversarial defense. SAP prunes a random subset of activations" |
| Mitigating Adversarial Effects Through Randomization ([pdf](https://arxiv.org/pdf/1711.01991))| 2017 | "we use two randomization operations: random resizing, which resizes the input images to a random size, and random padding, which pads zeros around the input images in a random manner" |
| Pixeldefend: Leveraging generative models to understand and defend against adversarial examples ([pdf](https://arxiv.org/pdf/1710.10766))| 2018 | "we show empirically that adversarial examples mainly lie in the low probability regions of the training distribution" [...] "a new approach that purifies a maliciously perturbed image by moving it back towards the distribution seen in the training data. The purified image is then run through an unmodified classifier, making our method agnostic to both the classifier and the attacking method" |
| Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models ([pdf](https://arxiv.org/pdf/1805.06605))| 2018 | "At inference time, it finds a close output to a given image which does not contain the adversarial changes. This output is then fed to the classifier." |

# Detection 

Via a second neural network

| Paper | Year | Notes |
|----|---|---|
| Adversarial and Clean Data Are Not Twins ([pdf](https://arxiv.org/pdf/1704.04960))| 2017 | "we show that we can build a simple binary classifier separating the adversarial apart from the clean data with accuracy over 99%" |
| On the (Statistical) Detection of Adversarial Examples ([pdf](https://arxiv.org/pdf/1702.06280))| 2017 | "we show that they are not drawn from the same distribution than the original data, and can thus be detected using statistical tests" |
| On Detecting Adversarial Perturbations ([pdf](https://arxiv.org/pdf/1702.04267))| 2017 | "we propose to augment deep neural networks with a small "detector" subnetwork" |
| MagNet: a two-pronged defense against adversarial examples ([pdf](https://arxiv.org/pdf/1705.09064))| 2017 | "MagNet includes one or more separate detector networks and a reformer network" |

PCA to detect statistical properties

| Paper | Year | Notes |
|----|---|---|
| Dimensionality Reduction as a Defense against Evasion Attacks on Machine Learning Classifiers ([pdf](https://pdfs.semanticscholar.org/b05e/86841ca65f4ba483b04e465fd54984ad6306.pdf))| 2017 | "dimensionality reduction via Principal Component Analysis to enhance the resilience of machine learning" <br><br> "our key findings are that the defenses are (i) effective [...] (ii) applicable across a range of ML classifiers, including Support Vector Machines and Deep Neural Networks" |
| Early Methods for Detecting Adversarial Images ([pdf](https://arxiv.org/pdf/1608.00530))| 2017 | "We deploy three methods to detect adversarial images." [...] "Our best detection method reveals that adversarial images place abnormal emphasis on the lower-ranked principal components from PCA." |
|Adversarial Examples Detection in Deep Networks with Convolutional Filter Statistics ([pdf](https://arxiv.org/pdf/1612.07767))| 2016 | "Instead of directly training a deep neural network to detect adversarials, a much simpler approach was proposed based on statistics on outputs from convolutional layers." [...] "The resulting classifier is non-subdifferentiable, hence creates a difficulty for adversaries to attack by using the gradient of the classifier"|

Other

| Paper | Year | Notes |
|----|---|---|
| Detecting Adversarial Samples from Artifacts ([pdf](https://arxiv.org/pdf/1703.00410))| 2017 | "looking at Bayesian uncertainty estimates" [...] "and by performing density estimation"|
| Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality ([pdf](https://arxiv.org/pdf/1801.02613))| 2018 | "we show that a potential application of LID is to distinguish adversarial examples, and the preliminary results show that it can outperform several state-of-the-art detection measures by large margins for five attack strategies" |

# Bypassing

## Defenses  

After some techniques to destroy adversarial examples have been introduced papers were published to bypass these defenses.

| Paper | Year | Notes |
|----|---|---|
| Synthesizing Robust Adversarial Examples ([pdf](https://arxiv.org/pdf/1707.07397))| 2018 | Bypass most defenses based on image transformations. <br><br> "We demonstrate the existence of robust 3D adversarial objects, and we present the first algorithm for synthesizing examples that are adversarial over a chosen distribution of transformations." <br><br> "robust to noise, distortion, and affine transformation" <br><br> 3D turtle is classified as rifle. https://youtu.be/XaQu7kkQBPc |
| Obfuscated gradients give a false sense of security: circumventing defenses to adversarial examples ([pdf](https://arxiv.org/pdf/1802.00420))| 2018 | Again, a popular defense technique (i.e. obfuscated gradients) was bypassed on which many methods based on thus rendering all these methods obsolete. |
| Defensive Distillation is Not Robust to Adversarial Examples ([pdf](https://arxiv.org/pdf/1607.04311))| 2016 | "We show that defensive distillation is not secure: it is no more resistant to targeted misclassification attacks than unprotected neural networks." <br><br> Bypass for "Distillation as a defense to adversarial perturbations against deep neural networks" |

## Detection 

| Paper | Year | Notes |
|----|---|---|
| Adversarial examples are not easily detected: Bypassing ten detection methods ([pdf](https://arxiv.org/pdf/1705.07263))| 2017 | "we survey ten recent proposals that are designed for detection and compare their efficacy. We show that all can be defeated by constructing new loss functions. We conclude that adversarial examples are significantly harder to detect than previously appreciated" |
| MagNet and "Efficient Defenses Against Adversarial Attacks" are Not Robust to Adversarial Examples ([pdf](https://arxiv.org/pdf/1711.08478))| 2017 | "MagNet and "Efficient Defenses..." were recently proposed as a defense to adversarial examples. We find that we can construct adversarial examples that defeat these defenses with only a slight increase in distortion." |

# Adversarial examples in other domains 

| Paper | Year | Notes |
|----|---|---|
| Audio Adversarial Examples: Targeted Attacks on Speech-to-Text ([pdf](https://arxiv.org/pdf/1801.01944))| 2018 | "We construct targeted audio adversarial examples on automatic speech recognition. Given any audio waveform, we can produce another that is over 99.9% similar, but transcribes as any phrase we choose" |

# Critism 

| Paper | Year | Notes |
|----|---|---|
| No need to worry about adversarial examples in object detection in autonomous vehicles ([pdf](https://arxiv.org/pdf/1707.03501))| 2017 | "even if adversarial perturbations might cause a deep neural network detector to misdetect a stop sign image in a physical environment when the photo is taken from a particular range of distances and angles, they cannot reliably fool object detectors across a scale of different distances and angles" |

# Theoretical work 

| Paper | Year | Notes |
|----|---|---|
| Are adversarial examples inevitable? ([pdf](https://arxiv.org/pdf/1809.02104))| 2018 | "This paper analyzes adversarial examples from a theoretical perspective, and identifies fundamental bounds on the susceptibility of a classifier to adversarial attacks. We show that, for certain classes of problems, adversarial examples are inescapable." |
| Adversarial Spheres ([pdf](https://arxiv.org/pdf/1801.02774v1.pdf)) | 2018 | "study a simple synthetic dataset of classifying between two concentric high dimensional spheres" <br><br> "we prove that any model which misclassifies a small constant fraction of a sphere will be vulnerable to adversarial perturbations" <br><br> Title of version v3 of this paper is "The Relationship Between High-Dimensional Geometry and Adversarial Examples" (see [arxiv](https://arxiv.org/abs/1801.02774) for history)|

# Similar methods to fool image classifiers

| Paper | Year | Notes |
|----|---|---|
| One pixel attack for fooling deep neural networks ([pdf](https://arxiv.org/pdf/1710.08864))| 2017 | For very small images it is sufficient to modify just one pixel. Someone might guess this is just a pixel/sensor error but not an attack. |
| Adversarial Patch ([pdf](https://arxiv.org/pdf/1712.09665))| 2017 | "We present a method to create universal, robust, targeted adversarial image patches in the real world." https://github.com/tensorflow/cleverhans/tree/master/examples/adversarial_patch |
| Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition ([pdf](https://www.archive.ece.cmu.edu/~lbauer/papers/2016/ccs2016-face-recognition.pdf))| 2016 | "We define and investigate a novel class of attacks: attacks that are physically realizable and inconspicuous, and allow an attacker to evade recognition or impersonate another individual" |





