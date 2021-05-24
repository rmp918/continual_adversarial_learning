# continual_adversarial_learning
## Introduction

   With the mature development of image recognition technology, computer vision has also obtained impressive results in research through many well-known deep learning networks. However,  in recent years, there have been problems related to image interference with model identification. The major problem is how to prevent people from using this technology to damage models or users when computer vision technology is commercialized or popularized in the future. From the perspective of defense, this research proposed a defense model, using the concepts of adversarial training and continual learning proposed by Madry to establish an effective and flexible model.

   We used this research to propose a CMAT model as our defense model against current well-known attacks. This research explored whether CMAT is applicable to defense networks through visualization and experimental data. This research is also the first paper in this field that used continual learning with basic defense techniques. I hope that the results of this paper could be used as an experimental reference for future related research. 
https://etd.lis.nsysu.edu.tw/ETD-db/ETD-search-c/view_etd?URN=etd-1118120-170138

## Implementation

   Part of the codes in this repo are borrowed/modified from Steven C. Y. Hung, Cheng-Hao Tu, Cheng-En Wu, Chien-Hung Chen, Yi-Ming Chan, and Chu-Song Chen, "Compacting, Picking and Growing for Unforgetting Continual Learning," Thirty-third Conference on Neural Information Processing Systems, NeurIPS 2019
https://github.com/ivclab/CPG

---

## Dependencies
    Python>=3.6
    PyTorch>=1.0
    tqdm
---
## CMAT Structure
![image](https://github.com/rmp918/continual_adversarial_learning/blob/main/CMAT_structure.png)

   CMAT is adapted from the architecture of CPG. The biggest difference is to change the way of training, from the normal training of CPG to adversarial training. 
Through the original CPG architecture, we hope that when the model learns multiple tasks, it can use what has been learned before to enhance the performance of the following tasks as much as possible. 


## Result

![image](https://github.com/rmp918/continual_adversarial_learning/blob/main/Visualization-of-accuracy-on-Cifar-100.png)

   We showed the accuracy over various difficulties on Cifar-100. In this bar chart, the accuracy was gradually decreasing as the step size increases. The simpler the perturbation had the higher accuracy was. It indicated that as the difficulty increases or decreases, the relative accuracy had a negative correlation with the difficulty. This situation was also in line with previous research findings. And comparing CMAT with PackNet, we could see that CMAT also had the better accuracy.

![image](https://github.com/rmp918/continual_adversarial_learning/blob/main/Comparison%20with%20different%20method.png)
   We compared with different method about defense model. All of these result used PGD-10/PGD-40 and perturbation 𝜖=8/255.We found our result and PackNet that they were higher performance than others.PackNet outperformed adversarial training by 9% of accuracy, and at least outperformed ANP-VS by 4% of accuracy; CMAT could be better than PackNet by about 3%. The model we implemented was a relatively simple and basic method of adversarial training. We proved our method(CMAT) was the best model on defense adversarial attack on Cifar-100 dataset.

![image](https://github.com/rmp918/continual_adversarial_learning/blob/main/PackNet%20experiment%20in%20high-resolution%20datasets.png)
![image](https://github.com/rmp918/continual_adversarial_learning/blob/main/CMAT%20experiment%20in%20high-resolution%20datasets.png)
   We can find our model and PackNet that they were good performance about recognizing high-resolution image. We also used F1-score to prove the correctness about our result. Although the performance of the first data set was relatively poor at the beginning, it could have better performance in subsequent datasets. The reason was that continual learning networks such as PackNet and CPG usually do more pruning or compression networks in the first dataset, and then gradually improve its performance. This phenomenon also occurred in recognizing normal pictures (Hung et al., 2019).



## References:

1	Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., and Fergus, R.: ‘Intriguing properties of neural networks’, arXiv preprint arXiv:1312.6199, 2013

2	Eykholt, K., Evtimov, I., Fernandes, E., Li, B., Rahmati, A., Xiao, C., Prakash, A., Kohno, T., and Song, D.: ‘Robust physical-world attacks on deep learning visual classification’, in Editor (Ed.)^(Eds.): ‘Book Robust physical-world attacks on deep learning visual classification’ (2018, edn.), pp. 1625-1634

3	Papernot, N., McDaniel, P., Goodfellow, I., Jha, S., Celik, Z.B., and Swami, A.: ‘Practical black-box attacks against machine learning’, in Editor (Ed.)^(Eds.): ‘Book Practical black-box attacks against machine learning’ (ACM, 2017, edn.), pp. 506-519

4	Papernot, N., McDaniel, P., and Goodfellow, I.: ‘Transferability in machine learning: from phenomena to black-box attacks using adversarial samples’, arXiv preprint arXiv:1605.07277, 2016

5	Tramèr, F., Papernot, N., Goodfellow, I., Boneh, D., and McDaniel, P.: ‘The space of transferable adversarial examples’, arXiv preprint arXiv:1704.03453, 2017

6	Su, D., Zhang, H., Chen, H., Yi, J., Chen, P.-Y., and Gao, Y.: ‘Is robustness the cost of accuracy?--A comprehensive study on the robustness of 18 deep image classification models’, in Editor (Ed.)^(Eds.): ‘Book Is robustness the cost of accuracy?--A comprehensive study on the robustness of 18 deep image classification models’ (2018, edn.), pp. 631-648


7	Hendrycks, D., and Dietterich, T.: ‘Benchmarking neural network robustness to common corruptions and perturbations’, arXiv preprint arXiv:1903.12261, 2019

8	Fawzi, A., Moosavi-Dezfooli, S.-M., and Frossard, P.: ‘Robustness of classifiers: From adversarial to random noise’, in Editor (Ed.)^(Eds.): ‘Book Robustness of classifiers: From 
adversarial to random noise’ (2016, edn.), pp. 1632-1640

9	Ford, N., Gilmer, J., Carlini, N., and Cubuk, D.: ‘Adversarial examples are a natural consequence of test error in noise’, arXiv preprint arXiv:1901.10513, 2019

10	Yu, H., Liu, A., Liu, X., Yang, J., and Zhang, C.: ‘Towards Noise-Robust Neural Networks via Progressive Adversarial Training’, arXiv preprint arXiv:1909.04839, 2019

11	Mallya, A., and Lazebnik, S.: ‘Packnet: Adding multiple tasks to a single network by iterative pruning’, in Editor (Ed.)^(Eds.): ‘Book Packnet: Adding multiple tasks to a single network by iterative pruning’ (2018, edn.), pp. 7765-7773

12	Madry, A., Makelov, A., Schmidt, L., Tsipras, D., and Vladu, A.: ‘Towards deep learning models resistant to adversarial attacks’, arXiv preprint arXiv:1706.06083, 2017

13	Hung, C.-Y., Tu, C.-H., Wu, C.-E., Chen, C.-H., Chan, Y.-M., and Chen, C.-S.: ‘Compacting, picking and growing for unforgetting continual learning’, in Editor (Ed.)^(Eds.): ‘Book 
Compacting, picking and growing for unforgetting continual learning’ (2019, edn.), pp. 13647-13657

14	Schmidt, L., Santurkar, S., Tsipras, D., Talwar, K., and Madry, A.: ‘Adversarially robust generalization requires more data’, in Editor (Ed.)^(Eds.): ‘Book Adversarially robust generalization requires more data’ (2018, edn.), pp. 5014-5026

15	Sun, K., Zhu, Z., and Lin, Z.: ‘Towards understanding adversarial examples systematically: Exploring data size, task and model factors’, arXiv preprint arXiv:1902.11019, 2019

16	Zhong, Z., Jin, L., and Xie, Z.: ‘High performance offline handwritten chinese character recognition using googlenet and directional feature maps’, in Editor (Ed.)^(Eds.): ‘Book High performance offline handwritten chinese character recognition using googlenet and directional feature maps’ (IEEE, 2015, edn.), pp. 846-850

17	Goodfellow, I.J., Shlens, J., and Szegedy, C.: ‘Explaining and harnessing adversarial examples’, arXiv preprint arXiv:1412.6572, 2014

18	Akhtar, N., and Mian, A.: ‘Threat of adversarial attacks on deep learning in computer vision: A survey’, IEEE Access, 2018, 6, pp. 14410-14430

19	Kurakin, A., Goodfellow, I., and Bengio, S.: ‘Adversarial examples in the physical world’, arXiv preprint arXiv:1607.02533, 2016

20	Papernot, N., McDaniel, P., Jha, S., Fredrikson, M., Celik, Z.B., and Swami, A.: ‘The limitations of deep learning in adversarial settings’, in Editor (Ed.)^(Eds.): ‘Book The 
limitations of deep learning in adversarial settings’ (IEEE, 2016, edn.), pp. 372-387

21	Su, J., Vargas, D.V., and Sakurai, K.: ‘One pixel attack for fooling deep neural networks’, IEEE Transactions on Evolutionary Computation, 2019

22	Carlini, N., and Wagner, D.: ‘Towards evaluating the robustness of neural networks’, in Editor (Ed.)^(Eds.): ‘Book Towards evaluating the robustness of neural networks’ (IEEE, 
2017, edn.), pp. 39-57

23	Moosavi-Dezfooli, S.-M., Fawzi, A., and Frossard, P.: ‘Deepfool: A simple and accurate method to fool deep neural networks’, in Editor (Ed.)^(Eds.): ‘Book Deepfool: A simple and 
accurate method to fool deep neural networks’ (2016, edn.), pp. 2574-2582

24	Das, S., and Suganthan, P.N.: ‘Differential evolution: A survey of the state-of-the-art’, IEEE transactions on evolutionary computation, 2010, 15, (1), pp. 4-31

25	Baluja, S., and Fischer, I.: ‘Adversarial transformation networks: Learning to generate adversarial examples’, arXiv preprint arXiv:1703.09387, 2017

26	Athalye, A., Carlini, N., and Wagner, D.: ‘Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples’, arXiv preprint arXiv:1802.00420, 
2018

27	Liu, Q., Li, P., Zhao, W., Cai, W., Yu, S., and Leung, V.C.: ‘A survey on security threats and defensive techniques of machine learning: A data driven view’, IEEE access, 2018, 
6, pp. 12103-12117

28	Nelson, B., Barreno, M., Chi, F.J., Joseph, A.D., Rubinstein, B.I., Saini, U., Sutton, C., Tygar, J., and Xia, K.: ‘Misleading learners: Co-opting your spam filter’: ‘Machine 
learning in cyber trust’ (Springer, 2009), pp. 17-51

29	Papernot, N., McDaniel, P., Wu, X., Jha, S., and Swami, A.: ‘Distillation as a defense to adversarial perturbations against deep neural networks’, in Editor (Ed.)^(Eds.): ‘Book 
Distillation as a defense to adversarial perturbations against deep neural networks’ (IEEE, 2016, edn.), pp. 582-597

30	Tramèr, F., Kurakin, A., Papernot, N., Goodfellow, I., Boneh, D., and McDaniel, P.: ‘Ensemble adversarial training: Attacks and defenses’, arXiv preprint arXiv:1705.07204, 2017


31	Sengupta, S., Chakraborti, T., and Kambhampati, S.: ‘MTDeep: Boosting the security of deep neural nets against adversarial attacks with moving target defense’, in Editor (Ed.)^(Eds.): ‘Book MTDeep: Boosting the security of deep neural nets against adversarial attacks with moving target defense’ (2018, edn.), pp. 

32	Dwork, C.: ‘Differential privacy’, Encyclopedia of Cryptography and Security, 2011, pp. 338-340

33	Abadi, M., Chu, A., Goodfellow, I., McMahan, H.B., Mironov, I., Talwar, K., and Zhang, L.: ‘Deep learning with differential privacy’, in Editor (Ed.)^(Eds.): ‘Book Deep learning with differential privacy’ (ACM, 2016, edn.), pp. 308-318

34	Parisi, G.I., Kemker, R., Part, J.L., Kanan, C., and Wermter, S.: ‘Continual lifelong learning with neural networks: A review’, Neural Networks, 2019, 113, pp. 54-71

35	McClelland, J.L., McNaughton, B.L., and O'Reilly, R.C.: ‘Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory’, Psychological review, 1995, 102, (3), pp. 419

36	Pfülb, B., and Gepperth, A.: ‘A comprehensive, application-oriented study of catastrophic forgetting in dnns’, arXiv preprint arXiv:1905.08101, 2019

37	Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A.A., Milan, K., Quan, J., Ramalho, T., and Grabska-Barwinska, A.: ‘Overcoming catastrophic forgetting in neural networks’, Proceedings of the national academy of sciences, 2017, 114, (13), pp. 3521-3526

38	Zenke, F., Poole, B., and Ganguli, S.: ‘Continual learning through synaptic intelligence’, Proceedings of machine learning research, 2017, 70, pp. 3987

39	Rebuffi, S.-A., Kolesnikov, A., Sperl, G., and Lampert, C.H.: ‘icarl: Incremental classifier and representation learning’, in Editor (Ed.)^(Eds.): ‘Book icarl: Incremental classifier and representation learning’ (2017, edn.), pp. 2001-2010

40	Shin, H., Lee, J.K., Kim, J., and Kim, J.: ‘Continual learning with deep generative replay’, in Editor (Ed.)^(Eds.): ‘Book Continual learning with deep generative replay’ (2017, edn.), pp. 2990-2999

41	Wu, Y., Chen, Y., Wang, L., Ye, Y., Liu, Z., Guo, Y., Zhang, Z., and Fu, Y.: ‘Incremental classifier learning with generative adversarial networks’, arXiv preprint arXiv:1802.00853, 2018

42	Hinton, G.E., and Salakhutdinov, R.R.: ‘Reducing the dimensionality of data with neural networks’, science, 2006, 313, (5786), pp. 504-507

43	Mesnil, G., Dauphin, Y., Glorot, X., Rifai, S., Bengio, Y., Goodfellow, I., Lavoie, E., Muller, X., Desjardins, G., and Warde-Farley, D.: ‘Unsupervised and transfer learning 
challenge: a deep learning approach’, in Editor (Ed.)^(Eds.): ‘Book Unsupervised and transfer learning challenge: a deep learning approach’ (JMLR. org, 2011, edn.), pp. 97-111

44	Rusu, A.A., Rabinowitz, N.C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K., Pascanu, R., and Hadsell, R.: ‘Progressive neural networks’, arXiv preprint arXiv:1606.04671, 2016

45	Krizhevsky, A., and Hinton, G.: ‘Learning multiple layers of features from tiny images’, 2009

46	Deng, L.: ‘The MNIST database of handwritten digit images for machine learning research [best of the web]’, IEEE Signal Processing Magazine, 2012, 29, (6), pp. 141-142

47	Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., and Ng, A.Y.: ‘Reading digits in natural images with unsupervised feature learning’, 2011

48	Simonyan, K., and Zisserman, A.: ‘Very deep convolutional networks for large-scale image recognition’, arXiv preprint arXiv:1409.1556, 2014

49	He, K., Zhang, X., Ren, S., and Sun, J.: ‘Deep residual learning for image recognition’, in Editor (Ed.)^(Eds.): ‘Book Deep residual learning for image recognition’ (2016, edn.), 
pp. 770-778

50	Aljundi, R., Chakravarty, P., and Tuytelaars, T.: ‘Expert gate: Lifelong learning with a network of experts’, in Editor (Ed.)^(Eds.): ‘Book Expert gate: Lifelong learning with a network of experts’ (2017, edn.), pp. 3366-3375

51	Chen, Z., and Liu, B.: ‘Lifelong machine learning’, Synthesis Lectures on Artificial Intelligence and Machine Learning, 2018, 12, (3), pp. 1-207

52	McCann, B., Keskar, N.S., Xiong, C., and Socher, R.: ‘The natural language decathlon: Multitask learning as question answering’, arXiv preprint arXiv:1806.08730, 2018

53	Masse, N.Y., Grant, G.D., and Freedman, D.J.: ‘Alleviating catastrophic forgetting using context-dependent gating and synaptic stabilization’, Proceedings of the National Academy 
of Sciences, 2018, 115, (44), pp. E10467-E10475

54	Madaan, D., Shin, J., and Hwang, S.J.: ‘Adversarial neural pruning with latent vulnerability suppression’, arXiv preprint arXiv:1908.04355, 2019

55	Zhang, H., and Xu, W.: ‘Adversarial Interpolation Training: A Simple Approach for Improving Model Robustness’, 2019

56	Zhu, M., and Gupta, S.: ‘To prune, or not to prune: exploring the efficacy of pruning for model compression’, arXiv preprint arXiv:1710.01878, 2017

57	Mallya, A., Davis, D., and Lazebnik, S.: ‘Piggyback: Adapting a single network to multiple tasks by learning to mask weights’, in Editor (Ed.)^(Eds.): ‘Book Piggyback: Adapting a 
single network to multiple tasks by learning to mask weights’ (2018, edn.), pp. 67-82

58	Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L.: ‘Imagenet: A large-scale hierarchical image database’, in Editor (Ed.)^(Eds.): ‘Book Imagenet: A large-scale 
hierarchical image database’ (Ieee, 2009, edn.), pp. 248-255

59	Bottou, L.: ‘Large-scale machine learning with stochastic gradient descent’: ‘Proceedings of COMPSTAT'2010’ (Springer, 2010), pp. 177-186

60	Pillai, I., Fumera, G., and Roli, F.: ‘F-measure optimisation in multi-label classifiers’, in Editor (Ed.)^(Eds.): ‘Book F-measure optimisation in multi-label classifiers’ (IEEE, 
2012, edn.), pp. 2424-2427

61	Rice, L., Wong, E., and Kolter, J.Z.: ‘Overfitting in adversarially robust deep learning’, arXiv preprint arXiv:2002.11569, 2020

62	Shafahi, A., Najibi, M., Ghiasi, M.A., Xu, Z., Dickerson, J., Studer, C., Davis, L.S., Taylor, G., and Goldstein, T.: ‘Adversarial training for free!’, in Editor (Ed.)^(Eds.): ‘Book Adversarial training for free!’ (2019, edn.), pp. 3358-3369

63	Hendrycks, D., Lee, K., and Mazeika, M.: ‘Using pre-training can improve model robustness and uncertainty’, arXiv preprint arXiv:1901.09960, 2019

64	Zhang, H., Yu, Y., Jiao, J., Xing, E.P., Ghaoui, L.E., and Jordan, M.I.: ‘Theoretically principled trade-off between robustness and accuracy’, arXiv preprint arXiv:1901.08573, 
2019

65	Ribani, R., and Marengoni, M.: ‘A survey of transfer learning for convolutional neural networks’, in Editor (Ed.)^(Eds.): ‘Book A survey of transfer learning for convolutional 
neural networks’ (IEEE, 2019, edn.), pp. 47-57

66	Guo, Y., Shi, H., Kumar, A., Grauman, K., Rosing, T., and Feris, R.: ‘Spottune: transfer learning through adaptive fine-tuning’, in Editor (Ed.)^(Eds.): ‘Book Spottune: transfer 
learning through adaptive fine-tuning’ (2019, edn.), pp. 4805-4814

67	Theagarajan, R., Chen, M., Bhanu, B., and Zhang, J.: ‘Shieldnets: Defending against adversarial attacks using probabilistic adversarial robustness’, in Editor (Ed.)^(Eds.): ‘Book Shieldnets: Defending against adversarial attacks using probabilistic adversarial robustness’ (2019, edn.), pp. 6988-6996

68	Hayes, J., and Danezis, G.: ‘Learning universal adversarial perturbations with generative models’, in Editor (Ed.)^(Eds.): ‘Book Learning universal adversarial perturbations with generative models’ (IEEE, 2018, edn.), pp. 43-49


