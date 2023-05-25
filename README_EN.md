# Multimodal-Emotion-Analysis-Algorithm-Combining-Text-and-Images
multimodal-sarcasm-detection

**[介绍](./README.md)**

**[Introduction ](./README_EN.md)**

****
## Catalogue
* [Foreword](#Foreword)
* [FileStructure](#FileStructure)
* [LibrariesAndVersions](#LibrariesAndVersions)
* [TheModelArchitecture](#TheModelArchitecture)

## Foreword
This project was improved under the premise of reproducing the paper, the address of the original paper：https://aclanthology.org/P19-1239/

Performance：acc:0.917, pre:0.871, rec:0.930, **f1:0.900**, acu:0.956, loss:0.3920
Better than the original paper

## FileStructure
```
.
│  ReadMe.md 
├─codes - Main Running folder
│  │  ImageFeature.py      - Networks that process features of pictures
│  │  ExtractFeature.py    - Networks that deal with category features
│  │  TextFeature.py       - Networks that process text features
│  │  NNManager.py         - Total network framework
│  │  Main.py              - main
│  │  Attention.py         - Attention mechanism
│  │  Function.py          - Static functions and variables
│  │  ImageRegionNet.py    - Generate image vectors
│  │  MyDataSet.py         - Constructing the dataset
│  │  ResNet.py            - ResNet
│  │  DATASET.py           - Enumeration
│  │  DataAugment.py       - Data augmentation
│  │  废弃代码.py
│  │  
│  ├─ablation - Ablation experiment
│  │      DeleteEFNet.py   - Delete early fusion
│  │      DeleteRFNet.py   - Delete representation fusion
│  │      ReplaceEFNet.py  - Image features were used for early fusion
│  │      
│  ├─modelCompare - Comparison of models
│  │      BiLSTMBert.py    - Text features only
│  │ 
├─image - 
│  ├─imageDataSet          - Original image
│  ├─augmentImages         - Image after data augmentation
│  └─imageVector           - Image vector
|
├─class
│     classGloveVector.npy - Pre-trained Glove vectors 
│     classVocab.py3       - Category vocabulary
│     image2class.py3      - augmentImages+imageDataSet Corresponding category
│     image2classBefore.py3- imageDataSet Corresponding category
|
├─text
│     textGloveVector.npy  - Pre-trained Glove vectors 
│     textVocab.py3        - Words vocabulary
│     text.txt             - Text
│     train_text           - Train set 0.8
│     test_text            - Test set 0.1
│     valid_text           - Valid set 0.1
|
├─modelWeights - Pre-trained model parameters
│  │  resnet50.pth         - # https://download.pytorch.org/models/
│  │  
│  └─bert-base-cased       - # https://huggingface.co/bert-base-cased
│          
└─outputs 
   └─****-**-**
      │  ****-**-**.pth    - Model parameters
      │  describe.txt      - logs
      │  logs.py3          - trian test valid lr 
      |    
      └─runs               - TensorBoard   1. cd The current folder 2. tensorboard --logdir=runs 3. Open the URL
```
## LibrariesAndVersions
Win10 + Anaconda
|     Name     |   Version   |
| :----------: | :---------: |
|    python    |    3.8.0    |
|   pytorch    | 1.8.1+cu111 |
|  matplotlib  |    3.4.2    |
| scikit-learn |    1.2.2    |
| transformers |   4.29.1    |
|    pillow    |    9.5.0    |

```
pip install ***==**** -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

## TheModelArchitecture
![image](https://github.com/2573943723/Multimodal-Emotion-Analysis-Algorithm-Combining-Text-and-Images/assets/67378023/33d52f5e-359e-4049-bdc4-3b2e5fc808b0)

