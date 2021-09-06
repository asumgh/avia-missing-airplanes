# Missed Airplanes (https://cups.mail.ru/)

## Table of contents

<!--ts-->

   * [Install packages and dependencies](#1-Install-packages-and-dependencies)
   * [Project structure](#2-Project-structure)
   * [Task](#3-Task)
   * [Data description](#4-Data-description)
   * [Augmentation](#5-Augmentation)
   * [Learning](#6-Learning)
   * [Result](#7-Result)

<!--te-->

## 1. Install packages and dependencies
```bash
  pip install -r requirements.txt
  pip install -e .
```
## 2. Project structure

```bash
.
├── 7 folds current model                                                                               <-- First folder with models
│   ├── You should install models
│       in this catalog from 
│       https://drive.google.com/drive/folders/1REAS6CUAllJ7HyDDP3Xf9sprExrJUkhI?usp=sharing                <-- Link for download models
├── mean acc and f1                                                                                     <-- Second folder with models
│   ├── You should install models
│   │    in this catalog from
│   │    https://drive.google.com/drive/folders/1zvM4BXmOvuhwQM6dEcIieLdcZ4UnOx-n?usp=sharing               <-- Link for download models
│   ├── src                                                                                                 <-- Main functions
│   │   ├── __init__.py                                                                                         <-- Initialization
│   │   ├── augment.py                                                                                          <-- Audmentation functions
│   │   ├── dataset.py                                                                                          <-- Dataset functions
│   │   ├── global_var.py                                                                                       <-- Global variables
│   │   └── modeling.py                                                                                         <-- Train loop
│   └── Avia_base.ipynb                                                                                     <-- Training file
├── model full                                                                                          <-- Third folder with models
│   ├── You should install models
│   │   in this catalog from
│   │   https://drive.google.com/drive/folders/1nShNy0YlmQNzUc9XjBCmipqAPkW_Wtma?usp=sharing                <-- Link for download models
│   ├── src                                                                                                 <-- Main functions
│   │   ├── __init__.py                                                                                         <-- Initialization
│   │   ├── augment.py                                                                                          <-- Audmentation functions
│   │   ├── dataset.py                                                                                          <-- Dataset functions
│   │   ├── global_var.py                                                                                       <-- Global variables
│   │   └── modeling.py                                                                                         <-- Train loop
│   └── Avia_base.ipynb                                                                                     <-- Training file
├── Avia_base.ipynb                                                                                     <-- Training file with making prediction
├── README.md
├── requirements.txt                                                                                    <-- Description of dependencies
└── setup.py                                                                                            <-- Building python-packages file
```
## 3. Task 
In this competition we are searching missed airplanes in satellite images. It is classification task.

## 4. Data description
Training:
  * 31080 images
  * 7899 airplanes on it
  * 40 are generated
Test:
  * 101000 images

## 5. Augmentation
I used albumentation with these components:
  * One of: HorizontalFlip, VerticalFlip, RandomRotate90
  * One of: HueSaturationValue, RandomGamma, RandomBrightnessContrast
  * ShiftScaleRotate

## 6. Learning
It is blending three different-parameters models and each learn on 7 stratified folds.
Each model is ResNet18(https://arxiv.org/abs/1512.03385)

## 7. Result
Public score = 1.0007232982844492
Privat score = 1.0009728024369016