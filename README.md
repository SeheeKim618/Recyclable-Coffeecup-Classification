# Recyclable-coffeecup-classification
## Project
### Product
![conv_image_80](https://user-images.githubusercontent.com/76892271/200024668-6f8792c0-398d-4854-a30a-3077bcb037ca.png)

### Model flow
![conv_image_80](https://user-images.githubusercontent.com/76892271/200021532-f5956ae0-0060-48be-a561-a8222cd02dee.png)


## Quick Start
### Data preparation
In our experiments, we use custom dataset. The datasets should be put in data, respecting the following tree directory:
```
${ROOT}
|-- data
`-- |-- coffeecup
    `-- |-- train
        |   |-- plastic
        |   |-- paper
        |   |-- paper_in
        |   |-- waste
        `-- valid
            |-- plastic
            |-- paper
            |-- paper_in
            |-- waste
        `-- test
            |-- plastic
            |-- paper
            |-- paper_in
            |-- waste
```

### Training and Testing
For training:
```
python train.sh
```
For testing:
```
python infer.sh
```
### Classification Model
 * Resnet [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)]
