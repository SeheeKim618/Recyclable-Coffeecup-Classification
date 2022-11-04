# Recyclable-coffeecup-classification
This recyclable coffee cup classification code was used at the KB Kookmin Bank Software Competition. 
## Project Description
We created an automated collecting system that detects recyclable coffee cups in real-time.
### Product
![conv_image_80](https://user-images.githubusercontent.com/76892271/200024668-6f8792c0-398d-4854-a30a-3077bcb037ca.png)

### Model flow
We used the pre-trained ResNet18 as the backbone.
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
This is our dataset examples:
![conv_image_80](https://user-images.githubusercontent.com/76892271/200028376-0fc42439-e1c8-496e-a7ce-7f50916d6f7b.png)

### Training and Testing
For training:
```
python train.sh
```
For testing:
```
python infer.sh
```
## References
 * ResNet [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)]
