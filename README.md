# GAN_Cycle (Pix2Pix)
Cycle Gan 방식으로 PyTorch framework 사용하여 GAN을 진행하는 프로젝트 입니다.

## Implementations

- Pix2PixGAN
- CycleGAN
- GAN_Loss (LSGAN, VanillaGAN)
- Data(Prepare)
- Muti GPU Training

## 프로젝트 구조
```
GAN_CYCLE_PIX2PIX
├─ .gitignore
├─ __README.md
├─ configs # 학습 시 사용할 하이퍼 파라미터, 데이터셋 설정 등 Configuration을 위한 yaml 파일 경로
├─ dataset # Image Data Generator 모듈
├─ models # Detector, Convolution Module 등 구현
│  ├─ generator
│  ├─ discriminator
│  └─ loss
├─ module # 학습을 위한 Pytorch Lightning 모듈
├─ onnx_module # onnx 변형을 위한 모듈
├─ train_gan.py # Detector 학습 스크립트
└─ utils

```

## Requirements
`requirements.txt` 파일을 참고하여 Anaconda 환경 설정 (conda install 명령어)  
`PyYaml`  
`PyTorch`  
`Pytorch Lightning`

## Config Train Parameters

기본 설정값은 ./configs/default_settings.yaml에 정의됩니다.  
Train 스크립트 실행 시 입력되는 CFG 파일로 하이퍼파라미터 및 학습 기법을 설정할 수 있습니다.

[default_settings.yaml](./configs/default_settings.yaml)

    // ./configs/*.yaml 파일 수정
    // ex) cls_frostnet -> default_settings 파라미터를 업데이트 해서 사용
    model : 'Pix2Pix_GAN'
    dataset_name : Cityscapes
    epochs: 500
    data_path : '/mnt/gan_train/'
    save_dir : './saved'
    workers: 4
    ...

## Train Pix2PixGAN

Pix2PixGAN 모델 Train 스크립트 입니다.

    python train_gan.py --cfg configs/gan_pix2pix.yaml

## Test Pix2PixGAN
segmenation 되어있는 데이터를 입력으로 넣어 주었을 경우 생성된 데이터입니다.
학습이 잘 안되는 경우 generator와 discriminator의 capacity 늘려 테스트 하는 것이 좋습니다.
![Pix2Pix](./inference/result/inference.png)

## Test CycleGAN
각각 real 데이터와 segmentation 데이터를 입력으로 넣어 주었을 경우 생성된 데이터입니다.

## TODOs
- generator, discriminator Networks Customizing
- Deployment를 위한 ONNX Conversion Script, Torch Script 추가
- QAT, Grad Clip, SWA, FP16 등 학습 기법 추가 및 테스트