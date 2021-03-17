# Autoencoder, RNN + CF

## 1. Autoencoder

### 개요
* Autocoder가 더 다른 신경망 추천 시스템에 비해 빠르다(computational advantage)

## Architecture

* ![](C:\Users\wonca\OneDrive\바탕 화면\shbaek\git_repository\study_recommendation_system\04_autoencoder_CF\img\autorec_architecture.PNG)
* User, Item 중 하나를 선택해 구성
* 

### 기존 CF 모델 비교

|항목|RMB|MF|AUTOREC|비고|
|:---|:---|:---|:---|:---|
|모델|확률 모델|ㅁ|autoencoder|ㅁ|
|목적함수|최대우도|RMSE|RMSE|ㅁ|
|학습방식|Contrastive Divergence|gradient-based, linear|gradient-based, nonlinear|gradient-based 가  contrasive divergence 보다 상대적으로 빠름|
|파라메터|rating 마다 parameter 추정||rating 수에 영향을 받지 않음|AUTOREC 이 RMB에 비해 파라메터 수가 적다|
|역전파방식|ㅁ|ㅁ|ㅁ|ㅁ|
|입력데이터|평점|평점, item, user 모두 embedding|agnostic to rating, item, user 둘중 하나만 embedding|AUTOREC 의 파라메터가 더 적음, 과적합 방지|
|학습요소|ㅁ|Linear|Nonlinear|ㅁ|

## 2. RNN +CF



## References

* [AutoRec: Autoencoders Meet Collaborative Filtering](http://users.cecs.anu.edu.au/~u5098633/papers/www15.pdf)



## TODO

* RMB 모델
  * 개념
  *  contrasive divergence
* 최대우도
* Linear vs Nonlinear

* Discriminate vs Gerative