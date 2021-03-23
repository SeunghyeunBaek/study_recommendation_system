## TOC

* [Factorization Machines](## Factorization Machines)
  * [개요]()
  * [Dataset구성]()
  * [모델수식]()
  * [연산속도향상]()
  * [일반화(D-way)]()
  * 성능
* [Wide & Deep learning](## Wide & Deep learning)
  * 개요
  * 모델 구성
  * 시스템구성
  * 성능

## Factorization Machines

### 개요

* Factorization machines(**FM**) = Matrix Factorization model(**MF**) + Support Vector Machine(**SVM**)
  * **Sparse data**
    * Latent vector 를 학습해서 sparse data 에서도 잘 작동함(MF)
  * **General predictor**
    * MF 는 특정한 input data 에서만 사용 가능하지만 FM 은 추천외 다른 Task에서도 활용가능(SVM)
  * **Linear complexity**
    * $$O(kn^2)$$  ->  $$O(kn)$$

### Dataset구성

![](C:\Users\wonca\OneDrive\바탕 화면\shbaek\git_repository\study_recommendation_system\05_wide_deeplearning_CF\image\FM_dataset.png)

* S = {(user, movie, rate)}
* ST = {(user, movie, rate, time, movie_last, target)}

### 모델수식

* 모델 비교

| 알고리즘  | 수식                                                         | 시간복잡도 | 학습요소                                                     | 비고                                                    |
| --------- | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ | ------------------------------------------------------- |
| Linear    | $$\hat{y(x)}=w_0 +\Sigma w_ix_i$$                            | $$O(n)$$   | $$w_0 \in R, w \in R^n$$                                     | * Interaction 학습 불가                                 |
| Poly(2)   | $$\hat{y(x)}=w_0+ \Sigma w_ix_i + \Sigma\Sigma W_{ij}x_ix_j$$ | $$O(n^2)$$ | $$w_0 \in R, w \in R^n, W\in R^{n\times n}$$                 | * Sparse data 에서 작동 어려움<br>* 느림                |
| **FM(2)** | $$\hat{y(x)}=w_0+ \Sigma_{i=0}^n w_ix_i + \Sigma_{i=0}^n\Sigma_{j=i+1}^n <v_i, v_j>x_ix_j$$ | $$O(kn)$$  | $$w_0 \in R, w \in R^n, V \in R^{n \times k}$$               | * Feature 의 Latent vector 를 학습(k 차원)              |
| *MF*      | $$\hat{y(x)}=w_0 + w_u + w_i + <v_u, v_i>$$                  | $$O(n)$$   | $$w_0 \in R, w \in R^n, V_u \in R^{n \times k}, V_i \in R^{kxn}$$ | * FM 의 Feature 가 user, item 밖에 없을 때 MF 와 동일함 |



  * $$w_0$$: Global bias

* $$w_i \in R^n$$

  * i 번 feature 의 가중치

* $$\Sigma_{i=0}^n\Sigma_{j=i+1}^n <v_i, v_j> x_ix_j$$

  * (i, i)는 고려하지 않는다.
  * (i, j) 와 (j, i) 의 관계는 같다.

* $$\hat{w}_{ij}$$ =: $$<v_i, v_j>$$,   $$\hat{w}_{ij} \in R$$

  ![](C:\Users\wonca\OneDrive\바탕 화면\shbaek\git_repository\study_recommendation_system\05_wide_deeplearning_CF\image\Feature_latent_vector.PNG)

  * $$v_i$$: i번 Feature 의 latent vector
  * $$<v_i, v_j> $$: i 번 feature, j 번 feature 간 interaction
    * W는 **positive definite matrix**
    * $$ W=VV^T, W\in R^{n\times n} , V \in R^{n \times k}$$

## 연산속도향상

* $$O(n^2)$$ ->  $$O(kn)$$

![](C:\Users\wonca\OneDrive\바탕 화면\shbaek\git_repository\study_recommendation_system\05_wide_deeplearning_CF\image\compuation.PNG)

* Gradient
  * $$\Sigma_{j=1}^n v_{j,f} x_j $$ 는 $$i$$ 와 상관없이 계산할 수 있음

![](C:\Users\wonca\OneDrive\바탕 화면\shbaek\git_repository\study_recommendation_system\05_wide_deeplearning_CF\image\grdient.PNG)



## 일반화(D-way)

* Interaction 항 수에 따라 일반화 할 수 있음

![](C:\Users\wonca\OneDrive\바탕 화면\shbaek\git_repository\study_recommendation_system\05_wide_deeplearning_CF\image\d_way.PNG)



## Wide & Deep learning(작성중)

### 개요

* **Wide & Deep learning = Wide Linear model(Memorization) + Deep nuralnet(Generalization)**

* Memorization, Generalization

  * Generalization, Memorization 의 오류 예시

    * Generalization
      * 학습: 갈매기 -> 난다, 비둘기-> 난다, 참새 -> 난다
      * 예측: 날개 있음 -> 난다
      * 오류: <span style="color:red">**팽귄 -> 날개 있음 -> 난다**</span>
    * Memorization
      * 학습: 갈매기-> 난다, 비둘기 -> 난다, 참새-> 난다, **팽귄 -> 못난다**
      * 예측: 갈매기->난다, 비둘기->난다, 참새->난다, **팽귄->못난다**
      * 오류: **<span style="color:red">타조-> 몰라</span>**

  * Gerneralization, Memorization 비교
  
| 항목          | Memorization                                                 | Generalization                                               |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 정의          | 과거 데이터를 잘 학습하는 성질                               | 새로운 데이터에 대해 잘 예측하는 성질                        |
| 알고리즘 예시 | Logistic regression - Cross production                       | Factorization machine, DNN - Embedding                       |
| 장점          | * 간단하고 설명 가능함<br>* Cross product 로 Interaction 학습 가능 | * 더 적은 Parameter 로 학습할 수 있음                        |
| 단점          | * 과거 데이터에 없던 item, feature interaction은 학습하지 않음<br/>* Generlization 을 위해 Feature engineering 필요(Grouping)<br> | * Feature engineering 덜 해도 됨<br>* Sparse dataset 에서 Embedding 하기 어려움(잘못된 추천할 수 있음)<br> |


* 더 적은 Parameter 로 학습할 수 있음
  * ![](C:\Users\wonca\OneDrive\바탕 화면\shbaek\git_repository\study_recommendation_system\05_wide_deeplearning_CF\image\Wide_vs_Deep.PNG)
* Sparse dataset 에서 Embedding 하기 어려움
  * ![](C:\Users\wonca\OneDrive\바탕 화면\shbaek\git_repository\study_recommendation_system\05_wide_deeplearning_CF\image\wide_vs_deep_specific.PNG)

### 모델

![](C:\Users\wonca\OneDrive\바탕 화면\shbaek\git_repository\study_recommendation_system\05_wide_deeplearning_CF\image\wide_deep_architecture.PNG)

* Linear model과 Neuralnet 을 결합한 형태
* Ensemble 은 두 모델이 각각 학습하고 예측값만 합치는 형태 / Joint trainning 은 학습 시 두 모델의  Weight를 함께 역전파함
* Linear model, Nuralnet 의 약점을 보완할 수 있음

#### WIDE

$$
y = w^Tx + b
$$

* $$x$$ 는 interaction 항을 모두 포함한 feature

$$
\phi_k(x) = \prod_{i=1}^d x_{i}^{c_{ki }}, c_{ki} \in 0, 1
$$

* $$c_k $$ 벡터에 포함할 항의 순서를 지정해 놓고 0, 1 로 제외함

$$
c_{ki} = [0, 1, 1, 0]\\ x_{i} = [0.1, 0.2, 0.2, 0.1] \\ \phi(x_{i}) = 0.1^0*0.2^1*0.2^1*0.1^0 = 0.4
$$



#### DEEP

$$
a^{(l+1)} = f(W^{(l)}a^{(l)}+b^{(l)})
$$

* 임의로 embedding 한 후 학습
  * $$l$$: 레이어 번호
  * $$a$$: 활성값
  * $$W$$: 가중치
  * $$b$$: 편향

#### 구조

![](C:\Users\wonca\OneDrive\바탕 화면\shbaek\git_repository\study_recommendation_system\05_wide_deeplearning_CF\image\wide_deep_laerning_architecture.PNG)

$$
P(Y=1|x) = \sigma(w^T_{wide}[x, \phi(x)] + w^T_{deep}a^{(l_f)}+b)
$$

* 새로운 데이터가 들어올 때 마다 재학습
  * 재학습 시 이전 embedding 값과 weight($$w_{wide}, w_{deep}$$)를 유지한채로 학습

### 시스템 구성

* 검색으로 후보군을 정하고 모델로 순위를 할당함

![](C:\Users\wonca\OneDrive\바탕 화면\shbaek\git_repository\study_recommendation_system\05_wide_deeplearning_CF\image\wid_deep_learning_recommender_system_architecture.PNG)

### 성능

![](C:\Users\wonca\OneDrive\바탕 화면\shbaek\git_repository\study_recommendation_system\05_wide_deeplearning_CF\image\wd_performance.PNG)

## 기타 용어 정리

* Factorization machines
  * Support vector machines
  * Positive definite matrix
  * Transformation in the dual form
  * Algorithms
    * PARAFAC
    * SVM
    * PITF
    * SVD ++

## References

* 논문
  * [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
  * [Wide and Deep Learing Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)
* 참고자료
  * [머신러닝, 딥러닝 예측모델 구현 어떻게 할까? Factorization Machine](https://www.youtube.com/watch?v=96vMbEz7nK8)
  * [Factorization Machine 리뷰_ylab](https://yamalab.tistory.com/107)
  * [Support vector machine_statsquest](https://www.youtube.com/watch?v=efR1C6CvhmE)
  * [Wide&Deeplearning_google_summit_2017_youtube](https://www.youtube.com/watch?v=NV1tkZ9Lq48)
  * [Wide&Deeplearning 논문 리뷰_youtube](https://www.youtube.com/watch?v=hKoJPqWLrI4)
  
  * [와이드 앤 딥러닝: 텐서플로우](https://youtu.be/NV1tkZ9Lq48)
* 코드

  * [Factorization Machine code](https://github.com/srendle/libfm)
  * [Wide & Deep Learning code](https://github.com/jrzaurin/pytorch-widedeep)
  
    

