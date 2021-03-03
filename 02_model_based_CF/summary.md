# 모델 기반 협업 필터링

*210224*

TOC

## 협업 필터링(Collaborativ filtering, CF)

* 협업 필터링의 종류

  * 이웃기반 협업 필터링(메모리 기반 협업 필터링)

    * 과거 데이터 기반 추천
    * item, user 들 관의 관계를 중심으로 user - item 간 거리 계산

  * 모델 기반 협업 필터링

    * 모델 기반 추천
    * Latent Factor model, item 과 user 를 factor 로 구성된 차원으로 embedding
      * user 와 item 을 잠재(Latent) 속성으로 표현

  * 하이브리드 협업 필터링

    |             | 이웃기반                                                     | 모델기반                    |
    | ----------- | ------------------------------------------------------------ | --------------------------- |
    | 구현 난이도 | 간단함                                                       | 복잡함                      |
    | 계산량      | 적음                                                         | 많음                        |
    | 확장성      | 새로운 user, item 이 추가돼도 안정적<br />새로운 user, item 추천 가능 | 새로운 user, item 추천 가능 |
    | 메모리      | Rating matrix 사용                                           | Rating matrix 를 압축       |
    | 예측속도    | ?                                                            |                             |
    | 문제점      | Sparse data<br />Cold start<br />                            |                             |

* 모델 기반 협업 필터링 종류

  * Association Rule Mining
  * Matrix Factorization
  * Probalistic model

* Latent factor model

  * item 과 user 를 모두 다양한 factor 로 구성된 차원으로 embedding

![](https://miro.medium.com/max/691/1*XPJRzrDiwfH7UfPHUkvEvA.png)

* 이웃 기반 vs Latent model 
  * 너는 영화 `타이타닉`을 좋아할 것이다.
    * 이웃 기반
      * 너는 타이타닉과 비슷한 영화를 좋아한다.
      * 너와 비슷한 영화를 좋아하는 친구가 타이타닉을 좋아한다.
    * Latent 모델 기반
      * 타이타닉의 벡터와 너의 벡터가 는 멜로, 침몰, 음악 공간에서 가깝다

## Matrix Factorization(MF)

* user 가 선택하지 않은 item 을 추천하는 방법론.
  * user - item 행렬을 user - latent factor x latent factor - item 으로 분해.
  * user - latent factor x latent factor - item 연산으로 user 가 선택한 적 없는 item 에 대한 score 계산.
  
  ![https://buildingrecommenders.wordpress.com/2015/11/18/overview-of-recommender-algorithms-part-2/](https://buildingrecommenders.files.wordpress.com/2015/11/matrix-factorisation.png?w=900)

* 가정
  * user 는 Latent factor 로 구성된 벡터 공간에서 가까운 item 을 좋아할 것이다.
  * user - latent factor x latent factor - item 이 기존 데이터를 잘 재현 한다면, 결측값도 잘 예측할 수 있다.
  
* 한계

  * 내적으로는 벡터간의 유사도를 완전히 표현할 수 없다.
    * S23(0.66)> S12(0.5)> S13(0.4)
    * S41(0.6)> S43(0.4)> S42(0.2)
  * 인기 있는 아이템(성분값이 큰 벡터)는 모든 사람에게 추천할 수 있다.

![](https://user-images.githubusercontent.com/43728746/75696703-5a4a9d80-5cef-11ea-91f4-47253fd10553.png)

### SVD(Singular value decomposition)

* 행렬 분해 기법
* Sparse matrix 에서는 잘 작동하지 않음

### SGD(Stochastic Gradient Descent)

* 평점이 존재하는 부분에 대해 행렬 분해 후 오차를 최소화 하는 방향으로 학습
* User latent, Item latent 를 동시에 최적화
* SVD에 비해 결측치를 잘 처리할 수 있음

### ALS(Alternating Least Squares)

* User latent, Item latent 행렬 중 하나를 고정한채로 최적화 진행

## Logistic MF

*  Implicit 데이터로 MF 수행
*  선호도를 0과 1사이의 값으로 가정

## BPR

* event 가 발생한 item 이 그렇지 않은 item 보다 선호도가 높다는 가정
* user 별로 item x item matrix 생성
![](https://user-images.githubusercontent.com/43728746/78500414-28ff3a80-7791-11ea-907f-1316eb3a7c29.PNG)
* 좋아하는 item 선호도 - 싫어하는 item 선호도가 최대가 되는 방향으로 학습
  * $>_{u}$: 유저의 선호 정보(유저는 i 보다 j 를 좋아한다)
  * $p(\Theta)$: 파라메터가 발생할 확률(사전정보)


$$
\begin {align*} & p(\Theta | >_{u})  \propto p(>_{u} | \Theta) \ p(\Theta) \\ \\ \because \ \ p(\Theta | >_{u}) & = \frac{p(\Theta , >_{u})}{ p(>_{u})} = \frac{p(>_{u} | \Theta) p(\Theta)}{ p(>_{u})}  \propto  p(>_{u} | \Theta) \ p(\Theta) \end {align*} \\
$$

---

* References
  * [Matrix Factorization 기술을 이용한 넷플릭스 추천시스템](https://medium.com/curg/matrix-factorization-%EA%B8%B0%EC%88%A0%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EB%84%B7%ED%94%8C%EB%A6%AD%EC%8A%A4-%EC%B6%94%EC%B2%9C-%EC%8B%9C%EC%8A%A4%ED%85%9C-7455a40ad527)
  
  * [Matrix Factorization Techniques for recommender systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
  
  * [Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf)

  * [Neural Collaborative Filtering 리뷰](https://leehyejin91.github.io/post-ncf/)

  * [03. 협업필터링 기반 추천시스템 - SGD](https://eda-ai-lab.tistory.com/528)
  
  * [Matrix Factorization에 대해 이해, Alternating Least Square (ALS) 이해](https://yeo0.github.io/data/2019/02/23/Recommendation-System_Day8/#_title)
  * [[논문 리뷰] Neural Collaborative Filtering](https://leehyejin91.github.io/post-ncf/)
