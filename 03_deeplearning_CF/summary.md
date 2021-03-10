# Deep learning CF



* [Neural Collaborate Filtering](#Neural Collaborate Filtering)

* [Item2vec](#Item2vec)

## Neural Collaborate Filtering

* MF 의 문제점
  * 선형 연산으로는 user, item latent vector 관계를 제대로 표현할 수 없음
  * item latent vector 내 모든 구성 성분이 큰 vector 는 취향과 상관없이 모두에게 추천할 수 있음

* 기본 구조

  ![](https://user-images.githubusercontent.com/43728746/76158443-20e3b900-6159-11ea-8a5b-1d09578c740e.png)

  * input: user - item one-hot encoding vector
  
* Embedding layer

  * P 의 row 를 user latent vector 로 사용
    * ![](https://user-images.githubusercontent.com/43728746/76158485-8c2d8b00-6159-11ea-9c05-e3dc04ec1a0a.png)

## Item2vec

* word2vec: 한 단어(A given word)에 대해 다른 단어의 근접(within window size)존재 확률 학습
* item2vec
  * 한 item에 대한 다른 item 의 근접 존재 확률 학습(유사도)
    * 신규 user, item 에 대한 처리 방안?
  * word2vec 과 동일
    * word: item
    * sentence: 한 user 에 대해 비슷한 평점을 받은 item 집합
      * 4점 이상 -> Liked sentence
      * 4점 미만 -> Disliked sentence

## References

* [[논문 리뷰] Neural Collaborative Filtering](https://leehyejin91.github.io/post-ncf/)
* [Item2Vec tutorial](https://github.com/bwange/Item2vec_Tutorial_with_Recommender_System_Application/blob/master/Making_Your_Own_Recommender_System_with_Item2Vec.ipynb)

