"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
# Test case 1
# Real Input and Real Output

print("\n\n\nTest Case 1: Real Input and Real Output")

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("\n\n")
    print("Criteria :", criteria)
    print("RMSE: ", rmse(y_hat, y))
    print("MAE: ", mae(y_hat, y))
    print("\n\n")

# Test case 2
# Real Input and Discrete Output

print("\n\n\nTest Case 2: Real Input and Discrete Output")

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("\n\n")
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y))
    for cls in y.unique():
        print("Precision: ", precision(y_hat, y, cls))
        print("Recall: ", recall(y_hat, y, cls))
    print("\n\n")


# Test case 3
# Discrete Input and Discrete Output

print("\n\n\nTest Case 3: Discrete Input and Discrete Output")

N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("\n\n")
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y))
    for cls in y.unique():
        print("Precision: ", precision(y_hat, y, cls))
        print("Recall: ", recall(y_hat, y, cls))
    print("\n\n")

# Test case 4
# Discrete Input and Real Output

print("\n\n\nTest Case 4: Discrete Input and Real Output")

N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randn(N))

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("\n\n")
    print("Criteria :", criteria)
    print("RMSE: ", rmse(y_hat, y))
    print("MAE: ", mae(y_hat, y))
    print("\n\n")


"""
Test Case 1: Real Input and Real Output
  ?(Ft: 0<=0.05329582319796417)
  Y:      ?(Ft: 1<=0.20811805510481907)
    Y:        ?(Ft: 3<=-0.21990505439890984)
      Y:          ?(Ft: 2<=-0.22320997159287242)
        Y:            ?(Ft: 4<=0.32562005536409444)
          Y:  -0.8895144296255233
          N:  1.158595579007404
        N:            ?(Ft: 4<=-1.568610766924162)
          Y:  -0.6800247215784908
          N:  0.23225369716100355
      N:          ?(Ft: 4<=-0.9582751181258865)
        Y:  0.6565536086338297
        N:            ?(Ft: 2<=-0.5010367518925393)
          Y:  0.01300189187790702
          N:  -0.0771017094141042
    N:        ?(Ft: 4<=0.17202548140054982)
      Y:          ?(Ft: 2<=0.8694746584607027)
        Y:  -0.7537361643574896
        N:  0.82206015999449
      N:          ?(Ft: 2<=-0.21924537745127903)
        Y:  0.4738329209117875
        N:            ?(Ft: 3<=-0.6856485062192972)
          Y:  1.8657745111447566
          N:  1.1216205960754615
  N:      ?(Ft: 2<=-0.20304861707516575)
    Y:        ?(Ft: 1<=-0.21187967651507972)
      Y:          ?(Ft: 4<=0.11383787046026832)
        Y:            ?(Ft: 3<=0.4565842912286555)
          Y:  0.8271832490360238
          N:  1.4535340771573169
        N:            ?(Ft: 3<=0.48829376195377644)
          Y:  0.787084603742452
          N:  0.8727437481811486
      N:          ?(Ft: 3<=0.2728888257462647)
        Y:            ?(Ft: 4<=-0.7065980911615423)
          Y:  0.4127809269364983
          N:  -0.2550224746204133
        N:  0.3411519748166439
    N:        ?(Ft: 3<=-0.0026852133151751556)
      Y:          ?(Ft: 1<=-0.5286972628784017)
        Y:            ?(Ft: 4<=0.14758167200349676)
          Y:  -0.7143514180263678
          N:  0.27669079933001905
        N:            ?(Ft: 4<=-1.011452357446305)
          Y:  -1.1913034972026486
          N:  0.29307247329868125
      N:          ?(Ft: 4<=-0.893366037296093)
        Y:  0.9633761292443218
        N:            ?(Ft: 1<=0.5586371594863372)
          Y:  -0.28509473400291696
          N:  -0.8158102849654383



Criteria : information_gain
RMSE:  0.5354636487425356
MAE:  0.21118798319688617



  ?(Ft: 0<=0.05329582319796417)
  Y:      ?(Ft: 1<=0.20811805510481907)
    Y:        ?(Ft: 3<=-0.21990505439890984)
      Y:          ?(Ft: 2<=-0.22320997159287242)
        Y:            ?(Ft: 4<=0.32562005536409444)
          Y:  -0.8895144296255233
          N:  1.158595579007404
        N:            ?(Ft: 4<=-1.568610766924162)
          Y:  -0.6800247215784908
          N:  0.23225369716100355
      N:          ?(Ft: 4<=-0.9582751181258865)
        Y:  0.6565536086338297
        N:            ?(Ft: 2<=-0.5010367518925393)
          Y:  0.01300189187790702
          N:  -0.0771017094141042
    N:        ?(Ft: 4<=0.17202548140054982)
      Y:          ?(Ft: 2<=0.8694746584607027)
        Y:  -0.7537361643574896
        N:  0.82206015999449
      N:          ?(Ft: 2<=-0.21924537745127903)
        Y:  0.4738329209117875
        N:            ?(Ft: 3<=-0.6856485062192972)
          Y:  1.8657745111447566
          N:  1.1216205960754615
  N:      ?(Ft: 2<=-0.20304861707516575)
    Y:        ?(Ft: 1<=-0.21187967651507972)
      Y:          ?(Ft: 4<=0.11383787046026832)
        Y:            ?(Ft: 3<=0.4565842912286555)
          Y:  0.8271832490360238
          N:  1.4535340771573169
        N:            ?(Ft: 3<=0.48829376195377644)
          Y:  0.787084603742452
          N:  0.8727437481811486
      N:          ?(Ft: 3<=0.2728888257462647)
        Y:            ?(Ft: 4<=-0.7065980911615423)
          Y:  0.4127809269364983
          N:  -0.2550224746204133
        N:  0.3411519748166439
    N:        ?(Ft: 3<=-0.0026852133151751556)
      Y:          ?(Ft: 1<=-0.5286972628784017)
        Y:            ?(Ft: 4<=0.14758167200349676)
          Y:  -0.7143514180263678
          N:  0.27669079933001905
        N:            ?(Ft: 4<=-1.011452357446305)
          Y:  -1.1913034972026486
          N:  0.29307247329868125
      N:          ?(Ft: 4<=-0.893366037296093)
        Y:  0.9633761292443218
        N:            ?(Ft: 1<=0.5586371594863372)
          Y:  -0.28509473400291696
          N:  -0.8158102849654383



Criteria : gini_index
RMSE:  0.5354636487425356
MAE:  0.21118798319688617






Test Case 2: Real Input and Discrete Output
  ?(Ft: 0<=-0.08577209844000608)
  Y:      ?(Ft: 1<=0.44715936490844055)
    Y:        ?(Ft: 4<=0.11563937343109286)
      Y:          ?(Ft: 2<=-0.3501673123701943)
        Y:  0
        N:            ?(Ft: 3<=0.5857057757795396)
          Y:  1
          N:  0
      N:          ?(Ft: 2<=-0.3990407165380366)
        Y:            ?(Ft: 3<=1.3075281686510607)
          Y:  2
          N:  1
        N:  1
    N:        ?(Ft: 4<=0.7469212799511924)
      Y:          ?(Ft: 2<=0.40101415560434994)
        Y:            ?(Ft: 3<=-0.29097371552333723)
          Y:  1
          N:  1
        N:            ?(Ft: 3<=0.828292538801651)
          Y:  4
          N:  1
      N:          ?(Ft: 2<=0.21051094476048596)
        Y:  3
        N:  4
  N:      ?(Ft: 4<=-0.2669591683111141)
    Y:        ?(Ft: 3<=0.1845393567779638)
      Y:          ?(Ft: 2<=0.26602065173186673)
        Y:            ?(Ft: 1<=-0.07052003868470604)
          Y:  1
          N:  1
        N:            ?(Ft: 1<=-1.6515636718855937)
          Y:  2
          N:  4
      N:          ?(Ft: 1<=0.7638357306619156)
        Y:            ?(Ft: 2<=0.13241527997090224)
          Y:  2
          N:  1
        N:  2
    N:        ?(Ft: 2<=-0.2164032710349905)
      Y:          ?(Ft: 3<=0.03061377836251874)
        Y:            ?(Ft: 1<=-0.3542725977210545)
          Y:  4
          N:  2
        N:  4
      N:          ?(Ft: 1<=-0.6815059560682802)
        Y:  3
        N:  0



Criteria : information_gain
Accuracy:  0.9
Precision:  1.0
Recall:  0.8
Precision:  0.8181818181818182
Recall:  0.9
Precision:  1.0
Recall:  1.0
Precision:  1.0
Recall:  1.0
Precision:  0.75
Recall:  1.0



  ?(Ft: 2<=0.03576416326859091)
  Y:      ?(Ft: 1<=0.27178617784966114)
    Y:        ?(Ft: 4<=0.09458922166660425)
      Y:          ?(Ft: 0<=0.22995239273586537)
        Y:  0
        N:            ?(Ft: 3<=0.022303650801366915)
          Y:  4
          N:  1
      N:          ?(Ft: 0<=0.1821090369230475)
        Y:            ?(Ft: 3<=1.3075281686510607)
          Y:  2
          N:  1
        N:  2
    N:        ?(Ft: 3<=0.35343870545413)
      Y:  4
      N:          ?(Ft: 0<=-0.08997024152158373)
        Y:  3
        N:            ?(Ft: 4<=-0.6041050169667981)
          Y:  2
          N:  4
  N:      ?(Ft: 4<=0.23918585056270386)
    Y:        ?(Ft: 0<=-0.02110753487694299)
      Y:          ?(Ft: 1<=0.1895116329357593)
        Y:            ?(Ft: 3<=0.5857057757795396)
          Y:  1
          N:  0
        N:  1
      N:          ?(Ft: 3<=0.03218384232332724)
        Y:            ?(Ft: 1<=-1.6515636718855937)
          Y:  2
          N:  4
        N:            ?(Ft: 1<=0.2708309101159001)
          Y:  1
          N:  1
    N:        ?(Ft: 3<=0.030619563856678206)
      Y:          ?(Ft: 0<=0.005689911950707532)
        Y:  1
        N:            ?(Ft: 1<=-0.6815059560682802)
          Y:  3
          N:  0
      N:          ?(Ft: 0<=-0.7656511127976824)
        Y:  4
        N:            ?(Ft: 1<=0.9991883637102204)
          Y:  1
          N:  4



Criteria : gini_index
Accuracy:  0.9
Precision:  1.0
Recall:  0.9
Precision:  0.8181818181818182
Recall:  0.9
Precision:  1.0
Recall:  0.8
Precision:  1.0
Recall:  1.0
Precision:  0.75
Recall:  1.0






Test Case 3: Discrete Input and Discrete Output
  ?(Ft: 1)
  = 3:     ?(Ft: 4)
    = 4:       ?(Ft: 0)
      = 0: 0
      = 1: 2
      = 3: 0
    = 2: 2
    = 0: 3
    = 3: 3
  = 0:     ?(Ft: 4)
    = 4:       ?(Ft: 0)
      = 3: 0
      = 0: 4
    = 1:       ?(Ft: 0)
      = 3:         ?(Ft: 2)
        = 4:           ?(Ft: 3)
          = 0: 0
    = 2: 4
    = 0: 0
  = 4:     ?(Ft: 0)
    = 4: 1
    = 3:       ?(Ft: 2)
      = 1: 3
      = 3: 1
    = 2: 1
    = 0: 3
  = 1:     ?(Ft: 0)
    = 3: 2
    = 1: 2
    = 0: 3
  = 2:     ?(Ft: 2)
    = 4: 2
    = 0: 0
    = 2: 1
    = 1: 1



Criteria : information_gain
Accuracy:  0.9666666666666667
Precision:  0.875
Recall:  1.0
Precision:  1.0
Recall:  1.0
Precision:  1.0
Recall:  0.8888888888888888
Precision:  1.0
Recall:  1.0
Precision:  1.0
Recall:  1.0



  ?(Ft: 1)
  = 3:     ?(Ft: 4)
    = 4:       ?(Ft: 0)
      = 0: 0
      = 1: 2
      = 3: 0
    = 2: 2
    = 0: 3
    = 3: 3
  = 0:     ?(Ft: 4)
    = 4:       ?(Ft: 0)
      = 3: 0
      = 0: 4
    = 1:       ?(Ft: 0)
      = 3:         ?(Ft: 2)
        = 4:           ?(Ft: 3)
          = 0: 0
    = 2: 4
    = 0: 0
  = 4:     ?(Ft: 0)
    = 4: 1
    = 3:       ?(Ft: 2)
      = 1: 3
      = 3: 1
    = 2: 1
    = 0: 3
  = 1:     ?(Ft: 0)
    = 3: 2
    = 1: 2
    = 0: 3
  = 2:     ?(Ft: 2)
    = 4: 2
    = 0: 0
    = 2: 1
    = 1: 1



Criteria : gini_index
Accuracy:  0.9666666666666667
Precision:  0.875
Recall:  1.0
Precision:  1.0
Recall:  1.0
Precision:  1.0
Recall:  0.8888888888888888
Precision:  1.0
Recall:  1.0
Precision:  1.0
Recall:  1.0






Test Case 4: Discrete Input and Real Output
  ?(Ft: 4)
  = 0:     ?(Ft: 0)
    = 3: 1.0062928092144405
    = 1:       ?(Ft: 1)
      = 4: 0.8356921120651418
      = 3: 1.2012139221639448
    = 0: 1.1677820616598074
    = 2: 0.3376026620752022
    = 4: -0.48760622407249354
  = 2:     ?(Ft: 0)
    = 4:       ?(Ft: 1)
      = 0: -0.5768918695231487
      = 3: 0.8711247034316923
      = 1: -0.32602353216784113
    = 1:       ?(Ft: 1)
      = 3: 0.5298041779152828
      = 0: -0.42098448082026296
    = 2: -2.4716445001272893
    = 3: 0.08658978747289992
  = 1:     ?(Ft: 1)
    = 0: -1.129706854657618
    = 1:       ?(Ft: 0)
      = 4: 1.4415686206579004
      = 3: -0.4080753730215514
    = 2: 0.37114587337130883
    = 3: -0.4325581878196209
    = 4: -2.038124535177854
  = 4:     ?(Ft: 3)
    = 3:       ?(Ft: 0)
      = 2: -0.7968952554704768
      = 4: -1.008086310917404
    = 1: -0.2030453860429927
    = 0: 0.39445214237829684
    = 4: 2.075400798645439
  = 3:     ?(Ft: 2)
    = 2: 0.57707212718054
    = 4: -0.6039851867158206
    = 1: -0.15567723539207948
    = 0:       ?(Ft: 3)
      = 0: 0.2544208433012131
      = 2: 0.2897748568964129
    = 3: -0.4118769661224674



Criteria : information_gain
RMSE:  0.0
MAE:  0.0



  ?(Ft: 4)
  = 0:     ?(Ft: 0)
    = 3: 1.0062928092144405
    = 1:       ?(Ft: 1)
      = 4: 0.8356921120651418
      = 3: 1.2012139221639448
    = 0: 1.1677820616598074
    = 2: 0.3376026620752022
    = 4: -0.48760622407249354
  = 2:     ?(Ft: 0)
    = 4:       ?(Ft: 1)
      = 0: -0.5768918695231487
      = 3: 0.8711247034316923
      = 1: -0.32602353216784113
    = 1:       ?(Ft: 1)
      = 3: 0.5298041779152828
      = 0: -0.42098448082026296
    = 2: -2.4716445001272893
    = 3: 0.08658978747289992
  = 1:     ?(Ft: 1)
    = 0: -1.129706854657618
    = 1:       ?(Ft: 0)
      = 4: 1.4415686206579004
      = 3: -0.4080753730215514
    = 2: 0.37114587337130883
    = 3: -0.4325581878196209
    = 4: -2.038124535177854
  = 4:     ?(Ft: 3)
    = 3:       ?(Ft: 0)
      = 2: -0.7968952554704768
      = 4: -1.008086310917404
    = 1: -0.2030453860429927
    = 0: 0.39445214237829684
    = 4: 2.075400798645439
  = 3:     ?(Ft: 2)
    = 2: 0.57707212718054
    = 4: -0.6039851867158206
    = 1: -0.15567723539207948
    = 0:       ?(Ft: 3)
      = 0: 0.2544208433012131
      = 2: 0.2897748568964129
    = 3: -0.4118769661224674



Criteria : gini_index
RMSE:  0.0
MAE:  0.0

"""