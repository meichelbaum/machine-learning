Accuracies(r^2):

Multiple Linear Regression:  0.6576337570381492

Polynomial Regression:       0.9328974463819352

Support Vector Regression:   0.9188280950736751

Decision Tree Regression:    0.9666654686222395

Random Forest Regression:    0.9671791691324331




Notes:  
ALWAYS split before scaling to prevent information leaks (https://www.udemy.com/course/machinelearning/learn/lecture/19019768#overview splitting)

batch=32 by default
epochs: watch for residuals

Classifier:
  activation="relu" on hidden layers
  activation="sigmoid" on output layer
  Loss_funcion="binary_crossentropy", "categorical_crossentropy"  
  activation=  "adam"               , "softmax"
Regressor:
  activation="none" on last layer
