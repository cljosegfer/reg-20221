rm(list=ls())
options(warn = -1)

# params
dir_matlab_file <- 'data'
datasets <- c('australian', 
              'banknote', 
              'breastcancer', 
              'breastHess', 
              'bupa', 
              'climate', 
              'diabetes', 
              'fertility', 
              'german', 
              'golub', 
              'haberman', 
              'heart', 
              'ILPD', 
              'parkinsons', 
              'sonar')
K <- 10
fold_n <- 1
i <- 2

# read
print(datasets[i])
filename <- sprintf('%s/exportBase_%s_folds_10_exec_%s.mat',
                    dir_matlab_file, datasets[i], fold_n)
data_mat <- R.matlab::readMat(filename)

# train / test
train <- data_mat$data[[1]]
class_train <- data_mat$data[[2]]
class_train[class_train == -1] = 0
test <- data_mat$data[[3]]
class_test <- data_mat$data[[4]]
class_test[class_test == -1] = 0

# glm
data <- data.frame(y = class_train, x = train)
model <- glm(class_train ~ train, data = data, family = 'binomial')
data$y_hat <- predict(model, data, type = "response")

beta_true <- model$coefficients

# model
X <- cbind(1, train)
Y <- class_train

beta <- matrix(0, nrow = ncol(X), ncol = 1)

epochs <- model$iter
tol <- 0.001
epoch <- 0
delta <- 1e6
while (delta > tol & epoch < epochs){
  p <- as.vector(1 / (1 + exp(- X %*% beta)))
  W <- p * (1 - p) * diag(length(p))
  W_inv <- 1 / (p * (1 - p)) * diag(length(p))
  Z <- X %*% beta + W_inv %*% (Y - p)
  
  beta_old <- beta
  beta <- solve(t(X) %*% W %*% X) %*% t(X) %*% W %*% Z
  delta <- max(abs(beta - beta_old))
  epoch <- epoch + 1
  print(c(epoch, delta))
}

epsilon <- beta_true - beta
print(epsilon)
