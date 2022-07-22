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
# lambda <- 1e3

lambdas <- seq(0, 0.003, 0.0001)
report <- matrix(0, nrow = length(lambdas), ncol = 3)
for (lambda in lambdas){
  # first guess
  # beta <- matrix(0, nrow = ncol(X), ncol = 1)
  y_num = matrix(0, nrow = nrow(Y), ncol = ncol(Y))
  y_num[Y == 0] = 0.05
  y_num[Y == 1] = 0.95
  y_lm <- - log(1 / y_num - 1)
  
  data_lm <- data.frame(y = y_lm, x = train)
  linear <- lm(y_lm ~ train, data = data_lm)
  
  beta <- linear$coefficients
  
  # epochs <- model$iter
  epochs <- 100
  tol <- 0.001
  epoch <- 0
  delta <- 1e6
  while (delta > tol & epoch < epochs){
    p <- as.vector(1 / (1 + exp(- X %*% beta)))
    W <- p * (1 - p) * diag(length(p))
    W_inv <- 1 / (p * (1 - p)) * diag(length(p))
    Z <- X %*% beta + W_inv %*% (Y - p)
    
    beta_old <- beta
    beta <- solve(t(X) %*% W %*% X + lambda * diag(ncol(X))) %*% t(X) %*% W %*% Z
    delta <- max(abs(beta - beta_old))
    epoch <- epoch + 1
    # print(c(epoch, delta))
  }
  
  # press
  y_hat <- as.vector(1 / (1 + exp(- X %*% beta)))
  r <- Y - y_hat
  H <- X %*% solve(t(X) %*% W %*% X + lambda * diag(ncol(X))) %*% t(X) %*% W
  
  y_bar <- mean(Y)
  
  press <- 0
  sqt <- 0
  for (i in 1:nrow(r)){
    press <- press + (r[i] / (1 - H[i, i]))^2
    sqt <- sqt + (Y[i] - y_bar)^2
  }
  R2_pred <- 1 - press / sqt
  
  # cv
  X_test <- cbind(1, test)
  y_hat <- as.vector(1 / (1 + exp(- X_test %*% beta)))
  mse <- sum((class_test - y_hat)^2) / length(y_hat)
  
  # log
  print(c(lambda, R2_pred, mse))
  idx <- which(lambda == lambdas)
  report[idx, 1] <- lambda
  report[idx, 2] <- R2_pred
  report[idx, 3] <- mse
}

# plot
plot(report[, 1], report[, 2], 'l', ylab = 'R2_pred', xlab = 'lambda')
plot(report[, 1], report[, 3], 'l', ylab = 'mse', xlab = 'lambda')
