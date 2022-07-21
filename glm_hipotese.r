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

# newdata <- data.frame(y = class_test, x = test)
# newdata$y_hat <- predict(model, newdata = newdata, type = 'response')

# # plot
# plot(data$y_hat)
# lines(data$y, col = 'red')
# 
# plot(newdata$y_hat)
# lines(newdata$y, col = 'red')

# soft
X <- cbind(1, train)
Y <- class_train

p <- data$y_hat
beta <- model$coefficients

W <- p * (1 - p) * diag(length(p))
W_inv <- 1 / (p * (1 - p)) * diag(length(p))
# Z <- X %*% beta_maisum + solve(W) %*% (Y - p)
Z <- X %*% beta + W_inv %*% (Y - p)

beta_maisum <- solve(t(X) %*% W %*% X) %*% t(X) %*% W %*% Z

epsilon <- beta_maisum - beta
print(epsilon)

epochs <- 10
for (epoch in 1:epochs){
  p <- as.vector(1 / (1 + exp(- X %*% beta_maisum)))
  W <- p * (1 - p) * diag(length(p))
  W_inv <- 1 / (p * (1 - p)) * diag(length(p))
  # Z <- X %*% beta_maisum + solve(W) %*% (Y - p)
  Z <- X %*% beta_maisum + W_inv %*% (Y - p)

  beta_maisum <- solve(t(X) %*% W %*% X) %*% t(X) %*% W %*% Z
}

epsilon <- beta_maisum - beta
print(epsilon)
