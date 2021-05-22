# ML01 Spring 2021
# TD on neural networks

# Loading the Boston data set in packge MASS
library(MASS)
? Boston # description of the dataset
data(Boston)
n<-nrow(Boston)

set.seed(20210511)
#### Preprocessing

# Standardization
I<-c(1:3,5:13) # indices of quantitative predictors (we exclude 'chas' which is binary)
xx<-scale(Boston[,I])
Boston[,I]<-xx

# We split the data into training and test sets
train<-sample(n,300)
Boston.train<-Boston[train,]
Boston.test<-Boston[-train,]


#### Part I:  we  predict medv from predictor 'lstat'
plot(Boston$lstat,Boston$medv)

# This vector will be useful for plotting the prediction function
x<-seq(min(Boston$lstat)-0.5,max(Boston$lstat)+0.5,0.01)

library('nnet')

# First trial with 5 hidden units
# We train 10 neural networks and keep the one with the smallest training error
loss<-Inf
for(i in 1:10){
  nn<- nnet(medv ~ lstat, data=Boston.train, size=5, linout = TRUE,maxit=200,trace=FALSE)
  print(c(i,nn$value))
  if(nn$value<loss){
    loss<-nn$value
    nn1.best<-nn
  }
}
pred1<- predict(nn1.best,newdata=data.frame(lstat=x)) # For plotting
pred1.test<-predict(nn1.best,newdata=Boston.test) 
mse1<-mean((pred1.test-Boston.test$medv)^2)  # Test MSE

# Second trial with 10 hidden units
loss<-Inf
for(i in 1:10){
  nn<- nnet(medv ~ lstat, data=Boston.train, size=10, linout = TRUE,maxit=200,trace=FALSE)
  print(c(i,nn$value))
  if(nn$value<loss){
    loss<-nn$value
    nn2.best<-nn
  }
}
pred2<- predict(nn2.best,newdata=data.frame(lstat=x))
pred2.test<-predict(nn2.best,newdata=Boston.test)
mse2<-mean((pred2.test-Boston.test$medv)^2)

print(c(mse1,mse2)) # The big neural network is not so good

# Plot to compare the two solutions
plot(Boston$lstat,Boston$medv)
lines(x,pred1,lwd=2,col="red")
lines(x,pred2,lwd=2,col="blue")

# Regularization with weight decay coefficient = 0.1
loss<-Inf
for(i in 1:10){
  nn<- nnet(medv ~ lstat, data=Boston.train, size=10, linout = TRUE,maxit=200,
            trace=FALSE,decay=0.1)
  print(c(i,nn$value))
  if(nn$value<loss){
    loss<-nn$value
    nn3.best<-nn
  }
}
pred3<- predict(nn3.best,newdata=data.frame(lstat=x))
pred3.test<-predict(nn3.best,newdata=Boston.test)
mse3<-mean((pred3.test-Boston.test$medv)^2)
print(c(mse1,mse2,mse3)) # The regularized NN performs better

plot(Boston$lstat,Boston$medv)
lines(x,pred1,lwd=2,col="red")
lines(x,pred2,lwd=2,col="blue")
lines(x,pred3,lwd=2,col="green")

# Selection of the optimal weight decay coefficient by 5-fold cross-validation
K<-5
ntrain<-300
folds=sample(1:K,ntrain,replace=TRUE)
lambda<-c(0.01,0.02,0.05,0.1,0.5,1,2)
N<-length(lambda)
CV<-rep(0,N)
for(i in (1:N)){
  for(k in (1:K)){
    nn<- nnet(medv ~ lstat, data=Boston.train[folds!=k,],size=10, linout = TRUE, 
              decay=lambda[i], trace=FALSE)
    pred<-predict(nn,newdata=Boston.train[folds==k,])
    CV[i]<-CV[i]+ sum((pred-Boston.train$medv[folds==k])^2)
  }
  CV[i]<-CV[i]/ntrain
}
plot(lambda,CV,type='l')
lambda.opt<-lambda[which.min(CV)] # Best decay coefficient

# Re-training on the whole training set using the optimal lambda
loss<-Inf
for(i in 1:10){
  nn<- nnet(medv ~ lstat, data=Boston.train, size=10, linout = TRUE,maxit=200,
            trace=FALSE,decay=lambda.opt)
  print(c(i,nn$value))
  if(nn$value<loss){
    loss<-nn$value
    nn4.best<-nn
  }
}
pred4<- predict(nn4.best,newdata=data.frame(lstat=x))
pred4.test<-predict(nn4.best,newdata=Boston.test)
mse4<-mean((pred4.test-Boston.test$medv)^2) # Test error
print(c(mse1,mse2,mse3,mse4)) # The NN with optimal lambda performs well


### Part II: now we use all predictors


loss<-Inf
for(i in 1:10){
  nn<- nnet(medv ~ ., data=Boston.train, size=30, linout = TRUE,maxit=200,trace=FALSE)
  print(c(i,nn$value))
  if(nn$value<loss){
    loss<-nn$value
    nn.best<-nn
  }
}
pred.test<-predict(nn.best,newdata=Boston.test)
mse<-mean((pred.test-Boston.test$medv)^2)

# Selection of the optimal weight decay coefficient by 5-fold cross-validation
# We do 10 times 5-fold cross-validation to get a smooth curve
K<-5
ntrain<-300
lambda<-c(0.01,0.05,0.1,0.5,1,5,10)
N<-length(lambda)
CV<-matrix(0,N,10)
for(j in 1:10){
  print(j)
  folds=sample(1:K,ntrain,replace=TRUE)
  for(i in (1:N)){
    for(k in (1:K)){
      nn<- nnet(medv ~ ., data=Boston.train[folds!=k,],size=30, linout = TRUE, 
                decay=lambda[i], trace=FALSE)
      pred<-predict(nn,newdata=Boston.train[folds==k,])
      CV[i,j]<-CV[i,j]+ sum((pred-Boston.train$medv[folds==k])^2)
    }
  }
}
CVmean<-rowMeans(CV)/ntrain

plot(lambda,CVmean,type='l')
lambda.opt<-lambda[which.min(CVmean)] # Best decay coefficient

# Re-training on the whole training set using the optimal lambda
loss<-Inf
for(i in 1:10){
  nn<- nnet(medv ~ ., data=Boston.train, size=30, linout = TRUE,maxit=200,
            trace=FALSE,decay=lambda.opt)
  print(c(i,nn$value))
  if(nn$value<loss){
    loss<-nn$value
    nn.best<-nn
  }
}
pred.nn<-predict(nn.best,newdata=Boston.test)
mse.nn<-mean((pred.test-Boston.test$medv)^2) # Test error
print(c(mse1,mse2,mse3,mse4,mse.nn))

# Comparison with linear regression 
fit.lm<- lm(medv ~ ., data=Boston.train)
pred.lm<-predict(fit.lm,newdata=Boston.test)
mse.lm<-mean((pred.lm-Boston.test$medv)^2)

# Comparison with Lasso
library(glmnet)
xapp<-as.matrix(Boston.train[,1:13])
yapp<-Boston.train[,14]
cv.out<-cv.glmnet(xapp,yapp,alpha=1)
plot(cv.out)
fit.lasso<-glmnet(xapp,yapp,lambda=cv.out$lambda.min,alpha=1)
xtst<-as.matrix(Boston.test[,1:13])
pred.lasso<-predict(fit.lasso,s=cv.out$lambda.min,newx=xtst)
mse.lasso<-mean((pred.lasso-Boston.test$medv)^2)

# Comparison with random forests

library(randomForest)
fit.RF<-randomForest(medv ~ .,data=Boston.train,mtry=3,importance=TRUE)
pred.RF<-predict(fit.RF,newdata=Boston.test,type="response")
mse.RF<-mean((pred.RF-Boston.test$medv)^2)

print(c(mse.nn,mse.lm,mse.lasso,mse.RF)) # RF perform better
# Plot of variable importance
varImpPlot(fit.RF) # Variables 'rm' and 'lstat' are the most informative



