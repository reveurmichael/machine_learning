library(glmnet)
library(leaps)
library(MASS)
library(naivebayes)
library(nnet)

# Part I: prostate data

prostate<-read.table('prostate.data',header = TRUE)  
summary(prostate)
train<-prostate$train
prostate<-prostate[,-10]
xtst<-as.matrix(prostate[train==FALSE,1:8])
ntst<-nrow(xtst)
X<-cbind(rep(1,ntst),xtst)
ytst<-prostate$lpsa[train==FALSE]

# Linear regression with all predictors
reg<- lm(lpsa ~.  ,data=prostate[train==TRUE,])
summary(reg)

pred<-predict(reg,newdata=prostate[train==FALSE,])
mse_full<-mean((ytst-pred)^2)

# Forward selection 
reg.forward<-regsubsets(lpsa~.,data=prostate[train==TRUE,],
                        method='forward',nvmax=30)
plot(reg.forward,scale="bic")
res<-summary(reg.forward)

# BIC
best<-which.min(res$bic)
ypred<-X[,res$which[best,]]%*%coef(reg.forward,best)
mse_forward_bic<-mean((ypred-ytst)^2)
# Adjusted R2
plot(reg.forward,scale="adjr2")
best<-which.max(res$adjr2)
ypred<-X[,res$which[best,]]%*%coef(reg.forward,best)
mse_forward_adjr2<-mean((ypred-ytst)^2)

# Backward selection 
reg.backward<-regsubsets(lpsa~.,data=prostate[train==TRUE,],method='backward',nvmax=30)
plot(reg.backward,scale="bic")
res<-summary(reg.backward)
# The results are the same as those of forward selection

# Optimal subset

reg.exhaustive<-regsubsets(lpsa~.,data=prostate[train==TRUE,],method='exhaustive',nvmax=30)
plot(reg.exhaustive,scale="bic")
res<-summary(reg.exhaustive)
# Again the results are the same

# Ridge
xapp<-as.matrix(prostate[train==TRUE,1:8])
yapp<-prostate$lpsa[train==TRUE]

cv.out<-cv.glmnet(xapp,yapp,alpha=0,standardize=TRUE)
plot(cv.out)

fit<-glmnet(xapp,yapp,lambda=cv.out$lambda.min,alpha=0,standardize=TRUE)
ridge.pred<-predict(fit,s=cv.out$lambda.min,newx=xtst)
mse_ridge<-mean((ytst-ridge.pred)^2)

# Lasso
cv.out<-cv.glmnet(xapp,yapp,alpha=1,standardize=TRUE)
plot(cv.out)

fit<-glmnet(xapp,yapp,lambda=cv.out$lambda.min,alpha=1,standardize=TRUE)
lasso.pred<-predict(fit,s=cv.out$lambda.min,newx=xtst)
mse_lasso<-mean((ytst-lasso.pred)^2)

print(c(mse_full,mse_forward_bic,mse_forward_adjr2, mse_ridge,mse_lasso))

#------------------------------------------------------------
# Part II: Vowel data

vowel <- read.table('vowel.data',
                    header=FALSE)
names(vowel)[11]<-'class'
n<-nrow(vowel)

ntrain<-round(2*n/3)
ntest<-n-ntrain
train<-sample(n,ntrain)
vowel.train<-vowel[train,]
vowel.test<-vowel[-train,]

K<-5
folds=sample(1:K,ntrain,replace=TRUE)
CV<-matrix(0,K,4)

for(k in (1:K)){
  fit.lda<- lda(class~.,data=vowel.train[folds!=k,])
  pred.lda<-predict(fit.lda,newdata=vowel.train[folds==k,])
  CV[k,1]<-sum(pred.lda$class!=vowel.train$class[folds==k])
  fit.qda<- qda(class~.,data=vowel.train[folds!=k,])
  pred.qda<-predict(fit.qda,newdata=vowel.train[folds==k,])
  CV[k,2]<-sum(pred.qda$class!=vowel.train$class[folds==k])
  fit.nb<- naive_bayes(as.factor(class)~.,data=vowel.train[folds!=k,])
  pred.nb<-predict(fit.nb,newdata=vowel.train[folds==k,],type="class")
  CV[k,3]<-sum(pred.nb!=vowel.train$class[folds==k])
  fit.logreg<- multinom(as.factor(class)~.,data=vowel.train[folds!=k,],trace=FALSE)
  pred.logreg<-predict(fit.logreg,newdata=vowel.train[folds==k,],type='class')
  CV[k,4]<-sum(pred.logreg!=vowel.train$class[folds==k])
}
err_cv=colSums(CV)/ntrain
print(err_cv)

fit.qda<- qda(class~.,data=vowel.train)
pred.qda<-predict(fit.qda,newdata=vowel.test)
err<- mean(pred.qda$class!=vowel.test$class)



