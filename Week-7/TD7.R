library(rpart)

####### Part I - Spam

# Q1
spam <- read.table("spambase.dat",
                   header=FALSE)
n<- nrow(spam)
p<-ncol(spam)-1
names(spam)[58]<-"class"
train<-sample(n,2*n/3)

# Q2
tree.spam<-rpart(class~.,data=spam,subset=train,
                 method="class",
                 control = rpart.control(xval = 10, 
                                         minbucket = 10, cp = 0))

plot(tree.spam,margin = 0.05)
text(tree.spam,pretty=0,cex=0.8)

yhat=predict(tree.spam,newdata=spam[-train,],type='class')
y.test <- spam[-train, "class"]
table(y.test,yhat)
err<-1-mean(y.test==yhat)
print(err)

# Q3
plotcp(tree.spam)
printcp(tree.spam)

pruned_tree<-prune(tree.spam,cp=0.0066722)
plot(pruned_tree,margin = 0.1)
text(pruned_tree,pretty=0)

yhat1=predict(pruned_tree,newdata=spam[-train,],type='class')
table(y.test,yhat1)
err1<-1-mean(y.test==yhat1)
print(err1)

# Q4
library(randomForest)
fit.RF<-randomForest(as.factor(class) ~ .,data=spam[train,],mtry=3,importance=TRUE)
pred.RF<-predict(fit.RF,newdata=spam[-train,],type="class")
err.RF<-1-mean(y.test==pred.RF)
print(err.RF)




# Q5
library(MASS)
# LDA
lda.spam<- lda(class~.,data=spam, subset=train)
pred.spam.lda<-predict(lda.spam,newdata=spam[-train,])
perf <-table(y.test,pred.spam.lda$class)
print(perf)
err.lda<-1-mean(y.test==pred.spam.lda$class)
print(err.lda)

# Logistic regression
fit.logreg<- glm(as.factor(class)~.,data=spam,family=binomial,subset=train)
pred.logreg<-predict(fit.logreg,newdata=spam[-train,],type='response')
perf <-table(y.test,pred.logreg>0.5)
print(perf)
err.logreg<-1-mean(y.test==(pred.logreg>0.5))
print(err.logreg)

# Q6
summary(fit.logreg)
# Plot of variable importance computed by random forests
varImpPlot(fit.RF) 

####### Part II - Prostate


prostate<-read.table('prostate.data',
                     header = TRUE)  

tree.lpsa<-rpart(lpsa~. -train,data=prostate[prostate$train==TRUE,],method="anova",
                 control = rpart.control(xval = 10, minbucket = 2, cp = 0))

printcp(tree.lpsa)
plotcp(tree.lpsa)

pruned_tree<-prune(tree.lpsa,cp=0.12)
plot(pruned_tree,margin=0.05)
text(pruned_tree,pretty=0)

pred<-predict(pruned_tree,newdata=prostate[prostate$train==FALSE,])
ytest<-prostate$lpsa[prostate$train==FALSE]
mse.tree<-mean((ytest-pred)^2)
print(mse.tree)

# Random forests

fit.RF<-randomForest(lpsa ~ .,data=prostate[prostate$train==TRUE,],mtry=3,
                     importance=TRUE)
pred.RF<-predict(fit.RF,newdata=prostate[prostate$train==FALSE,],type="response")
mse.RF<-mean((ytest-pred.RF)^2)
print(mse.RF)


# Comparison

# LS regression
reg<- lm(lpsa ~. - train ,data=prostate[prostate$train==TRUE,])
pred<-predict(reg,newdata=prostate[prostate$train==FALSE,])
mse.ls<-mean((ytest-pred)^2)
print(mse.ls)

# Ridge
library(glmnet)
xapp<-as.matrix(prostate[prostate$train==TRUE,1:8])
xtst<-as.matrix(prostate[prostate$train==FALSE,1:8])
yapp<-prostate$lpsa[prostate$train==TRUE]
cv.out<-cv.glmnet(xapp,yapp,alpha=0,standardize=TRUE)
fit<-glmnet(xapp,yapp,lambda=cv.out$lambda.min,alpha=0,standardize=TRUE)
ridge.pred<-predict(fit,s=cv.out$lambda.min,newx=xtst)
mse.ridge<-mean((ytest-ridge.pred)^2)

# Lasso
cv.out<-cv.glmnet(xapp,yapp,alpha=1,standardize=TRUE)
fit<-glmnet(xapp,yapp,lambda=cv.out$lambda.min,alpha=1,standardize=TRUE)
lasso.pred<-predict(fit,s=cv.out$lambda.min,newx=xtst)
mse.lasso<-mean((ytest-lasso.pred)^2)

print(c(mse.tree,mse.RF,mse.ls, mse.ridge,mse.lasso))

# Subset selection
library(leaps)
reg.forward<-regsubsets(lpsa~.-train,data=prostate[prostate$train==TRUE,],
                        method='forward',nvmax=30)
plot(reg.forward,scale="bic")
res<-summary(reg.forward)

# BIC
best<-which.min(res$bic)
ntst<-nrow(xtst)
X<-cbind(rep(1,ntst),xtst)

ypred<-X[,res$which[best,]]%*%coef(reg.forward,best)
mse.forward.bic<-mean((ypred-ytest)^2)

print(c(mse.tree,mse.ls, mse.ridge,mse.lasso,mse.forward.bic))

# Adjusted R2
plot(reg.forward,scale="adjr2")
best<-which.max(res$adjr2)

ypred<-X[,res$which[best,]]%*%coef(reg.forward,best)
mse.forward.adjr2<-mean((ypred-ytest)^2)

print(c(mse.tree,mse.RF,mse.ls, mse.ridge,mse.lasso,mse.forward.bic,mse.forward.adjr2))

# Analysis of variable importance
summary(reg)
# Plot of variable importance computed by random forests
varImpPlot(fit.RF) 



