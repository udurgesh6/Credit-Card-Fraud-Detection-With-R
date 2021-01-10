# Importing the datset
creditcard <- read_csv("creditcard.csv")

# Glance at the structure of the dataset
str(creditcard)

# Convert class to a factor variable
creditcard$Class <- factor(creditcard$Class, levels = c(0,1))

#Get the summary of the data
summary(creditcard)

# Count the missing values
sum(is.na(creditcard))

# Get the distribution of the legitimate and fraud transactions
table(creditcard$Class)

#get the percentage of fraud and legitimate transactions
prop.table(table(creditcard$Class))

# Pie Chart of credit card transactions
labels <- c("legit","fraud")
labels <- paste(labels, round(100*prop.table(table(creditcard$Class)), 2))
labels <- paste0(labels,"%")

pie(table(creditcard$Class), labels, col = c("orange","red"), main = "Pie Chart of Credit Card Transactions")

# No Model Predictions
predictions <- rep.int(0, nrow(creditcard))
predictions <- factor(predictions, levels = c(0,1))

#install.packages('caret')
#install.packages('e1071')
library(caret)
library(e1071)
confusionMatrix(data = predictions, reference = creditcard$Class)


library(dplyr)
set.seed(1)
creditcard <- creditcard %>% sample_frac(0.1)
table(creditcard$Class)

library(ggplot2)
ggplot(data = creditcard, aes(x=V1, y=V2, col = Class)) + geom_point() + theme_bw() + scale_color_manual(values = c('dodgerblue2', 'red'))


#install.packages('caTools')
library(caTools)
set.seed(123)
data_sample <- sample.split(creditcard$Class, SplitRatio = 0.80)

train_data <- subset(creditcard, data_sample==TRUE)
test_data = subset(creditcard, data_sample==FALSE)

dim(train_data)
dim(test_data)

#Random Over Sampling
table(train_data$Class)

n_legit <- 22750
new_frac_legit <- 0.50
new_n_total <- n_legit/new_frac_legit
new_n_total


#install.packages('ROSE')
library(ROSE)
oversampling_result <- ovun.sample(Class ~ ., data = train_data, method = "over", N = new_n_total, seed = 2019)

oversampled_credit <- oversampling_result$data
table(oversampled_credit$Class)

ggplot(data = oversampled_credit, aes(x=V1, y=V2, col=Class)) + geom_point(position = position_jitter(width = 0.1)) + theme_bw() + scale_color_manual(values = c('dodgerblue2', 'red'))


# Random under sampling
table(train_data$Class)
n_fraud <- 35
new_frac_fraud <- 0.50
new_n_total <- n_fraud /new_frac_fraud

undersampling_result <- ovun.sample(Class ~ .,
                                    data = train_data,
                                    method = "under",
                                    N = new_n_total,
                                    seed = 2019)
undersampled_credit <- undersampling_result$data
table(undersampled_credit$Class)

ggplot(data = undersampled_credit, aes(x=V1, y=V2, col=Class)) + geom_point() + theme_bw() + scale_color_manual(values = c('dodgerblue2', 'red'))



# Both ROS and RUS

n_new <- nrow(train_data)
fraction_fraud_new <- 0.50

sampling_result <- ovun.sample(Class ~ .,
                               data = train_data,
                               method = "both",
                               N = n_new,
                               p = fraction_fraud_new,
                               seed = 2019)

sampled_credit <- sampling_result$data
table(sampled_credit$Class)
prop.table(table(sampled_credit$Class))

ggplot(data = sampled_credit, aes(x=V1, y=V2, col=Class)) + geom_point(position = position_jitter(width = 0.2)) + theme_bw() + scale_color_manual(values = c('dodgerblue2', 'red'))


# Using Synthetic Minority Oversampling Technique to balance the data
#install.packages('smotefamily')
library(smotefamily)
table(train_data$Class)


# Set the number of fraud and legitimate cases, and the desied percentage of legitimate cases
n0 <- 22750
n1 <- 35
r0 <- 0.6

# Calculate the value for the dup_size parameter of SMOTE
ntimes <- ((1-r0)/r0) * (n0/n1) - 1

smote_output = SMOTE(X = train_data[ , -c(1, 31)],
                     target = train_data$Class,
                     K = 5,
                     dup_size = ntimes)

credit_smote <- smote_output$data
colnames(credit_smote)[30] <- "Class"
prop.table(table(credit_smote$Class))


# Class distribution for original dataset
ggplot(data = credit_smote, aes(x=V1, y=V2, col=Class)) + geom_point() + theme_bw() + scale_color_manual(values = c('dodgerblue2', 'red'))


install.packages('rpart')
install.packages('rpart.plot')

library(rpart)
library(rpart.plot)
CART_model <- rpart(Class ~ . , credit_smote)

rpart.plot(CART_model, extra = 0, type = 5, tweak = 1.2)

# Predict fraud classes
predicted_val <- predict(CART_model, test_data, type = 'class')

# Build Confusion MAtrix
library(caret)
confusionMatrix(predicted_val, test_data$Class)



# Decision tree without smote
CART_model <- rpart(Class ~ . , train_data[,-1])
rpart.plot(CART_model, extra = 0, type = 5, tweak = 1.2)

# Predict fraud cases
predicted_val <- predict(CART_model, test_data[-1], type = 'class')

library(caret)
confusionMatrix(predicted_val, test_data$Class)





CART_model <- rpart(Class ~ . , credit_smote)

rpart.plot(CART_model, extra = 0, type = 5, tweak = 1.2)

# Predict fraud classes
predicted_val <- predict(CART_model, test_data, type = 'class')

# Build Confusion MAtrix
library(caret)
confusionMatrix(predicted_val, test_data$Class)


predicted_val <- predict(CART_model, creditcard[,-1], type = 'class')
confusionMatrix(predicted_val, creditcard$Class)
