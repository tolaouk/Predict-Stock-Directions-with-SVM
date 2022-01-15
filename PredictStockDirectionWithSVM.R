rm(list = ls())
#============================================================================
library(e1071)
library(ggplot2)
library(smotefamily)
library(flexclust)
library(caret)
#============================================================================
# Part 1 Data Set
wd = "/Users/jerrychien/Desktop/OneDrive - University Of Houston/6350 - Statistical Learning and Data Mining/HW/HW 4/Stock Data/"
comp = c("WMT", "AMZN", "AAPL", "CVS", "XOM", "UNH", "BRK-A", "MCK", "ABC", "GOOG",
         "T", "CI", "F", "COST", "FDX", "CVX", "CAH", "MSFT", "JPM", "GM")
for (i in 1:20){
  temp = read.csv(paste(wd, comp[i], ".csv", sep = ""))[, 5]
  temp = cbind(c(1:length(temp)), temp)
  colnames(temp) = c("T", "S")
  assign(comp[i], as.data.frame(temp))
}

# Part 2 Pre-Processing
for (i in 1:20){
  temp = get(comp[i])
  temp["Y"] = 0
  for (j in 2:1006){
    temp[j, "Y"] = (temp[j, "S"] - temp[j - 1, "S"]) / temp[j - 1, "S"]
  }
  assign(comp[i], temp)
}

# Part 3 Create cases and features vectors
df = data.frame(matrix(ncol = 200, nrow = 995))
colnames(df) = c(paste("X", c(1:200), sep = ""))
for (i in 1:20){
  temp = get(comp[i])
  for (j in 2:996){
    df[j - 1, ((i - 1)*10 + 1):(i*10)] = temp[c(j:(j+9)), "Y"]
  }
}

# Part 4 Create two disjoint class of cases
comp20 = get(comp[20])
TRUC = as.factor(ifelse(comp20[c(11:1005), "Y"] >= 0.006, "HIGH", "LOW"))
class_size = data.frame()
for (i in c("LOW", "HIGH")){
  class_size["Number", i] = table(TRUC)[i]
  class_size["Percentage", i] = round(prop.table(table(TRUC))[i] * 100, 1)
}

# Part 5 PCA Analysis
CORR = cor(df)
E = eigen(CORR)
L = E$values
W = E$vector
ggplot() +
  geom_point(aes(x = c(1:200), y = L)) +
  ylab("Eigenvalue") + xlab("k") +
  scale_y_continuous(expand = c(0, 0.25)) + scale_x_continuous(expand = c(0, 2)) +
  theme(text = element_text(size = 25), plot.margin = margin(10, 15, 10, 10, "point"))
PEV = NULL
for (m in 1:200){
  PEV[m] = sum(L[c(1:m)]) / 200 * 100
}

h = sum(PEV < 90) + 1 # h to have PEV >= 90%

ggplot() +
  geom_point(aes(x = c(1:200), y = PEV)) +
  geom_segment(aes(x = h, xend = h, y = 0, yend = 90), size = 1, colour="blue") +
  geom_segment(aes(x = 0, xend = h, y = 90, yend = 90), size = 1, colour="blue") +
  geom_text(aes(x = h + 9, y = 3, label = paste("k =", h)), size = 7, color = "blue") +
  geom_text(aes(x = 15, y = 93, label = "PEV >= 90%"), size = 7, color = "blue") +
  ylab("PEV (%)") + xlab("k") +
  scale_y_continuous(expand = c(0, 1)) + scale_x_continuous(expand = c(0, 2)) +
  theme(text = element_text(size = 25), plot.margin = margin(10, 15, 10, 10, "point"))
U =  as.data.frame(as.matrix(df) %*% as.matrix(W))[, c(1:h)]

final_df = cbind(U, TRUC) 

# Part 6 Training and Test Sets
set.seed(20211125)
training_set = NULL
test_set = NULL
for (i in as.factor(c("LOW", "HIGH"))){
  temp = subset(final_df, TRUC == i)
  sample_number = sample(x = dim(temp)[1], size = round(dim(temp)[1] * 0.85))
  training_set = rbind(training_set, temp[sample_number, ])
  test_set = rbind(test_set, temp[-sample_number, ])
}

size_test_training = data.frame()
for (i in c("training_set", "test_set")){
  size_test_training["LOW", i] = dim(subset(get(i), TRUC == "LOW"))[1]
  size_test_training["HIGH", i] = dim(subset(get(i), TRUC == "HIGH"))[1]
  size_test_training["Total", i] = dim(get(i))[1]
}

# Part 7 Linear SVM
cost_list_linear = c(0.01, 0.25, 0.5, 0.75, 1, 2, 2.5, 5, 7.5, 10)
result_linear = data.frame(matrix(nrow = length(cost_list_linear), ncol = 5))
colnames(result_linear) = c("Cost", "AccTrain", "AccTest", "AccTest/AccTrain", "%Support")
for (i in 1:length(cost_list_linear)){
  start_time = Sys.time()
  linear_svm = svm(x = training_set[, -119], y = training_set[, 119], kernel = "linear", cost = cost_list_linear[i])
  end_time = Sys.time()
  linear_pred_training = predict(linear_svm, training_set[, -119])
  linear_pred_test = predict(linear_svm, test_set[, -119])
  AccTrain = round(mean(linear_pred_training == training_set[, 119]) * 100, 1)
  AccTest = round(mean(linear_pred_test == test_set[, 119]) * 100, 1)
  result_linear[i, "Cost"] = linear_svm$cost
  result_linear[i, "AccTrain"] = AccTrain
  result_linear[i, "AccTest"] = AccTest
  result_linear[i, "AccTest/AccTrain"] = round((AccTest / AccTrain) * 100, 1)
  result_linear[i, "%Support"] = round((sum(linear_svm$nSV) / dim(training_set)[1]) * 100, 1)
  result_linear[i, "Computing Time"] = round(end_time - start_time, 2)
}

# Part 8 Plot
ggplot() +
  geom_line(aes(x = cost_list_linear, y = unlist(result_linear["AccTrain"]), col = "Acc Train"), size = 1) +
  geom_line(aes(x = cost_list_linear, y = unlist(result_linear["AccTest"]), col = "Acc Test"), size = 1) +
  geom_line(aes(x = cost_list_linear, y = unlist(result_linear["AccTest/AccTrain"]), col = "Acc Ratio"), size = 1) +
  geom_line(aes(x = cost_list_linear, y = unlist(result_linear["%Support"]), col = "%Support"), size = 1) +
  ylab("%") + xlab("Cost") +
  scale_color_manual(name = "Legend", values = c("Acc Train" = "green", "Acc Test" = "blue", "Acc Ratio" = "red", "%Support" = "black")) +
  scale_x_continuous(breaks = cost_list_linear, expand = c(0.01, 0.01)) +
  theme(text = element_text(size = 25), legend.position = c(0.9, 0.5), legend.text = element_text(size = 25), legend.title = element_text(size = 25)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
cost_list_linear_new = c(0.1, 0.8, 3, 3.5, 4, 6, 7, 8, 9, 15, 20, 25)
result_linear_new = data.frame(matrix(nrow = length(cost_list_linear_new), ncol = 5))
colnames(result_linear_new) = c("Cost", "AccTrain", "AccTest", "AccTest/AccTrain", "%Support")
for (i in 1:length(cost_list_linear_new)){
  start_time = Sys.time()
  linear_svm_new = svm(x = training_set[, -119], y = training_set[, 119], kernel = "linear", cost = cost_list_linear_new[i])
  end_time = Sys.time()
  linear_pred_training_new = predict(linear_svm_new, training_set[, -119])
  linear_pred_test_new = predict(linear_svm_new, test_set[, -119])
  AccTrain = round(mean(linear_pred_training_new == training_set[, 119]) * 100, 1)
  AccTest = round(mean(linear_pred_test_new == test_set[, 119]) * 100, 1)
  result_linear_new[i, "Cost"] = linear_svm_new$cost
  result_linear_new[i, "AccTrain"] = AccTrain
  result_linear_new[i, "AccTest"] = AccTest
  result_linear_new[i, "AccTest/AccTrain"] = round((AccTest / AccTrain) * 100, 1)
  result_linear_new[i, "%Support"] = round((sum(linear_svm_new$nSV) / dim(training_set)[1]) * 100, 1)
  result_linear_new[i, "Computing Time"] = round(end_time - start_time, 2)
}

ggplot() +
  geom_line(aes(x = cost_list_linear_new, y = unlist(result_linear_new["AccTrain"]), col = "Acc Train"), size = 1) +
  geom_line(aes(x = cost_list_linear_new, y = unlist(result_linear_new["AccTest"]), col = "Acc Test"), size = 1) +
  geom_line(aes(x = cost_list_linear_new, y = unlist(result_linear_new["AccTest/AccTrain"]), col = "Acc Ratio"), size = 1) +
  geom_line(aes(x = cost_list_linear_new, y = unlist(result_linear_new["%Support"]), col = "%Support"), size = 1) +
  ylab("%") + xlab("C") +
  scale_color_manual(name = "Legend", values = c("Acc Train" = "green", "Acc Test" = "blue", "Acc Ratio" = "red", "%Support" = "black")) +
  scale_x_continuous(breaks = cost_list_linear_new, expand = c(0.01, 0)) +
  theme(text = element_text(size = 25), legend.position = c(0.9, 0.5), legend.text = element_text(size = 25), legend.title = element_text(size = 25)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
# Part 9 Radial Kernel
## Part 9.1 Hilbert Distance
training_HIGH = subset(training_set, TRUC == "HIGH")
training_LOW = subset(training_set, TRUC == "LOW")
gamma_list = c(0.001, 0.005, 10, 100)
for (i in 1:length(gamma_list)){
  SIM = exp(-gamma_list[i] * (dist2(training_HIGH[, -119], training_LOW[, -119]))^2)
  Hdist = sqrt(2 - 2*SIM)
  assign(paste("H_dist_", i, sep = ""), c(Hdist))
}

ggplot() +
  geom_histogram(aes(x = H_dist_1, fill = "0.001"), alpha = 0.5, bins = 50) +
  geom_histogram(aes(x = H_dist_2, fill = "0.005"), alpha = 0.5, bins = 50) +
  geom_histogram(aes(x = H_dist_3, fill = "10"), alpha = 0.5, bins = 50) +
  geom_histogram(aes(x = H_dist_4, fill = "100"), alpha = 0.5, bins = 50) +
  ggtitle("Histogram of H Distance with different Gamma") +
  scale_fill_manual(name = "Gamma", values = c("0.001" = "green", "0.005" = "blue", "10" = "red", "100" = "black")) +
  scale_x_continuous(breaks = c(0, 0.5, 1, 1.414)) +
  theme(text = element_text(size = 25), legend.position = c(0.85, 0.8), legend.text = element_text(size = 25), legend.title = element_text(size = 25), plot.title = element_text(hjust = 0.5)) + 
  ylab("Density") + xlab("H Distance")
gamma_list_radial = c(0.005, 0.01, 0.1, 1)

## Part 9.2 Error Weight Coefficient
cost_list_radial = c(0.0001, 0.001, 0.01, 0.1, 0.5, 0.75, 1, 5, 10, 20)

## Part 9.3 Apply Radial Kernel SVM
for (i in 1:length(gamma_list_radial)){
  temp = data.frame()
  for (j in 1:length(cost_list_radial)){
    start_time = Sys.time()
    radial_svm = svm(x = training_set[, -119], y = training_set[, 119], kernel = "radial", gamma = gamma_list_radial[i], cost = cost_list_radial[j])
    end_time = Sys.time()
    radial_pred_training = predict(radial_svm, training_set[, -119])
    radial_pred_test = predict(radial_svm, test_set[, -119])
    AccTrain = round(mean(radial_pred_training == training_set[, 119]) * 100, 1)
    AccTest = round(mean(radial_pred_test == test_set[, 119]) * 100, 1)
    temp[j, "Cost"] = radial_svm$cost
    temp[j, "AccTrain"] = AccTrain
    temp[j, "AccTest"] = AccTest
    temp[j, "AccTest/AccTrain"] = round((AccTest / AccTrain) * 100, 1)
    temp[j, "%Support"] = round((sum(radial_svm$nSV) / dim(training_set)[1]) * 100, 1)
    temp[j, "Computing Time"] = round(end_time - start_time, 2)
  }
  assign(paste("result_radial_gamma_", i, sep = ""), temp)
}

# Part 10 
## First pair (C = 0.0001, Gamma = 0.005) Reason: Lowest % of Support Vector with 100% Accuracy Ratio
gamma_pair_1 = c(0.003, 0.004, 0.005, 0.006, 0.007)
C_pair_1 = c(0.0003, 0.0002, 0.0001, 0.00009, 0.00008)
pair_1 = data.frame()
for (i in 1:5){
  pair_1_svm = svm(x = training_set[, -119], y = training_set[, 119], kernel = "radial", gamma = gamma_pair_1[i], cost = C_pair_1[i])
  radial_pred_training = predict(pair_1_svm, training_set[, -119])
  radial_pred_test = predict(pair_1_svm, test_set[, -119])
  AccTrain = round(mean(radial_pred_training == training_set[, 119]) * 100, 1)
  AccTest = round(mean(radial_pred_test == test_set[, 119]) * 100, 1)
  pair_1[i, "Cost"] = pair_1_svm$cost
  pair_1[i, "Gamma"] = pair_1_svm$gamma
  pair_1[i, "AccTrain"] = AccTrain
  pair_1[i, "AccTest"] = AccTest
  pair_1[i, "AccTest/AccTrain"] = round((AccTest / AccTrain) * 100, 1)
  pair_1[i, "%Support"] = round((sum(pair_1_svm$nSV) / dim(training_set)[1]) * 100, 1)
  pair_1[i, "Computing Time"] = round(end_time - start_time, 2)
}

## Second pair (C = 0.5, Gamma = 0.005) Reason: Both Test Accuracy and Training Accuracy start to increase comparing lower C at same gamma.
gamma_pair_2 = c(0.003, 0.004, 0.005, 0.006, 0.007)
C_pair_2 = c(0.52, 0.51, 0.5, 0.49, 0.48)
pair_2 = data.frame()
for (i in 1:5){
  pair_2_svm = svm(x = training_set[, -119], y = training_set[, 119], kernel = "radial", gamma = gamma_pair_2[i], cost = C_pair_2[i])
  radial_pred_training = predict(pair_2_svm, training_set[, -119])
  radial_pred_test = predict(pair_2_svm, test_set[, -119])
  AccTrain = round(mean(radial_pred_training == training_set[, 119]) * 100, 1)
  AccTest = round(mean(radial_pred_test == test_set[, 119]) * 100, 1)
  pair_2[i, "Cost"] = pair_2_svm$cost
  pair_2[i, "Gamma"] = pair_2_svm$gamma
  pair_2[i, "AccTrain"] = AccTrain
  pair_2[i, "AccTest"] = AccTest
  pair_2[i, "AccTest/AccTrain"] = round((AccTest / AccTrain) * 100, 1)
  pair_2[i, "%Support"] = round((sum(pair_2_svm$nSV) / dim(training_set)[1]) * 100, 1)
  pair_2[i, "Computing Time"] = round(end_time - start_time, 2)
}

## Third pair (C = 1, Gamma = 0.005) Reason: Highest Test Accuracy over all pairs
gamma_pair_3 = c(0.003, 0.004, 0.005, 0.006, 0.007)
C_pair_3 = c(1.2, 1.1, 1, 0.9, 0.8)
pair_3 = data.frame()
for (i in 1:5){
  pair_3_svm = svm(x = training_set[, -119], y = training_set[, 119], kernel = "radial", gamma = gamma_pair_3[i], cost = C_pair_3[i])
  radial_pred_training = predict(pair_3_svm, training_set[, -119])
  radial_pred_test = predict(pair_3_svm, test_set[, -119])
  AccTrain = round(mean(radial_pred_training == training_set[, 119]) * 100, 1)
  AccTest = round(mean(radial_pred_test == test_set[, 119]) * 100, 1)
  pair_3[i, "Cost"] = pair_3_svm$cost
  pair_3[i, "Gamma"] = pair_3_svm$gamma
  pair_3[i, "AccTrain"] = AccTrain
  pair_3[i, "AccTest"] = AccTest
  pair_3[i, "AccTest/AccTrain"] = round((AccTest / AccTrain) * 100, 1)
  pair_3[i, "%Support"] = round((sum(pair_3_svm$nSV) / dim(training_set)[1]) * 100, 1)
  pair_3[i, "Computing Time"] = round(end_time - start_time, 2)
}

## Fourth pair (C = 10, Gamma = 0.005) Reason: High Test Accuracy with perfect Training Accuracy and lower % of Support Vector
gamma_pair_4 = c(0.003, 0.004, 0.005, 0.006, 0.007)
C_pair_4 = c(12, 11, 10, 9, 8)
pair_4 = data.frame()
for (i in 1:5){
  pair_4_svm = svm(x = training_set[, -119], y = training_set[, 119], kernel = "radial", gamma = gamma_pair_4[i], cost = C_pair_4[i])
  radial_pred_training = predict(pair_4_svm, training_set[, -119])
  radial_pred_test = predict(pair_4_svm, test_set[, -119])
  AccTrain = round(mean(radial_pred_training == training_set[, 119]) * 100, 1)
  AccTest = round(mean(radial_pred_test == test_set[, 119]) * 100, 1)
  pair_4[i, "Cost"] = pair_4_svm$cost
  pair_4[i, "Gamma"] = pair_4_svm$gamma
  pair_4[i, "AccTrain"] = AccTrain
  pair_4[i, "AccTest"] = AccTest
  pair_4[i, "AccTest/AccTrain"] = round((AccTest / AccTrain) * 100, 1)
  pair_4[i, "%Support"] = round((sum(pair_4_svm$nSV) / dim(training_set)[1]) * 100, 1)
  pair_4[i, "Computing Time"] = round(end_time - start_time, 2)
}

## Fifth pair (C = 1, Gamma = 0.01) Reason: Almost perfect Training Accuracy but low Accuracy Ratio
gamma_pair_5 = c(0.008, 0.009, 0.01, 0.011, 0.012)
C_pair_5 = c(1.2, 1.1, 1, 0.9, 0.8)
pair_5 = data.frame()
for (i in 1:5){
  pair_5_svm = svm(x = training_set[, -119], y = training_set[, 119], kernel = "radial", gamma = gamma_pair_5[i], cost = C_pair_5[i])
  radial_pred_training = predict(pair_5_svm, training_set[, -119])
  radial_pred_test = predict(pair_5_svm, test_set[, -119])
  AccTrain = round(mean(radial_pred_training == training_set[, 119]) * 100, 1)
  AccTest = round(mean(radial_pred_test == test_set[, 119]) * 100, 1)
  pair_5[i, "Cost"] = pair_5_svm$cost
  pair_5[i, "Gamma"] = pair_5_svm$gamma
  pair_5[i, "AccTrain"] = AccTrain
  pair_5[i, "AccTest"] = AccTest
  pair_5[i, "AccTest/AccTrain"] = round((AccTest / AccTrain) * 100, 1)
  pair_5[i, "%Support"] = round((sum(pair_5_svm$nSV) / dim(training_set)[1]) * 100, 1)
  pair_5[i, "Computing Time"] = round(end_time - start_time, 2)
}

# Part 11
best_svm = svm(x = training_set[, -119], y = training_set[, 119], kernel = "linear", cost = 6)
best_pred_training = predict(best_svm, training_set[, -119])
best_pred_test = predict(best_svm, test_set[, -119])
conf_training= round(prop.table(confusionMatrix(data = best_pred_training, reference = training_set[, 119])$table, margin = 2) * 100, 1)
conf_test= round(prop.table(confusionMatrix(data = best_pred_test, reference = test_set[, 119])$table, margin = 2) * 100, 1)

sv_percentage = data.frame()
sv_percentage["HIGH", "%Support"] = round(best_svm$nSV[1] / dim(training_set)[1] * 100, 1)
sv_percentage["LOW", "%Support"] = round(best_svm$nSV[2] / dim(training_set)[1] * 100, 1)

# Part 12
ggplot() +
  geom_line(aes(x = comp20[, "T"], y = comp20[, "S"])) +
  scale_x_continuous(breaks = c(1, 250, 500, 750, 1006)) +
  ylab("S (Stock Price)") + xlab("Day") +
  theme(text = element_text(size = 25))
ggplot() +
  geom_line(aes(x = comp20[, "T"], y = comp20[, "Y"])) +
  scale_x_continuous(breaks = c(2, 250, 500, 750, 1006)) +
  ylab("Y (Rate of Return)") + xlab("Day") +
  theme(text = element_text(size = 25))
ggplot() +
  geom_point(aes(x = training_set[, 1], y = training_set[, 2], col = training_set[, 119])) +
  scale_color_manual(name ="True Class", values = c("HIGH" = "red", "LOW" = "blue")) +
  ylab("2nd Principal Component") + xlab("1st Principal Component") +
  theme(text = element_text(size = 25)) +
  theme(legend.position = c(0.91, 0.125), legend.text = element_text(size = 25), legend.title = element_text(size = 25), plot.title = element_text(hjust = 0.5), plot.margin = margin(10, 15, 10, 10, "point"))
ggplot() +
  geom_point(aes(x = test_set[, 1], y = test_set[, 2], col = test_set[, 119])) +
  scale_color_manual(name ="True Class", values = c("HIGH" = "red", "LOW" = "blue")) +
  ylab("2nd Principal Component") + xlab("1st Principal Component") +
  theme(text = element_text(size = 25)) +
  theme(legend.position = c(0.91, 0.125), legend.text = element_text(size = 25), legend.title = element_text(size = 25), plot.title = element_text(hjust = 0.5), plot.margin = margin(10, 15, 10, 10, "point"))
ggplot() +
  geom_point(aes(x = training_set[, 1], y = training_set[, 2], col = best_pred_training)) +
  scale_color_manual(name ="Prediction", values = c("HIGH" = "red", "LOW" = "blue")) +
  ylab("2nd Principal Component") + xlab("1st Principal Component") +
  theme(text = element_text(size = 25)) +
  theme(legend.position = c(0.91, 0.125), legend.text = element_text(size = 25), legend.title = element_text(size = 25), plot.title = element_text(hjust = 0.5), plot.margin = margin(10, 15, 10, 10, "point"))
ggplot() +
  geom_point(aes(x = test_set[, 1], y = test_set[, 2], col = best_pred_test)) +
  scale_color_manual(name ="Prediction", values = c("HIGH" = "red", "LOW" = "blue")) +
  ylab("2nd Principal Component") + xlab("1st Principal Component") +
  theme(text = element_text(size = 25)) +
  theme(legend.position = c(0.91, 0.125), legend.text = element_text(size = 25), legend.title = element_text(size = 25), plot.title = element_text(hjust = 0.5), plot.margin = margin(10, 15, 10, 10, "point"))
# Part 13 
## Part 13.2
best_svm_sv = training_set[rownames(best_svm$SV), ]

ggplot() +
  geom_point(aes(x = training_set[, 1], y = training_set[, 2], col = best_pred_training)) +
  geom_point(aes(x = best_svm_sv[, 1], y = best_svm_sv[, 2], shape = "SV"), size = 4, col ="green", stroke = 1) +
  scale_color_manual(name = "Training Set", values = c("HIGH" = "red", "LOW" = "blue")) +
  scale_shape_manual(name = "Support Vector", values = c("SV" = 1)) +
  ylab("2nd Principal Component") + xlab("1st Principal Component") +
  theme(text = element_text(size = 25)) +
  theme(legend.position = c(0.89, 0.225), legend.text = element_text(size = 25), legend.title = element_text(size = 25), plot.title = element_text(hjust = 0.5), plot.margin = margin(10, 15, 10, 10, "point"))
## Part 13.3
TRUC_pred = cbind(test_set[119], best_pred_test)
mis_low_as_high = test_set[rownames(subset(TRUC_pred, TRUC == "LOW" & TRUC != best_pred_test)), ]

ggplot() +
  geom_point(aes(x = test_set[, 1], y = test_set[, 2], col = test_set[, 119])) +
  geom_point(aes(x = mis_low_as_high[, 1], y = mis_low_as_high[, 2], shape = "Low as High"), size = 4, col ="black", stroke = 1) +
  scale_color_manual(name = "True Class", values = c("HIGH" = "red", "LOW" = "blue")) +
  scale_shape_manual(name = "Misclassified", values = c("Low as High" = 1)) +
  ylab("2nd Principal Component") + xlab("1st Principal Component") +
  theme(text = element_text(size = 25)) +
  theme(legend.position = c(0.89, 0.225), legend.text = element_text(size = 25), legend.title = element_text(size = 25), plot.title = element_text(hjust = 0.5))
## Part 13.4
mis_high_as_low = test_set[rownames(subset(TRUC_pred, TRUC == "HIGH" & TRUC != best_pred_test)), ]

ggplot() +
  geom_point(aes(x = test_set[, 1], y = test_set[, 2], col = test_set[, 119])) +
  geom_point(aes(x = mis_high_as_low[, 1], y = mis_high_as_low[, 2], shape = "High as Low"), size = 4, col ="black", stroke = 1) +
  scale_color_manual(name = "True Class", values = c("HIGH" = "red", "LOW" = "blue")) +
  scale_shape_manual(name = "Misclassified", values = c("High as Low" = 1)) +
  ylab("2nd Principal Component") + xlab("1st Principal Component") +
  theme(text = element_text(size = 25)) +
  theme(legend.position = c(0.89, 0.225), legend.text = element_text(size = 25), legend.title = element_text(size = 25), plot.title = element_text(hjust = 0.5))
# Part 14 
linear_computing = result_linear[, c("Cost", "Computing Time")]
radial_computing_1 = result_radial_gamma_1[, c("Cost", "Computing Time")]
radial_computing_2 = result_radial_gamma_2[, c("Cost", "Computing Time")]
radial_computing_3 = result_radial_gamma_3[, c("Cost", "Computing Time")]
radial_computing_4 = result_radial_gamma_4[, c("Cost", "Computing Time")]
