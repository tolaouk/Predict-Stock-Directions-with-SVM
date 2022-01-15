This report is going to predict a particular company’s stock price direction of a certain day in the future based on the rate of return of the stock price for the 10 previous days’ data of itself and other 19 companies. Below is the list of top 20 companies with highest revenue in the US in 2020. 
The period of these 20 companies is from 01/01/2015 to 12/31/2018 which is prior to the pandemic and give a more stable stock price. Also, the data will only focus on the stock price at closing time. After excluding the market closed dates, there are 1006 dates in the data set. We will denote the stock price for company j on day t as S_j (t). 
The target company is General Motor, which is the last one in Table 1. On each day “t”, S_20 (t+1) is unknown to all stock traders and the goal is to predict whether the stock price will be up or down for the next day i.e., day “t+1” with respect to the currently known S_20 (t). Thus, all the data from 20 companies will be used to build the model, since the true classification is available.

![image](https://user-images.githubusercontent.com/26745287/149601556-d430a34e-be33-42cc-b8d5-4c6ebc4da416.png)


Support Vector Machine (SVM) is the targeted algorithm to perform the task. Linear SVM and Radial Kernel SVM will both be implemented and compared on their performances.
