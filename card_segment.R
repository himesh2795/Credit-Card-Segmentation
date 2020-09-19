library(NbClust)


card_df = read.csv('credit-card-data.csv')
missing_val = data.frame(apply(card_df,2,function(X){sum(is.na(X))}))

# Remove missing values
card_df = na.omit(card_df)
missing_val = data.frame(apply(card_df,2,function(X){sum(is.na(X))}))

# Removing rows where purchases are zero but they have purchase transactions and purchase transactions are zero but they have purchases
card_df <- card_df[!((card_df$PURCHASES==0 & card_df$PURCHASES_TRX!=0) | (card_df$PURCHASES!=0 & card_df$PURCHASES_TRX==0)) ,]

# removing rows where cash advance is zero but they have cash advance transactions and cash advance transactions is zero but they have cash advance
card_df <- card_df[!((card_df$CASH_ADVANCE==0 & card_df$CASH_ADVANCE_TRX!=0) | (card_df$CASH_ADVANCE!=0 & card_df$CASH_ADVANCE_TRX==0)) ,]


#########################  1. Deriving key performance indicators(KPI) #########################################
# 1.1 - monthly average purchase
# 1.2 - monthly cash advance amount
# 1.3 - purchases by type (one-off, instalments)
# 1.4 - average amount per purchase
# 1.5 - average cash advance transaction
# 1.6 - limit usage (balance to credit limit ratio)
# 1.7 - payments to minimum payments ratio

card_df_KPI = data.frame(CUST_ID = card_df$CUST_ID)


## 1.1 - monthly average purchase
card_df_KPI$monthly_avg_purchase = card_df$PURCHASES / card_df$TENURE

## 1.2 - monthly cash advance amount
card_df_KPI$monthly_cash_advance = card_df$CASH_ADVANCE / card_df$TENURE

## 1.3 - purchases by type (one-off, installments)
card_df_KPI$monthly_oneoff_purchase = card_df$ONEOFF_PURCHASES / card_df$TENURE
card_df_KPI$monthly_installment_purchase = card_df$INSTALLMENTS_PURCHASES / card_df$TENURE

## 1.4 - average amount per purchase
card_df_KPI$average_amount_per_purchase = card_df$PURCHASES/card_df$PURCHASES_TRX

# Fill nan with 0
card_df_KPI[is.na(card_df_KPI)] = 0

## 1.5 - average cash advance transaction
card_df_KPI$average_amount_per_cash_advance = card_df$CASH_ADVANCE/card_df$CASH_ADVANCE_TRX

# Fill NaN with 0
card_df_KPI[is.na(card_df_KPI)] = 0

## 1.6 - limit usage (balance to credit limit ratio)
card_df_KPI$b2c_lr = card_df$BALANCE/ card_df$CREDIT_LIMIT * 100

## 1.7 - payments to minimum payments ratio
card_df_KPI$pay_to_minpay_ratio = card_df$PAYMENTS/ card_df$MINIMUM_PAYMENTS * 100
#########################################################################################################


##### 2. Insights using KPIs #####
### 2.1 Let's find how many uses purchases, cash advance and both
values <- c(sum(card_df_KPI$monthly_avg_purchase!=0 & card_df_KPI$monthly_cash_advance==0),
            sum(card_df_KPI['monthly_cash_advance']!=0 & card_df_KPI['monthly_avg_purchase']==0),
            sum(card_df_KPI['monthly_avg_purchase']!=0 & card_df_KPI['monthly_cash_advance']!=0),
            sum(card_df_KPI['monthly_avg_purchase']==0 & card_df_KPI['monthly_cash_advance']==0)
            )
labels <- c('Purchases', 'CAdvacne', 'Both', 'Notany') 
barplot(values,
        main = "Purchases vs Cash Advance vs Both",
        ylab = "Count",
        names.arg = labels,
        col = "darkred",
        horiz = FALSE)

### 2.2 Let's compare the amounts of purchases and cash advance of monthly usages
values <- c(sum(card_df_KPI['monthly_avg_purchase']),
            sum(card_df_KPI['monthly_cash_advance'])
)
labels <- c('Purchase','Cash Advance') 
barplot(values,
        main = "Amount for Purchase and Cash Advance",
        ylab = "Total Amount",
        names.arg = labels,
        col = "darkred",
        horiz = FALSE)

#### 2.3 We have two types of purchases, so let's find out how many uses oneoff, installments or both
values <- c(sum((card_df_KPI['monthly_oneoff_purchase']!=0) & (card_df_KPI['monthly_installment_purchase']==0)),
            sum((card_df_KPI['monthly_installment_purchase']!=0) & (card_df_KPI['monthly_oneoff_purchase']==0)),
            sum((card_df_KPI['monthly_oneoff_purchase']!=0) & (card_df_KPI['monthly_installment_purchase']!=0)),
            sum((card_df_KPI['monthly_oneoff_purchase']==0) & (card_df_KPI['monthly_installment_purchase']==0))
)
labels <- c('One-off','Installments','Both', 'Notany') 
barplot(values,
        main = "One-off vs Installments vs both",
        ylab = "Count",
        names.arg = labels,
        col = "darkred",
        horiz = FALSE)

#### 2.4 Total amount for each type of Purchases
values <- c(sum(card_df_KPI['monthly_oneoff_purchase']),
            sum(card_df_KPI['monthly_installment_purchase'])
)
labels <- c('One-off','Installments') 
barplot(values,
        main = "Amount compare for On-off Purchase and Installments",
        ylab = "total Amount",
        names.arg = labels,
        col = "darkred",
        horiz = FALSE)

### 2.5 Average amount per purchase and per cash advance comparison
values <- c(sum(card_df_KPI['average_amount_per_purchase']),
            sum(card_df_KPI['average_amount_per_cash_advance'])
)
labels <- c('Purchase','Cash Advance') 
barplot(values,
        main = "Average Purchase per transaction vs Average Cash Advance per transaction",
        ylab = "Total Average Amount",
        names.arg = labels,
        col = "darkred",
        horiz = FALSE)

### 2.6 People spending more than credit limit vs under credit limit but more than 50% vs less than 50%
values <- c(sum(card_df_KPI['b2c_lr']>100),
            sum((card_df_KPI['b2c_lr']>50) & (card_df_KPI['b2c_lr']<100)),
            sum(card_df_KPI['b2c_lr']<50)
)
labels <- c('>100','<100 & >50','<50') 
barplot(values,
        main = "Limit Usages in %",
        ylab = "Count",
        names.arg = labels,
        col = "darkred",
        horiz = FALSE)

### 2.7 People who pay more than their minimum payments vs less than the minimum payment range
' Here, <100 = Not paying enough, >100 = Paying more than required'
values <- c(sum(card_df_KPI['pay_to_minpay_ratio']<100),
            sum(card_df_KPI['pay_to_minpay_ratio']>100)
)
labels <- c('<100','>100') 
barplot(values,
        main = "Payments to the Minimum Payment",
        ylab = "Count",
        names.arg = labels,
        col = "darkred",
        horiz = FALSE)


##################################################################################################


######################################## 3. Clustering ####################################
# Dropping CUST_ID as there is no need
card_df = subset(card_df,select=-c(CUST_ID))

## Feature Scaling
normalized<-function(y) {
        
        x<-y[!is.na(y)]
        
        x<-(x - min(x)) / (max(x) - min(x))
        
        y[!is.na(y)]<-x
        
        return(y)
}

card_df_norm <-apply(card_df,2,normalized)

# To get optimum number of clusters
#NbClust_res = NbClust(card_df_norm, min.nc = 2, max.nc = 20, method="kmeans")

# Kmeans Algorithm
kmeans_model = kmeans(card_df_norm, 7)






###########################################################################################