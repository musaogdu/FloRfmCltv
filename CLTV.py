# !pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

# reading data
df_=pd.read_csv("flo_data_20k.csv")
df=df_.copy()
df.describe().T
df.head()

# necessary libraries and functions
def outlier_tresholds(dataframe, variable):
    quartile1=dataframe[variable].quantile(0.01)
    quartile3=dataframe[variable].quantile(0.99)
    interquantile_range= quartile3-quartile1
    up_limit=quartile3+1.5*interquantile_range
    low_limit=quartile1-1.5*interquantile_range
    return low_limit,up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit=outlier_tresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)

# suppressing outliers
columns=["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
"customer_value_total_ever_online"]

for col in columns:
    replace_with_thresholds(df,col)

# create new variables
df["order_num_total"]= df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
df["customer_value_total"]=df["customer_value_total_ever_online"]+df["customer_value_total_ever_offline"]

# Convert variables representing dates to the type 'date

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns]=df[date_columns].apply(pd.to_datetime)
df.info()

#set the analysis date as two days after the last date of purchase
df["last_order_date"].max()
today_date=dt.datetime(2021, 6, 1)


###

cltv_df=pd.DataFrame()
cltv_df["customer_id"]=df["master_id"]
cltv_df["recency_cltv_weekly"]=((df["last_order_date"]-df["first_order_date"]).astype("timedelta64[D]"))/7
cltv_df["T_weekly"]=((today_date-df["first_order_date"]).astype("timedelta64[D]"))/7
cltv_df["frequency"]=df["order_num_total"]
cltv_df["monetary_cltv_avg"]=df["customer_value_total"]/df["order_num_total"]
cltv_df.head()

###built the BG/NBD model

bgf=BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

### 3 months expected purchase from customers

bgf.conditional_expected_number_of_purchases_up_to_time(12,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency_cltv_weekly"],
                                                        cltv_df["T_weekly"]).sort_values(ascending=False).head(10)
#or

bgf.predict(12,
            cltv_df["frequency"],
            cltv_df["recency_cltv_weekly"],
            cltv_df["T_weekly"]).sort_values(ascending=False).head(10)

cltv_df["exp_sales_3_months"]=bgf.predict(4*3,
                                          cltv_df["frequency"],
                                          cltv_df["recency_cltv_weekly"],
                                          cltv_df["T_weekly"])
### 6 months expected purchase from customers
cltv_df["exp_sales_6_months"]=bgf.predict(4*6,
                                          cltv_df["frequency"],
                                          cltv_df["recency_cltv_weekly"],
                                          cltv_df["T_weekly"])

cltv_df[["exp_sales_3_months","exp_sales_6_months"]]

cltv_df.sort_values("exp_sales_3_months",ascending=False).head(10)
cltv_df.sort_values("exp_sales_6_months",ascending=False).head(10)

###built the GAMMA GAMMA model


ggf=GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])
cltv_df["exp_average_value"]=ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                     cltv_df["monetary_cltv_avg"])

cltv_df.head()

## calculate cltv 6 months period

cltv=ggf.customer_lifetime_value(bgf,
                                 cltv_df["frequency"],
                                 cltv_df["recency_cltv_weekly"],
                                 cltv_df["T_weekly"],
                                 cltv_df["monetary_cltv_avg"],
                                 time=6,
                                 freq="W",
                                 discount_rate=0.01)

cltv_df["cltv"]=cltv

cltv_df.sort_values("cltv", ascending=False).head(20)


### segmentation into 4 groups

cltv_df["cltv_segment"]=pd.qcut(cltv_df["cltv"],4, labels=["D","C", "B", "A"])
cltv_df.head()

### functionize the whole process

def create_cltv_df(dataframe):


    columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
    for col in columns:
        replace_with_thresholds(dataframe, col)

    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    dataframe = dataframe[~(dataframe["customer_value_total"] == 0) | (dataframe["order_num_total"] == 0)]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)


    dataframe["last_order_date"].max()  # 2021-05-30
    today_date = dt.datetime(2021, 6, 1)
    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = dataframe["master_id"]
    cltv_df["recency_cltv_weekly"] = ((dataframe["last_order_date"] - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["T_weekly"] = ((today_date - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["frequency"] = dataframe["order_num_total"]
    cltv_df["monetary_cltv_avg"] = dataframe["customer_value_total"] / dataframe["order_num_total"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]


    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])
    cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])
    cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])


    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                           cltv_df['monetary_cltv_avg'])


    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)
    cltv_df["cltv"] = cltv


    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df

cltv_df = create_cltv_df(df)


cltv_df.head(10)
