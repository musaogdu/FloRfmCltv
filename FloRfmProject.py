

import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%5f" % x)

df_=pd.read_csv("flo_data_20k.csv")
df=df_.copy()
df.head(10)
df.info
df.columns
df.isnull().sum()
df.describe().T
df.dropna
df["master_id"].nunique()

# create new variables
df["order_num_total"]= df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
df["customer_value_total"]=df["customer_value_total_ever_online"]+df["customer_value_total_ever_offline"]

# Convert variables representing dates to the type 'date
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns]=df[date_columns].apply(pd.to_datetime)
df.info()

df.groupby("order_channel").agg({"master_id": "count",
                                "order_num_total":"sum",
                                "customer_value_total": "sum"
                                 })

#top 10 highest-earning customers
df.groupby("master_id").agg({"customer_value_total" : "sum"}).sort_values("customer_value_total", ascending=False).head(10)
df.sort_values("customer_value_total", ascending=False)[:10]

#top 10 customers with the highest number of purchases
df.groupby("master_id").agg({"order_num_total": "sum"}).sort_values("order_num_total", ascending=False).head(10)
df.sort_values("order_num_total", ascending=False)[:10]


#Functionalization of the project

def function(dataframe):
    dataframe["order_num_total"] = dataframe["order_num_total_ever_offline"] + dataframe["order_num_total_ever_online"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_online"] + dataframe["customer_value_total_ever_offline"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
    return df

df["last_order_date_offline"].max()
df["last_order_date_online"].max()
today_date = dt.datetime(2021,6,2)

rfm=pd.DataFrame()
rfm["customer_id"]=df["master_id"]
rfm["recency"]= (today_date-df["last_order_date"]).astype("timedelta64[D]")
rfm["frequency"]=df["order_num_total"]
rfm["monetary"]=df["customer_value_total"]
rfm.head()

rfm.describe().T
rfm.shape
rfm["frequency_score"]=pd.qcut(rfm["frequency"].rank(method="first"),5, labels=[1,2,3,4,5])
rfm["recency_score"]=pd.qcut(rfm["recency"],5, labels=[5,4,3,2,1])
rfm["monetary_score"]=pd.qcut(rfm["monetary"],5, labels=[1,2,3,4,5])
rfm["RF_SCORE"]=(rfm["recency_score"].astype(str)+
                 rfm["frequency_score"].astype(str))
rfm.describe().T

#RFM segmentasyion
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"]=rfm["RF_SCORE"].replace(seg_map, regex=True)
rfm[["segment","recency","frequency", "monetary"]].groupby("segment").agg(["mean","count",])

target_segment_customer_ids=rfm[rfm["segment"].isin(["champions", "loyal_customers"])]["customer_id"]
cuts_ids=df[(df["master_id"].isin(target_segment_customer_ids)) & (df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
cuts_ids.to_csv("new_brand_target_customer_id.csv", index=False)

forty_percent_for_customer_ids=rfm[rfm["segment"].isin(["cant_loose", "at_risk","hibernating_", "new_customers"])]["customer_id"]
forty_cuts_ids = df[(df["master_id"].isin(forty_percent_for_customer_ids)) & ((df["interested_in_categories_12"].str.contains("ERKEK")) | (df["interested_in_categories_12"].str.contains("COCUK")))] ["master_id"]

forty_cuts_ids.to_csv("forty_percent_for_customer_ids.csv", index=False)