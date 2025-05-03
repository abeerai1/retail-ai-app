
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# العنوان
st.title("Retail AI Demand Prediction")

# تحميل البيانات
df = pd.read_csv("retail_data.csv")
df["demand_level"] = df["demand_level"].str.strip()
df = df[df["demand_level"].isin(["High", "Low"])].copy()

# تحويل النصوص إلى أرقام
label_cols = ["product name", "category", "weather", "demand_level"]
encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# تجهيز النموذج
X = df[["hour", "product name", "weather", "quantity sold"]]
y = df["demand_level"]
model = DecisionTreeClassifier()
model.fit(X, y)

# واجهة المستخدم
st.header("جرب التنبؤ")

hour = st.number_input("الساعة", min_value=0, max_value=23)
product = st.selectbox("اسم المنتج", encoders["product name"].classes_)
weather = st.selectbox("الطقس", encoders["weather"].classes_)
quantity = st.number_input("الكمية المباعة", min_value=0)

if st.button("تنـبـؤ"):
    input_df = pd.DataFrame([{
        "hour": hour,
        "product name": encoders["product name"].transform([product])[0],
        "weather": encoders["weather"].transform([weather])[0],
        "quantity sold": quantity
    }])
    prediction = model.predict(input_df)[0]
    label = encoders["demand_level"].inverse_transform([prediction])[0]
    st.success(f"الطلب المتوقع: {label}")
