import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    layout="wide",
    page_title="Traffic Predictor",
    page_icon="🚦"
)
st.title("🚦 Bengaluru Traffic Congestion Predictor")
st.subheader("Predict congestion using area, time, day and weather")


@st.cache_resource
def load_model():
    return joblib.load("traffic_model.pkl")

model = load_model()
st.sidebar.header("⚙️ Enter Traffic Details")
area = st.sidebar.selectbox("Select the Area",("Whitefield","Electronic City","Marathahalli","Silk Board","Koramangala","MG Road","Indiranagar","Hebbal","Yelahanka","Jayanagar"))
day = st.sidebar.selectbox("Choose the day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
weather = st.sidebar.radio("Weather", ["Sunny", "Rainy", "Cloudy"])
hour = st.sidebar.slider("Hour", min_value = 0,max_value = 23)
columns = ['Hour',
'Area_Electronic City','Area_Hebbal','Area_Indiranagar',
'Area_Jayanagar','Area_Koramangala','Area_MG Road',
'Area_Marathahalli','Area_Silk Board','Area_Whitefield',
'Area_Yelahanka','Day_Friday','Day_Monday','Day_Saturday',
'Day_Sunday','Day_Thursday','Day_Tuesday','Day_Wednesday',
'Weather_Cloudy','Weather_Rainy','Weather_Sunny']
col1, col2 = st.columns(2)

with col1:
    st.metric("Selected Area", area)

with col2:
    st.metric("Selected Hour", hour)
col3, col4 = st.columns(2)
st.write("")
with col3:
    st.metric("Selected Day", day)

with col4:
    st.metric("Selected Weather", weather)

st.divider()
if st.button("🚀 Predict Traffic"):
    with st.spinner("Analyzing traffic..."):
        input_data = dict.fromkeys(columns, 0)
        input_data["Area_"+area] = 1
        input_data["Day_"+day] = 1
        input_data["Weather_" + weather] = 1
        input_data["Hour"] = hour

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        if prediction[0] == "High":
            st.error("🚦 High Traffic Expected")
            st.info("Consider leaving earlier or alternate route.")

        elif prediction[0] == "Medium":
            st.warning("🚗 Medium Traffic Expected")

        else:
            st.success("🟢 Low Traffic Expected")
            
        confidence = max(model.predict_proba(input_df)[0]) * 100
        st.caption(f"Prediction Confidence: {confidence:.2f}%")
st.caption("Built by Chay | Python • ML • Streamlit")