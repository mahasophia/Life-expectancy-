import streamlit as st 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
st.title("Life Expectency Prediction App")
st.write("Predict life expectancy based on socio-economic and health factors")
@st.cache_data
def load_data():
    df=pd.read_csv("Life Expectancy Data.csv")
    return df 
df=load_data()
st.subheader("Dataset preview")
st.write(df.head())
st.subheader("📊 Dataset Statistics")
st.write(df.describe())
features=['Adult Mortality','Alcohol','GDP','Schooling',' HIV/AIDS']
target='Life expectancy '
X=df[features]
y=df[target]
X=X.fillna(X.mean())
y=y.fillna(y.mean())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
st.subheader("Model Performance")
st.write(f"****Mean Squared Error:***{mse:.2f}")
st.write(f"**R2 Score:**{r2:.2f}")
st.subheader("Feature importance (Coefficients)")
coeff_df=pd.DataFrame({'Feature':features,'Coefficient':model.coef_})
st.write(coeff_df)
st.subheader("Actual vs Predicted")
fig,ax=plt.subplots()
ax.scatter(y_test,y_pred)
ax.set_xlabel("Actual Life expectancy")
ax.set_title("Actual vs Predicted")
st.pyplot(fig)
st.subheader("Predict Life expectancy")
adult_mortality=st.slider("Adult Mortality",0,500,150)
alcohol=st.slider("Alcohol Consumption",0.0,20.0,5.0)
gdp=st.slider("GDP",0,100000,100000)
schooling=st.slider("Schooling(years)",0.0,20.0,10.0)
hiv=st.slider(" HIV/AIDS",0.0,50.0,1.0)
input_data=np.array([[adult_mortality,alcohol,gdp,schooling,hiv]])
prediction=model.predict(input_data)
st.success(f"Predicted Life Expectancy: {prediction[0]:.2f} years")
