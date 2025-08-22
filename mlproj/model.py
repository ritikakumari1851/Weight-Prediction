import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# The 'heights' list is our input data, also known as 'features' or 'X'
heights = [[1.5], [1.6], [1.7], [1.8], [1.9]]

# The 'weights' list is our output data, also known as 'labels' or 'y'
weights = [[50], [60], [68], [75], [82]]

# Create a Linear Regression model object
model = LinearRegression()

# 'model.fit()' learns the relationship between heights and weights
model.fit(heights, weights)

# --- Streamlit UI Components ---
st.title("Weight Prediction using Linear Regression")

# Use a number input widget to get height from the user
user_height = st.number_input("Enter a height in meters:", min_value=1.0, max_value=2.5, value=1.75, step=0.01)

if st.button("Predict Weight"):
    # Convert the user input to the correct format for the model
    new_height = np.array([[user_height]])
    
    # 'model.predict()' uses the learned relationship to make a prediction
    predicted_weight = model.predict(new_height)
    
    # Display the result to the user
    st.success(f"Predicted weight for {user_height:.2f}m is: {predicted_weight[0][0]:.2f} kg")

    # Optional: Display the plot
    fig, ax = plt.subplots()
    ax.scatter(heights, weights, color='blue', label='Actual Data')
    ax.plot(heights, model.predict(heights), color='red', label='Regression Line')
    ax.set_title('Height vs. Weight Prediction')
    ax.set_xlabel('Height (m)')
    ax.set_ylabel('Weight (kg)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)