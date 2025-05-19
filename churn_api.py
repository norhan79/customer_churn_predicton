import gradio as gr
import pandas as pd
import numpy as np
import pickle
import os
import transformers
from transformers import pipeline

import joblib

# Load the machine learning model and preprocessor
model_path = "/home/ubuntu/Hend/chatbots/Study/logistic_smote_model.pkl"
preprocessor_path = "/home/ubuntu/Hend/chatbots/Study/preprocessor_lr.pkl"
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

# Initialize text generation model for recommendations
# try:
#     recommender = pipeline("text-generation", model="gpt2")
# except Exception as e:
#     # Fallback if transformer model can't be loaded
#     print(f"Could not load transformer model: {e}")
recommender = None

# Customer retention recommendations using LLM where possible
def get_retention_recommendations(customer_data, churn_prob):
    recommendations = []
    
    # Categorize the probability into risk levels
    if churn_prob > 0.7:
        risk_level = "Very High"
    elif churn_prob > 0.5:
        risk_level = "High"
    elif churn_prob > 0.3:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    
    # Create a context for LLM recommendation generation
    context = f"Customer profile: Contract type: {customer_data['Contract']}, "
    context += f"Internet service: {customer_data['InternetService']}, "
    context += f"Tenure: {customer_data['tenure']} months, "
    context += f"Monthly charges: ${customer_data['MonthlyCharges']}, "
    context += f"Churn probability: {churn_prob:.2f}"
    
    # Try to generate personalized recommendations with LLM
    if recommender:
        try:
            prompt = f"{context}\nProvide three customer retention strategies:"
            llm_output = recommender(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
            
            # Extract just the generated recommendations (remove the prompt)
            generated_recommendations = llm_output.replace(prompt, "").strip().split("\n")
            
            # Clean up and add valid recommendations
            for rec in generated_recommendations:
                rec = rec.strip()
                if rec and not rec.isspace() and len(rec) > 10:
                    if not any(rec.lower() in existing.lower() for existing in recommendations):
                        recommendations.append(rec)
        except Exception as e:
            print(f"LLM recommendation generation failed: {e}")
    
    # Backup rule-based recommendations if LLM fails or doesn't provide enough
    if not recommendations or len(recommendations) < 2:
        # Recommendations based on contract duration
        if customer_data["Contract"] == "Month-to-month":
            recommendations.append("Offer special discount when upgrading to annual contract")
        
        # Recommendations based on services used
        if customer_data["InternetService"] == "Fiber optic":
            if customer_data["OnlineSecurity"] == "No":
                recommendations.append("Offer 3-month free trial of online security service")
            if customer_data["TechSupport"] == "No":
                recommendations.append("Provide one month of free technical support")
        
        # Recommendations based on tenure
        if customer_data["tenure"] < 12:
            recommendations.append("Send appreciation gift for new customers")
        
        # Recommendations based on monthly charges
        if customer_data["MonthlyCharges"] > 70:
            recommendations.append("Offer discounted plan with the same core services")
    
    # Add general recommendations if the list is still empty
    if not recommendations:
        recommendations = [
            "Contact the customer to ensure satisfaction with services",
            "Provide customized promotional offers",
            "Invite the customer to participate in the loyalty program"
        ]
    
    return risk_level, recommendations

# Define the user interface using Gradio
def predict_churn(senior_citizen, partner, dependents, tenure,
                multiple_lines, internet_service, online_security, online_backup,
                device_protection, tech_support, streaming_tv, streaming_movies,
                contract, paperless_billing, payment_method, monthly_charges):
    
    # Convert data to DataFrame - removed 'gender', 'PhoneService', and 'TotalCharges'
    input_data = {
        "SeniorCitizen": 1 if senior_citizen else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges
    }
    
    input_df = pd.DataFrame([input_data])
    try:
        # Preprocess the input data using the loaded preprocessor
        processed_input = preprocessor.transform(input_df)
        
        # Predict churn probability
        churn_prob = model.predict_proba(processed_input)[0][1]
        churn_class = model.predict(processed_input)[0]
        
        # Get recommendations
        risk_level, recommendations = get_retention_recommendations(input_data, churn_prob)
        
        # Prepare the result
        if churn_class == 1:
            prediction_text = "Customer is predicted to churn"
            result_color = "red"
        else:
            prediction_text = "Customer is predicted to stay"
            result_color = "green"
        
        # Format the final result
        result_html = f"""
        <div>
            <h3 style='color: {result_color};'>{prediction_text}</h3>
            <p><strong>Churn Probability:</strong> {churn_prob:.2%}</p>
            <p><strong>Risk Level:</strong> {risk_level}</p>
            
            <h4>Customer Retention Recommendations:</h4>
            <ul>
            {"".join([f"<li>{rec}</li>" for rec in recommendations])}
            </ul>
        </div>
        """
        
        return result_html
        
    except Exception as e:
        return f"<div style='color: red;'>Error occurred: {str(e)}</div>"

# Create the user interface - removed 'gender', 'PhoneService', and 'TotalCharges' fields
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Customer Churn Prediction Model with Retention Recommendations
    Enter customer data to predict churn probability and get retention recommendations
    """)
    
    with gr.Row():
        with gr.Column():
            senior_citizen = gr.Checkbox(label="Senior Citizen")
            partner = gr.Radio(label="Has Partner", choices=["Yes", "No"], value="No")
            dependents = gr.Radio(label="Has Dependents", choices=["Yes", "No"], value="No")
            tenure = gr.Slider(label="Tenure (months)", minimum=0, maximum=72, value=12)
            multiple_lines = gr.Radio(label="Multiple Lines", choices=["Yes", "No", "No phone service"], value="No")
            internet_service = gr.Radio(label="Internet Service", choices=["DSL", "Fiber optic", "No"], value="Fiber optic")
            online_security = gr.Radio(label="Online Security", choices=["Yes", "No", "No internet service"], value="No")
            online_backup = gr.Radio(label="Online Backup", choices=["Yes", "No", "No internet service"], value="No")
            monthly_charges = gr.Slider(label="Monthly Charges", minimum=0, maximum=150, value=70)

        with gr.Column():
            device_protection = gr.Radio(label="Device Protection", choices=["Yes", "No", "No internet service"], value="No")
            tech_support = gr.Radio(label="Tech Support", choices=["Yes", "No", "No internet service"], value="No")
            streaming_tv = gr.Radio(label="Streaming TV", choices=["Yes", "No", "No internet service"], value="No")
            streaming_movies = gr.Radio(label="Streaming Movies", choices=["Yes", "No", "No internet service"], value="No")
            contract = gr.Radio(label="Contract Type", choices=["Month-to-month", "One year", "Two year"], value="Month-to-month")
            paperless_billing = gr.Radio(label="Paperless Billing", choices=["Yes", "No"], value="Yes")
            payment_method = gr.Radio(label="Payment Method", choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], value="Electronic check")
    
    predict_btn = gr.Button("Predict")
    result = gr.HTML(label="Result")
    
    predict_btn.click(
        fn=predict_churn,
        inputs=[senior_citizen, partner, dependents, tenure,
                multiple_lines, internet_service, online_security, online_backup,
                device_protection, tech_support, streaming_tv, streaming_movies,
                contract, paperless_billing, payment_method, monthly_charges],
        outputs=result
    )

# Run the application
if __name__ == "__main__":
    demo.launch(share=True, debug=True)