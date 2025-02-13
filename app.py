import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import nltk
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')

# Load an improved chatbot model
model_name = "microsoft/DialoGPT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Healthcare-specific responses
def healthcare_chatbot(user_input):
    user_input = user_input.lower()

    predefined_responses = {
        "symptom": "If you're experiencing symptoms, it's best to consult a doctor for an accurate diagnosis. Stay hydrated, rest well, and monitor your condition.",
        "appointment": "You can schedule an appointment with a doctor online or at your nearest hospital. Would you like help finding nearby clinics?",
        "medication": "It's important to take prescribed medications as directed. If you have concerns about dosage or side effects, consult your doctor or pharmacist.",
        "diet": "A balanced diet with fruits, vegetables, and lean proteins can boost your health. Avoid processed foods and drink plenty of water.",
        "exercise": "Regular exercise like walking, yoga, or strength training can improve overall health. Aim for at least 30 minutes daily.",
        "mental health": "Mental health is just as important as physical health. Consider meditation, talking to a friend, or seeking professional help if needed.",
        "immune system": "Boost your immune system naturally by eating vitamin-rich foods, staying active, getting enough sleep, and managing stress."
    }

    for key, response in predefined_responses.items():
        if key in user_input:
            return response

    # Generate AI-based response if no predefined answer is found
    return generate_response(user_input)

# Function to generate chatbot responses
def generate_response(user_input):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response

# Streamlit web app interface
def main():
    st.title("Healthcare Assistant Chatbot")

    user_input = st.text_input("How can I assist you today?", "")

    if st.button("Submit"):
        if user_input:
            st.write("User:", user_input)
            response = healthcare_chatbot(user_input)
            st.write("Healthcare Assistant:", response)
        else:
            st.write("Please enter a query.")

if __name__ == "__main__":
    main()
