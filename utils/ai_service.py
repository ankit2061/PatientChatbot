import os
import google.generativeai as genai
import streamlit as st

# Configure Google Gemini API
# Get API key from environment or use the provided one as fallback
def configure_genai():
    """Configure the Google Generative AI API with the API key from Streamlit secrets"""
    api_key = st.secrets["gemini"]["api_key"]
    genai.configure(api_key=api_key)

# Configure Gemini on module import
configure_genai()

def handle_errors(func):
    """Error handling decorator for AI functions"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error: {e}")
            return f"AI service error: {str(e)}"
    return wrapper

@handle_errors
def get_gemini_explanation(prompt):
    """Get explanations from Google Gemini AI"""
    try:
        # Use gemini-1.5-flash for fast responses
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Enhance the prompt with specific instructions for better responses
        enhanced_prompt = f"""
        You are a medical AI assistant helping patients understand medical and insurance terms.
        Please provide a clear, accurate, and concise explanation of:
        
        {prompt}
        
        Keep your response factual, ethical, and patient-friendly. Avoid speculation.
        """
        
        response = model.generate_content(enhanced_prompt)
        return response.text
    except Exception as e:
        return f"Gemini API Error: {e}"
