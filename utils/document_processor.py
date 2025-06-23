import os
import re
import cv2
import pytesseract
import pdfplumber
from PIL import Image

def handle_errors(func):
    """Error handling decorator for document processing functions"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error in {func.__name__}: {str(e)}")
    return wrapper

@handle_errors
def extract_text_from_image(image_path):
    """Extract text from images using OCR"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image")
    
    # Convert to grayscale for better OCR results
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply some preprocessing to improve OCR
    # Apply thresholding to handle different lighting conditions
    _, threshold = cv2.threshold(gray, a0=0, a1=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Use pytesseract to extract text
    text = pytesseract.image_to_string(threshold).strip()
    
    if not text:
        # Try again with the original grayscale image
        text = pytesseract.image_to_string(gray).strip()
    
    return text

@handle_errors
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF documents"""
    with pdfplumber.open(pdf_path) as pdf:
        # Extract text from each page and join with newlines
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

@handle_errors
def process_upload(file_path):
    """Process uploaded documents and extract relevant information"""
    
    # Validate file type
    if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
        raise ValueError("Unsupported file format. Supported types: PDF, PNG, JPG, JPEG")

    # Extract text based on file type
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verify the image is valid
        except Exception as e:
            raise ValueError(f"Invalid image file: {str(e)}")
        
        text = extract_text_from_image(file_path)
    else:
        text = extract_text_from_pdf(file_path)

    # Check if text extraction succeeded
    if not text.strip():
        raise ValueError("No text found in document")

    # Define extraction patterns for different fields
    patterns = {
        "name": r"Name:\s*([^\n]+)",
        "age": r"Age:\s*(\d+)",
        "patient_id": r"Insurance ID:\s*([A-Z0-9-]+)",
        "disease": r"Disease Name:\s*([^\n]+)",
        "gender": r"Gender:\s*([^\n]+)",
        "blood": r"Blood( Group)?:\s*([^\n]+)",
        "address": r"Address:\s*([^\n]+)",
        "phone": r"(Phone|Contact)( Number)?:\s*([^\n]+)",
        "medicines": r"Medication[s]?:\s*([^\n]+)"
    }
    
    # Extract fields using patterns
    fields = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # If the pattern has multiple capture groups, use the last one
            group_index = len(match.groups())
            fields[key] = match.group(group_index).strip() 
        else:
            fields[key] = "Not found"
    
    # Try alternative patterns for patient ID if not found
    if fields["patient_id"] == "Not found":
        alt_patterns = [
            r"ID[:\s]*([A-Z0-9-]+)",
            r"Patient[:\s]*([A-Z0-9-]+)",
            r"Record[:\s]*([A-Z0-9-]+)"
        ]
        
        for pattern in alt_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields["patient_id"] = match.group(1).strip()
                break
    
    # If we still couldn't find a patient ID, raise an error
    if fields["patient_id"] == "Not found":
        raise ValueError("Could not find Insurance ID in document")
            
    return fields, text
