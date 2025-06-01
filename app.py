from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI(
    title="Email Classification API",
    version="1.0.0",
    description="Classifies support emails into categories and masks personal information.",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Load pre-trained model
model = joblib.load("model.joblib")

# Initialize NER pipeline (multilingual)
ner = pipeline('ner', model='Davlan/xlm-roberta-base-ner-hrl', grouped_entities=True)

# Regex patterns for PII detection
EMAIL_REGEX = r'\b[\w\.-]+@[\w\.-]+\.\w{2,}\b'
AADHAAR_REGEX = r'\b\d{4}\s?\d{4}\s?\d{4}\b'
CARD_REGEX = r'\b(?:\d[ -]*?){13,19}\b'
CVV_REGEX = r'(?i)\b(?:cvv[:\s\-]*)?(\d{3,4})\b'
EXPIRY_REGEX = r'\b(0[1-9]|1[0-2])[\/\-]\d{2,4}\b'
PHONE_REGEX = r'\+?\d[\d\s\-]{7,14}\d'
DOB_REGEX = r'\b\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4}\b'

NER_TO_TOKEN = {
    'PER': 'full_name',
    'EMAIL': 'email',
    'DATE': 'dob'
}

def mask_pii(text, mapping=None, counter=None):
    if mapping is None:
        mapping = {}
    if counter is None:
        counter = {
            'full_name': 0,
            'email': 0,
            'phone_number': 0,
            'dob': 0,
            'aadhar_num': 0,
            'credit_debit_no': 0,
            'cvv_no': 0,
            'expiry_no': 0
        }

    # Mask NER entities first
    entities = ner(text)
    for ent in entities:
        label = ent['entity_group']
        if label in NER_TO_TOKEN:
            token_name = NER_TO_TOKEN[label]
            original = ent['word'].replace('##', '')
            token = f"[{token_name}_{counter[token_name]:03d}]"
            if original in text:
                text = text.replace(original, token, 1)
                mapping[token] = original
                counter[token_name] += 1

    # Mask regex patterns
    regex_map = [
        (CARD_REGEX, 'credit_debit_no'),
        (AADHAAR_REGEX, 'aadhar_num'),
        (PHONE_REGEX, 'phone_number'),
        (CVV_REGEX, 'cvv_no'),
        (EXPIRY_REGEX, 'expiry_no'),
        (EMAIL_REGEX, 'email'),
        (DOB_REGEX, 'dob')
    ]

    for regex, token_name in regex_map:
        def replacer(match):
            original = match.group(0)
            token = f"[{token_name}_{counter[token_name]:03d}]"
            counter[token_name] += 1
            mapping[token] = original
            return token
        text = re.sub(regex, replacer, text)

    return text, mapping

# Input schema
class EmailInput(BaseModel):
    input_email_body: str

# Classification Endpoint
@app.post("/classify")
def classify_email(data: EmailInput):
    raw_text = data.input_email_body

    # Masking using your advanced function
    masked_text, pii_map = mask_pii(raw_text)

    # Convert pii_map to a list for easier frontend use (optional)
    entity_list = [{"placeholder": k, "original": v} for k, v in pii_map.items()]

    # Prediction
    predicted_category = model.predict([masked_text])[0]

    # Response format
    return {
        "input_email_body": raw_text,
        "list_of_masked_entities": entity_list,
        "masked_email": masked_text,
        "category_of_the_email": predicted_category
    }

# Health check endpoint
@app.get("/")
def root():
    return {"message": "Email Classification API is running."}
