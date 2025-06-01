Email Classification for Support Team

1. Code Implementation:
The project has three core Python components: `app.py`, `email_classifier_final.ipynb`, and `model.joblib`. The `email_classifier_final.ipynb` notebook was utilized in Google Colab to carry out end-to-end data processing, masking, model training, and evaluation. A hybrid masking strategy was employed: Named Entity Recognition (NER) through the `xlm-roberta-base-ner-hrl` transformer model to identify entities such as full names, email addresses, and dates of birth and regular expressions (regex) to mask Aadhaar numbers, credit/debit card numbers, CVV, expiry dates, phone numbers, etc. The masked entities are saved in a structured mapping for potential restoration. Masked text was utilized to train a `LinearSVC` model with a `TfidfVectorizer`, which are both pipelined for convenience. The dataset was upsampling balanced, separated into train/test sets, and tested utilizing classification metrics as well as a confusion matrix. The trained model was serialized to `model.joblib` with `joblib`. The `app.py` script encloses this pipeline in a FastAPI application that provides a `/classify` endpoint, loads the model, uses the same masking function, and responds as JSON with the masked email, found entities, and classified category. This API was containerized and deployed to Hugging Face Spaces by way of a Dockerfile to meet the assignment requirement of a backend-only deployed API with strict input-output formatting and no frontend interface.

2. Report:
In this assignment, the task was to design an intelligent and secure email categorization system for a company's support department. The core idea was to create an automated backend system that could sort incoming support emails into one of the four pre-defined categories: Incident, Request, Change and Problem. In addition to categorization, another core requirement was the implementation of a strong Personal Identifiable Information (PII) and masking system.The masking logic would have to be done without depending on Large Language Models (LLMs), and the whole solution was to be hosted as a backend-only API on Hugging Face Spaces. This provided data privacy as well as real-time scalability of the system.

2.1 PII Masking:
A hybrid PII masking approach was followed in order to deal with sensitive information in a secure manner. Named Entity Recognition (NER) has been employed to identify entities such as `full_name`, `email`, and `dob` with the help of the `xlm-roberta-base-ner-hrl` model. For monetary and numerical information such as `credit_debit_no`, `aadhar_num`, `cvv_no`, `expiry_no`, and `phone_number`, custom **regular expressions (regex)** were implemented to precisely capture and replace the values with normalized placeholders such as `[credit_debit_no_001]` or `[aadhar_num_002]`. Every masked token was also kept in a structured mapping so that the original values could be recovered if necessary. The masking logic was done purely without any LLMs so that the constraints set by the assignment were met.

2.2 Email Classification:
For the classification of emails, a conventional machine learning pipeline was selected, pairing `TfidfVectorizer` and `LinearSVC`. Once the sensitive information was masked, the resultant email content was vectorized into numerical features using the TF-IDF approach. These attributes were then input to a Support Vector Classifier (SVC) that was trained to differentiate between the support categories on the basis of semantic and contextual attributes. The model was trained on a labeled data set where every email was tagged with one of the four classes: Incident, Request, Change, or Problem. After training, the model was serialized through `joblib` and served as part of the API.

2.3. Model Selection and Training:
The chosen model for this email classification problem was Linear Support Vector Classifier (LinearSVC) with the `scikit-learn` library. It was combined with a TfidfVectorizer to transform the masked email body into numerical feature vectors by extracting unigrams and bigrams. Following training on the balanced data, the model showed solid performance with a 89% overall accuracy. Class-wise, it obtained an F1-score of 0.98 for Request, 0.98 for Change, 0.81 for Problem, and 0.81 for Incident, which shows efficient classification in all support categories.


2.4. Challenges Encountered and Solutions Adopted:
Throughout development, various practical issues were faced. The initial regex patterns for Aadhaar and credit/debit card numbers were too inclusive, resulting in both being classified as Aadhaar numbers. This was addressed by refining the spacing and digit number patterns in the regex to distinguish Aadhaar (12 digits) from standard 16-digit card structures. Another problem surfaced with the NER model incorrectly labeling organization names or general salutations (e.g., "Dear Sir") as full names. To prevent this, contextual filtering and boundary checks were incorporated into the masking logic. Furthermore, releasing the app to Hugging Face was initially challenging due to Gradio compatibility and Python environment issues. This was addressed by migrating to a **Docker-based deployment**, providing increased control over the environment and the ability to have a clean `uvicorn`-based FastAPI backend setup.

3. Final Output:
Hugging Face Space:
API Endpoint:  https://sanabanu31-email-classifier.hf.space/classify

GitHub Repository:
Source Code:  https://github.com/sanadsugithub/email_classifier.git

Sample Input:
{
  "input_email_body": "Subject: Request for a Novel Data Visualization Tool I am inquiring about the installation of a new data visualization tool to boost our investment analytics. The tool should offer interactive and dynamic visualizations to assist investors in making educated choices.. My contact number is +49-30-3987-2105."
}

Sample Output:
{
  "input_email_body": "Subject: Request for a Novel Data Visualization Tool I am inquiring about the installation of a new data visualization tool to boost our investment analytics. The tool should offer interactive and dynamic visualizations to assist investors in making educated choices.. My contact number is +49-30-3987-2105.",
  "list_of_masked_entities": [
    {
      "placeholder": "[phone_number_000]",
      "original": "+49-30-3987-2105"
    }
  ],
  "masked_email": "Subject: Request for a Novel Data Visualization Tool I am inquiring about the installation of a new data visualization tool to boost our investment analytics. The tool should offer interactive and dynamic visualizations to assist investors in making educated choices.. My contact number is [phone_number_000].",
  "category_of_the_email": "Change"
}
