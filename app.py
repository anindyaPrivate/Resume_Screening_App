import streamlit as st
import pickle
import re
import nltk
from PyPDF2 import PdfReader

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load models
clf = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

# Function to clean resume content
def clean_data(text):
    # Remove URLs, mentions, hashtags, special characters, and additional whitespace
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Main app function
def main():
    # Add custom CSS for styling
    st.markdown("""
    <style>
    .main {
        background-color: #e0f7e9; /* Light green background */
        padding: 20px;
        border-radius: 8px;
    }
    .title {
        text-align: center;
        color: #0066cc; /* Blue title */
        font-size: 2.5em; /* Increased font size */
        font-weight: bold; /* Bold text */
        margin-bottom: 20px;
    }
    .header {
        color: #0066cc; /* Blue header */
        border-bottom: 2px solid #0066cc;
        padding-bottom: 10px;
        margin-bottom: 20px;
        font-size: 1.5em; /* Increased font size */
        font-weight: bold; /* Bold text */
    }
    .box {
        background-color: #ffffff;
        border: 1px solid #dcdcdc;
        border-radius: 8px;
        padding: 20px; /* Increased padding */
        margin-bottom: 20px; /* Increased margin */
        font-size: 1.2em; /* Increased font size */
    }
    .text-area {
        border-radius: 8px;
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        padding: 15px; /* Increased padding */
        font-size: 1.2em; /* Increased font size */
        color: #333333; /* Darker text color for better visibility */
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Resume Screening", "Resume Cleaner"])

    if page == "Resume Screening":
        st.markdown('<div class="title">Resume Screening App</div>', unsafe_allow_html=True)
        st.write("Upload your resume to predict the job category it belongs to.")

        st.markdown('<div class="header">Upload Your Resume</div>', unsafe_allow_html=True)
        upload_file = st.file_uploader("Choose a file in .txt or .pdf format", type=["txt", "pdf"])

        if upload_file is not None:
            file_extension = upload_file.name.split('.')[-1].lower()

            if file_extension == 'pdf':
                resume_text = extract_text_from_pdf(upload_file)
            else:
                resume_text = upload_file.read().decode("utf-8", errors="ignore")

            cleaned_resume = clean_data(resume_text)

            input_features = tfidf.transform([cleaned_resume])
            prediction_id = clf.predict(input_features)[0]

            category_mapping = {
                15: "Java Developer", 23: "Testing", 8: "DevOps Engineer",
                28: "Python Developer", 24: "Web Designing", 12: "HR",
                13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
                18: "Operations Manager", 6: "Data Science", 22: "Sales",
                16: "Mechanical Engineer", 1: "Arts", 7: "Database",
                11: "Electrical Engineering", 14: "Health and fitness",
                19: "PMO", 4: "Business Analyst", 9: "DotNet Developer",
                2: "Automation Testing", 17: "Network Security Engineer",
                21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate",
            }
            category_name = category_mapping.get(prediction_id, "Unknown")

            st.markdown('<div class="box"><h3>Prediction Result</h3></div>', unsafe_allow_html=True)
            st.success(f"The resume most likely belongs to the category: **{category_name}**")

    elif page == "Resume Cleaner":
        st.markdown('<div class="title">Resume Cleaner</div>', unsafe_allow_html=True)
        st.write("Upload your resume to view and clean the content.")

        st.markdown('<div class="header">Upload Your Resume</div>', unsafe_allow_html=True)
        upload_file = st.file_uploader("Choose a file in .txt or .pdf format", type=["txt", "pdf"])

        if upload_file is not None:
            file_extension = upload_file.name.split('.')[-1].lower()

            if file_extension == 'pdf':
                resume_text = extract_text_from_pdf(upload_file)
            else:
                resume_text = upload_file.read().decode("utf-8", errors="ignore")

            st.markdown('<div class="box"><h3>Original Resume Content</h3></div>', unsafe_allow_html=True)
            st.text_area("Original Resume", value=resume_text, height=300, key="original_resume_cleaner", class_="text-area")

            cleaned_resume = clean_data(resume_text)

            st.markdown('<div class="box"><h3>Cleaned Resume Content</h3></div>', unsafe_allow_html=True)
            st.text_area("Cleaned Resume", value=cleaned_resume, height=300, key="cleaned_resume_cleaner", class_="text-area")

if __name__ == '__main__':
    main()
