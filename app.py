import streamlit as st
import pickle
import re
import nltk
from PyPDF2 import PdfReader

nltk.download('punkt')
nltk.download("stopwords")

# Loading models
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
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Resume Screening", "Resume Cleaner"])

    if page == "Resume Screening":
        st.title("Resume Screening App")
        st.write("Upload your resume to predict the job category it belongs to.")

        st.header("Upload Your Resume")
        upload_file = st.file_uploader("Choose a file in .txt or .pdf format", type=["txt", "pdf"])

        if upload_file is not None:
            file_extension = upload_file.name.split('.')[-1].lower()

            if file_extension == 'pdf':
                resume_text = extract_text_from_pdf(upload_file)
            else:
                resume_text = upload_file.read().decode("utf-8", errors="ignore")

            # st.subheader("Original Resume Content")
            # st.write(resume_text)

            cleaned_resume = clean_data(resume_text)
            # st.subheader("Cleaned Resume Content")
            # st.write(cleaned_resume)

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

            st.subheader("Prediction Result")
            st.success(f"The resume most likely belongs to the category: **{category_name}**")

    elif page == "Resume Cleaner":
        st.title("Resume Cleaner")
        st.write("Upload your resume to view and clean the content.")

        st.header("Upload Your Resume")
        upload_file = st.file_uploader("Choose a file in .txt or .pdf format", type=["txt", "pdf"])

        if upload_file is not None:
            file_extension = upload_file.name.split('.')[-1].lower()

            if file_extension == 'pdf':
                resume_text = extract_text_from_pdf(upload_file)
            else:
                resume_text = upload_file.read().decode("utf-8", errors="ignore")

            st.subheader("Original Resume Content")
            st.text_area("Original Resume", value=resume_text, height=300)

            cleaned_resume = clean_data(resume_text)
            st.subheader("Cleaned Resume Content")
            st.text_area("Cleaned Resume", value=cleaned_resume, height=300)


if __name__ == '__main__':
    main()
