# 📄 Resume Screening App
Welcome to the Resume Screening App! This Streamlit application helps you analyze and clean resumes while predicting job categories based on the cleaned content.

## 🚀 Features
Resume Upload: Upload your resume in .txt or .pdf format.

Resume Screening: Predict the job category of the resume using a machine learning model.

Resume Cleaner: View and clean the resume content for better processing.
## 🛠️ Installation
To get started, clone this repository and install the required packages:
~~~bash
git clone https://github.com/anindyaPrivate/Resume_Screening_App.git
cd resume-screening-app
~~~
## 📋 Usage
1. Start the Streamlit App
Run the following command to start the app:
~~~bash
streamlit run app.py
~~~
## 2. Navigate Through Pages

Resume Screening: Upload your resume and get a job category prediction.

Resume Cleaner: Upload your resume, view the original and cleaned content, and manually clean it if needed.

## 3. Upload Your Resume

Choose a .txt or .pdf file.

View and clean the content on the “Resume Cleaner” page.

Get predictions on the “Resume Screening” page.

## 🎨 Customization
You can customize the appearance of the app by modifying the CSS in the app.py file. For more details, check out the app.py code and adjust the styles according to your preference.

## 🔧 Requirements
Python 3.x
Streamlit
scikit-learn
PyPDF2
nltk
pickle
Install these dependencies using:
```bash
pip install streamlit scikit-learn PyPDF2 nltk
```
## 📦 Files

app.py: The main application file.

model.pkl: The pre-trained machine learning model.

vectorizer.pkl: The TF-IDF vectorizer used for feature extraction.

requirements.txt: List of Python packages required.
## 📝 Notes

Ensure the model and vectorizer files are correctly placed in the project directory.


The app supports .txt and .pdf file formats for resume uploads.

For any issues or enhancements, please open an issue or submit a pull request!

## 📬 Contact
For any questions or feedback, feel free to reach out:

Email: my_official2023@outlook.com
GitHub: anindyaPrivate

Happy resume screening! 🎉
