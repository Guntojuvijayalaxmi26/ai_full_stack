
import streamlit as st
import re

from pypdf import PdfReader
from docx import Document

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📄 AI Resume Analyzer")
st.write(
    "Upload your resume and paste a job description "
    "to analyze your resume."
)

st.divider()


# =========================================================
# SKILLS DATABASE
# =========================================================

SKILLS = [
    "python",
    "java",
    "c",
    "c++",
    "javascript",
    "html",
    "css",
    "sql",
    "mysql",
    "mongodb",
    "git",
    "github",
    "linux",
    "aws",
    "azure",
    "docker",
    "kubernetes",

    "artificial intelligence",
    "ai",
    "machine learning",
    "deep learning",
    "data science",
    "data analysis",
    "natural language processing",
    "nlp",
    "computer vision",

    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "keras",
    "transformers",
    "sentence transformers",

    "streamlit",
    "flask",
    "django",
    "fastapi",
    "react",
    "node.js",
    "nodejs",
    "rest api",
    "api",

    "data structures",
    "algorithms",
    "dsa",
    "oops",
    "object oriented programming",
    "problem solving",

    "excel",
    "power bi",
    "tableau",

    "communication",
    "leadership",
    "teamwork"
]


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(file):

    try:
        reader = PdfReader(file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    except Exception as e:

        return ""


# =========================================================
# DOCX TEXT EXTRACTION
# =========================================================

def extract_docx_text(file):

    try:

        document = Document(file)

        text = ""

        for paragraph in document.paragraphs:

            if paragraph.text.strip():

                text += paragraph.text + "\n"

        return text

    except Exception as e:

        return ""


# =========================================================
# RESUME TEXT EXTRACTION
# =========================================================

def extract_resume_text(file):

    filename = file.name.lower()

    if filename.endswith(".pdf"):

        return extract_pdf_text(file)

    elif filename.endswith(".docx"):

        return extract_docx_text(file)

    return ""


# =========================================================
# FIND SKILLS
# =========================================================

def find_skills(text):

    text = text.lower()

    found = []

    for skill in SKILLS:

        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, text):

            found.append(skill)

    return sorted(set(found))


# =========================================================
# JOB MATCHING
# =========================================================

def calculate_similarity(resume_text, job_text):

    try:

        documents = [
            resume_text.lower(),
            job_text.lower()
        ]

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        matrix = vectorizer.fit_transform(documents)

        similarity = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        return round(similarity * 100, 2)

    except Exception:

        return 0


# =========================================================
# ATS SCORE
# =========================================================

def calculate_ats_score(
    resume_text,
    job_text,
    resume_skills,
    job_skills
):

    similarity = calculate_similarity(
        resume_text,
        job_text
    )

    # Job similarity = 50%
    score = similarity * 0.50

    # Skills = 30%
    if len(job_skills) > 0:

        matching = set(resume_skills) & set(job_skills)

        skill_score = (
            len(matching) / len(job_skills)
        ) * 30

        score += skill_score

    # Resume sections = 10%
    resume_lower = resume_text.lower()

    sections = [
        "education",
        "skills",
        "experience",
        "project",
        "projects",
        "certification",
        "contact"
    ]

    section_count = 0

    for section in sections:

        if section in resume_lower:

            section_count += 1

    score += min(section_count * 1.5, 10)

    # Resume length = 10%
    word_count = len(resume_text.split())

    if word_count >= 400:

        score += 10

    elif word_count >= 250:

        score += 7

    elif word_count >= 150:

        score += 5

    elif word_count >= 75:

        score += 3

    return min(round(score), 100)


# =========================================================
# SUGGESTIONS
# =========================================================

def generate_suggestions(
    resume_text,
    resume_skills,
    job_skills
):

    suggestions = []

    resume_lower = resume_text.lower()

    # Missing skills

    missing = [
        skill
        for skill in job_skills
        if skill not in resume_skills
    ]

    if missing:

        suggestions.append(
            "Consider adding these relevant skills if you "
            "actually have them: "
            + ", ".join(missing[:10])
        )

    # Education

    if "education" not in resume_lower:

        suggestions.append(
            "Add a clear Education section."
        )

    # Projects

    if (
        "project" not in resume_lower
        and
        "projects" not in resume_lower
    ):

        suggestions.append(
            "Add relevant projects with technologies "
            "and your contribution."
        )

    # Experience

    if "experience" not in resume_lower:

        suggestions.append(
            "Add internship, training, or relevant "
            "experience if you have any."
        )

    # Certifications

    if "certification" not in resume_lower:

        suggestions.append(
            "Add relevant certifications if available."
        )

    # Action words

    action_words = [
        "developed",
        "created",
        "built",
        "designed",
        "implemented",
        "developed",
        "optimized",
        "analyzed"
    ]

    if not any(
        word in resume_lower
        for word in action_words
    ):

        suggestions.append(
            "Use strong action words such as Developed, "
            "Implemented, Designed, Built and Analyzed."
        )

    # Numbers

    if not re.search(
        r"\d+%",
        resume_text
    ):

        suggestions.append(
            "Where possible, add measurable results such "
            "as percentages, numbers, users, or performance improvements."
        )

    if not suggestions:

        suggestions.append(
            "Your resume looks well structured. "
            "Keep improving it with relevant skills "
            "and measurable achievements."
        )

    return suggestions


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📌 How to Use")

    st.write("1️⃣ Upload your resume")

    st.write("2️⃣ Paste the job description")

    st.write("3️⃣ Click Analyze Resume")

    st.write("4️⃣ Check your ATS score")

    st.write("5️⃣ Improve missing skills")


# =========================================================
# RESUME UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload Resume",
    type=["pdf", "docx"]
)


# =========================================================
# JOB DESCRIPTION
# =========================================================

job_description = st.text_area(
    "💼 Paste Job Description",
    height=250,
    placeholder=(
        "Example:\n\n"
        "We are looking for a Python developer with "
        "knowledge of SQL, machine learning, Git, "
        "data structures and problem solving."
    )
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button(
    "🔍 Analyze Resume",
    type="primary"
):

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    if uploaded_file is None:

        st.error(
            "❌ Please upload your resume."
        )

        st.stop()

    if not job_description.strip():

        st.error(
            "❌ Please paste a job description."
        )

        st.stop()


    # -----------------------------------------------------
    # EXTRACT TEXT
    # -----------------------------------------------------

    with st.spinner(
        "Reading your resume..."
    ):

        resume_text = extract_resume_text(
            uploaded_file
        )


    if not resume_text.strip():

        st.error(
            "❌ Could not extract text from your resume."
        )

        st.info(
            "Make sure your PDF contains selectable text. "
            "Scanned PDFs require OCR."
        )

        st.stop()


    # -----------------------------------------------------
    # FIND SKILLS
    # -----------------------------------------------------

    resume_skills = find_skills(
        resume_text
    )

    job_skills = find_skills(
        job_description
    )


    # -----------------------------------------------------
    # MATCHING SKILLS
    # -----------------------------------------------------

    matching_skills = sorted(
        set(resume_skills)
        &
        set(job_skills)
    )


    # -----------------------------------------------------
    # MISSING SKILLS
    # -----------------------------------------------------

    missing_skills = sorted(
        set(job_skills)
        -
        set(resume_skills)
    )


    # -----------------------------------------------------
    # JOB MATCH
    # -----------------------------------------------------

    similarity = calculate_similarity(
        resume_text,
        job_description
    )


    # -----------------------------------------------------
    # ATS SCORE
    # -----------------------------------------------------

    ats_score = calculate_ats_score(
        resume_text,
        job_description,
        resume_skills,
        job_skills
    )


    # -----------------------------------------------------
    # SUGGESTIONS
    # -----------------------------------------------------

    suggestions = generate_suggestions(
        resume_text,
        resume_skills,
        job_skills
    )


    # =====================================================
    # RESULTS
    # =====================================================

    st.success(
        "✅ Resume analyzed successfully!"
    )

    st.divider()


    # -----------------------------------------------------
    # SCORE CARDS
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "ATS Score",
            f"{ats_score}%"
        )

    with col2:

        st.metric(
            "Job Match",
            f"{similarity}%"
        )

    with col3:

        st.metric(
            "Skills Found",
            len(resume_skills)
        )

    with col4:

        st.metric(
            "Resume Words",
            len(resume_text.split())
        )


    st.divider()


    # -----------------------------------------------------
    # MATCHING SKILLS
    # -----------------------------------------------------

    st.subheader(
        "✅ Matching Skills"
    )

    if matching_skills:

        cols = st.columns(3)

        for index, skill in enumerate(
            matching_skills
        ):

            with cols[index % 3]:

                st.success(skill)

    else:

        st.info(
            "No matching skills detected."
        )


    # -----------------------------------------------------
    # MISSING SKILLS
    # -----------------------------------------------------

    st.subheader(
        "❌ Missing Skills"
    )

    if missing_skills:

        cols = st.columns(3)

        for index, skill in enumerate(
            missing_skills
        ):

            with cols[index % 3]:

                st.error(skill)

    else:

        st.success(
            "No major missing skills detected."
        )


    # -----------------------------------------------------
    # ALL RESUME SKILLS
    # -----------------------------------------------------

    st.subheader(
        "🛠️ Skills Found in Resume"
    )

    if resume_skills:

        st.write(
            ", ".join(resume_skills)
        )

    else:

        st.info(
            "No known skills detected."
        )


    # -----------------------------------------------------
    # JOB SKILLS
    # -----------------------------------------------------

    st.subheader(
        "💼 Skills Found in Job Description"
    )

    if job_skills:

        st.write(
            ", ".join(job_skills)
        )

    else:

        st.info(
            "No known skills detected."
        )


    # -----------------------------------------------------
    # SUGGESTIONS
    # -----------------------------------------------------

    st.subheader(
        "💡 Resume Improvement Suggestions"
    )

    for suggestion in suggestions:

        st.write(
            "🔹 " + suggestion
        )


    # -----------------------------------------------------
    # SCORE INTERPRETATION
    # -----------------------------------------------------

    st.subheader(
        "📊 ATS Score Interpretation"
    )

    if ats_score >= 80:

        st.success(
            "Excellent! Your resume is strongly matched "
            "with this job description."
        )

    elif ats_score >= 60:

        st.warning(
            "Good! Your resume matches the job, but "
            "there is room for improvement."
        )

    elif ats_score >= 40:

        st.warning(
            "Moderate match. Consider adding relevant "
            "skills and improving your resume content."
        )

    else:

        st.error(
            "Low match. Customize your resume according "
            "to the job description."
        )


    # -----------------------------------------------------
    # EXTRACTED RESUME TEXT
    # -----------------------------------------------------

    with st.expander(
        "📄 View Extracted Resume Text"
    ):

        st.text(
            resume_text
        )
