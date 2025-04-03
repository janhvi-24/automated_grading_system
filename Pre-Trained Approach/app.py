from typing import Dict, List, Optional, Union
import streamlit as st
import spacy
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pdf2image
import google.generativeai as genai
import os
import re
import io
import tempfile
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import BertTokenizer, BertModel
from PIL import Image
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
import base64
import time
import random

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="📚 AI Grading Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for styling with enhanced themes
def set_background(image_file):
    """
    Set background image using base64 encoding
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
            background-color: rgba(255, 255, 255, 0.9);
            background-blend-mode: lighten;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


# Set background (choose from different themes)
THEMES = {
    "Academic": "assets/academic_theme.jpg",
    "Modern": "assets/modern_theme.jpg",
    "Dark": "assets/dark_theme.jpg",
    "Light": None
}

# Custom CSS for styling
st.markdown(f"""
<style>
    :root {{
        --primary-color: #3498db;
        --secondary-color: #2c3e50;
        --accent-color: #e74c3c;
        --success-color: #2ecc71;
        --warning-color: #f39c12;
        --danger-color: #e74c3c;
        --light-color: #ecf0f1;
        --dark-color: #2c3e50;
    }}

    .stApp {{
        background-color: rgba(255, 255, 255, 0.95);
    }}

    .sidebar .sidebar-content {{
        background-color: var(--secondary-color);
        color: white;
        background-image: linear-gradient(to bottom, #2c3e50, #34495e);
    }}

    .metric-card {{
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 5px solid var(--primary-color);
        transition: transform 0.3s ease;
        height: 100%;
    }}

    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.15);
    }}

    .grade-A {{
        color: var(--success-color);
        font-weight: bold;
        font-size: 1.8rem;
        text-shadow: 0 2px 4px rgba(46, 204, 113, 0.3);
    }}

    .grade-B {{
        color: #27ae60;
        font-weight: bold;
        font-size: 1.8rem;
    }}

    .grade-C {{
        color: var(--warning-color);
        font-weight: bold;
        font-size: 1.8rem;
    }}

    .grade-D {{
        color: #e67e22;
        font-weight: bold;
        font-size: 1.8rem;
    }}

    .grade-F {{
        color: var(--danger-color);
        font-weight: bold;
        font-size: 1.8rem;
        text-shadow: 0 2px 4px rgba(231, 76, 60, 0.3);
    }}

    .feedback-card {{
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
        border-top: 3px solid var(--primary-color);
    }}

    .resource-card {{
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
        transition: all 0.3s ease;
    }}

    .resource-card:hover {{
        transform: translateX(5px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }}

    .header {{
        color: var(--secondary-color);
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
    }}

    .subheader {{
        color: var(--primary-color);
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 5px;
    }}

    .model-badge {{
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        margin-right: 8px;
        margin-bottom: 8px;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }}

    .sbert-badge {{
        background-color: var(--primary-color);
        color: white;
    }}

    .bert-badge {{
        background-color: #9b59b6;
        color: white;
    }}

    .tfidf-badge {{
        background-color: var(--danger-color);
        color: white;
    }}

    .concept-badge {{
        background-color: var(--success-color);
        color: white;
    }}

    .gemini-badge {{
        background-color: var(--warning-color);
        color: white;
    }}

    .answer-type-badge {{
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        background-color: var(--secondary-color);
        color: white;
        margin-left: 10px;
        font-weight: bold;
    }}

    .progress-bar {{
        height: 10px;
        border-radius: 5px;
        background-color: #ecf0f1;
        margin-top: 10px;
        overflow: hidden;
    }}

    .progress-fill {{
        height: 100%;
        background: linear-gradient(to right, var(--primary-color), #2980b9);
        border-radius: 5px;
    }}

    .floating-animation {{
        animation: floating 3s ease-in-out infinite;
    }}

    @keyframes floating {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-10px); }}
        100% {{ transform: translateY(0px); }}
    }}

    .pulse-animation {{
        animation: pulse 2s infinite;
    }}

    @keyframes pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.4); }}
        70% {{ box-shadow: 0 0 0 10px rgba(46, 204, 113, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(46, 204, 113, 0); }}
    }}

    .glow-text {{
        animation: glow 2s ease-in-out infinite alternate;
    }}

    @keyframes glow {{
        from {{ text-shadow: 0 0 5px #fff, 0 0 10px #fff, 0 0 15px var(--primary-color), 0 0 20px var(--primary-color); }}
        to {{ text-shadow: 0 0 10px #fff, 0 0 20px #fff, 0 0 30px var(--primary-color), 0 0 40px var(--primary-color); }}
    }}

    .tabs {{
        display: flex;
        margin-bottom: 20px;
        border-bottom: 1px solid #ddd;
    }}

    .tab {{
        padding: 10px 20px;
        cursor: pointer;
        background-color: #f1f1f1;
        margin-right: 5px;
        border-radius: 5px 5px 0 0;
    }}

    .tab.active {{
        background-color: var(--primary-color);
        color: white;
    }}

    .tooltip {{
        position: relative;
        display: inline-block;
    }}

    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 200px;
        background-color: var(--secondary-color);
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }}

    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}

    .flip-card {{
        background-color: transparent;
        perspective: 1000px;
        margin-bottom: 20px;
        min-height: 200px;
    }}

    .flip-card-inner {{
        position: relative;
        width: 100%;
        height: 100%;
        text-align: center;
        transition: transform 0.6s;
        transform-style: preserve-3d;
    }}

    .flip-card:hover .flip-card-inner {{
        transform: rotateY(180deg);
    }}

    .flip-card-front, .flip-card-back {{
        position: absolute;
        width: 100%;
        height: 100%;
        backface-visibility: hidden;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }}

    .flip-card-front {{
        background-color: white;
        color: var(--secondary-color);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }}

    .flip-card-back {{
        background-color: var(--primary-color);
        color: white;
        transform: rotateY(180deg);
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}

    .feature-icon {{
        font-size: 2.5rem;
        margin-bottom: 15px;
        color: var(--primary-color);
    }}

    .feature-title {{
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 10px;
    }}

    .feature-desc {{
        font-size: 0.9rem;
    }}

    .timeline {{
        position: relative;
        max-width: 1200px;
        margin: 0 auto;
    }}

    .timeline::after {{
        content: '';
        position: absolute;
        width: 6px;
        background-color: var(--primary-color);
        top: 0;
        bottom: 0;
        left: 50%;
        margin-left: -3px;
    }}

    .timeline-container {{
        padding: 10px 40px;
        position: relative;
        background-color: inherit;
        width: 50%;
    }}

    .timeline-container::after {{
        content: '';
        position: absolute;
        width: 25px;
        height: 25px;
        right: -17px;
        background-color: white;
        border: 4px solid var(--primary-color);
        top: 15px;
        border-radius: 50%;
        z-index: 1;
    }}

    .left {{
        left: 0;
    }}

    .right {{
        left: 50%;
    }}

    .left::before {{
        content: " ";
        height: 0;
        position: absolute;
        top: 22px;
        width: 0;
        z-index: 1;
        right: 30px;
        border: medium solid var(--primary-color);
        border-width: 10px 0 10px 10px;
        border-color: transparent transparent transparent white;
    }}

    .right::before {{
        content: " ";
        height: 0;
        position: absolute;
        top: 22px;
        width: 0;
        z-index: 1;
        left: 30px;
        border: medium solid var(--primary-color);
        border-width: 10px 10px 10px 0;
        border-color: transparent white transparent transparent;
    }}

    .timeline-content {{
        padding: 20px 30px;
        background-color: white;
        position: relative;
        border-radius: 6px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }}

    .timeline-date {{
        color: var(--warning-color);
        font-weight: bold;
    }}

    .timeline-title {{
        color: var(--secondary-color);
        font-weight: bold;
        margin-bottom: 10px;
    }}

    .confetti {{
        position: fixed;
        width: 10px;
        height: 10px;
        background-color: #f00;
        border-radius: 50%;
        animation: confetti-fall 5s linear forwards;
    }}

    @keyframes confetti-fall {{
        0% {{ transform: translateY(-100vh) rotate(0deg); opacity: 1; }}
        100% {{ transform: translateY(100vh) rotate(360deg); opacity: 0; }}
    }}

    .animated-button {{
        position: relative;
        overflow: hidden;
        transition: all 0.3s;
    }}

    .animated-button::after {{
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 5px;
        height: 5px;
        background: rgba(255, 255, 255, 0.5);
        opacity: 0;
        border-radius: 100%;
        transform: scale(1, 1) translate(-50%);
        transform-origin: 50% 50%;
    }}

    .animated-button:focus:not(:active)::after {{
        animation: ripple 1s ease-out;
    }}

    @keyframes ripple {{
        0% {{
            transform: scale(0, 0);
            opacity: 0.5;
        }}
        20% {{
            transform: scale(25, 25);
            opacity: 0.3;
        }}
        100% {{
            opacity: 0;
            transform: scale(40, 40);
        }}
    }}

    .expandable-section {{
        margin-bottom: 20px;
    }}

    .expandable-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 15px;
        background-color: var(--primary-color);
        color: white;
        border-radius: 5px;
        cursor: pointer;
    }}

    .expandable-content {{
        padding: 15px;
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 0 0 5px 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }}

    /* New styles for consistent feature boxes */
    .feature-box {{
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 5px solid var(--primary-color);
        height: 100%;
        display: flex;
        flex-direction: column;
    }}

    .feature-box-icon {{
        font-size: 2.5rem;
        margin-bottom: 15px;
        color: var(--primary-color);
        text-align: center;
    }}

    .feature-box-title {{
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
    }}

    .feature-box-desc {{
        font-size: 0.9rem;
        text-align: center;
        flex-grow: 1;
    }}

    /* Diagram container */
    .diagram-container {{
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }}
</style>
""", unsafe_allow_html=True)


class NLPGradingModels:
    def __init__(self):
        # Load all models at initialization
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = spacy.load('en_core_web_md')
        self.vectorizer = TfidfVectorizer()
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        # Configure Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("Missing GOOGLE_API_KEY in environment variables")
            st.stop()

        genai.configure(api_key=api_key)
        self.gemini_vision = genai.GenerativeModel('gemini-1.5-flash-latest')
        self.gemini_text = genai.GenerativeModel('gemini-1.5-pro-latest')

    def get_bert_embedding(self, text):
        """Computes BERT embeddings for a given text."""
        inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def extract_key_concepts(self, text):
        """Extracts key concepts like nouns, verbs, and named entities."""
        doc = self.nlp(text)
        keywords = {token.lemma_.lower() for token in doc if token.pos_ in ['NOUN', 'VERB', 'PROPN']}
        entities = {ent.text.lower() for ent in doc.ents}
        return keywords.union(entities)

    def contradiction_penalty(self, model_keywords, student_keywords):
        """Applies a stricter penalty for conflicting key concepts (for short answers)."""
        incorrect_keywords = student_keywords - model_keywords
        penalty = min(1.2, len(incorrect_keywords) / max(len(model_keywords), 1))
        return penalty

    def key_concept_score(self, model_answer, student_answer, answer_type):
        """
        Calculates key concept score with different approaches based on answer type.
        For short answers: Strict penalties for incorrect concepts
        For paragraphs: More lenient with minimum baseline
        """
        model_keywords = self.extract_key_concepts(model_answer)
        student_keywords = self.extract_key_concepts(student_answer)
        intersection = model_keywords.intersection(student_keywords)
        concept_match = len(intersection) / max(len(model_keywords), 1)

        if answer_type == "short":
            penalty = self.contradiction_penalty(model_keywords, student_keywords)
            final_concept_score = max(0.1, concept_match - penalty)
            if concept_match < 0.3:
                final_concept_score = min(final_concept_score, 0.2)
            return final_concept_score
        else:
            return max(0.5, concept_match)  # Paragraph answers get more lenient scoring

    def tfidf_similarity(self, model_answer, student_answer):
        """Computes statistical similarity using TF-IDF."""
        corpus = [model_answer, student_answer]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    def compute_model_based_score(self, student_answer, model_answer, answer_type):
        """
        Combines SBERT, BERT, TF-IDF, and key concept evaluation with weights
        based on answer type.
        """
        # Get embeddings
        sbert_emb1 = self.sbert_model.encode(student_answer)
        sbert_emb2 = self.sbert_model.encode(model_answer)
        bert_emb1 = self.get_bert_embedding(student_answer)
        bert_emb2 = self.get_bert_embedding(model_answer)

        # Calculate similarities
        semantic_similarity = cosine_similarity([sbert_emb1], [sbert_emb2])[0][0]
        deep_context_similarity = cosine_similarity([bert_emb1], [bert_emb2])[0][0]
        tfidf_score = self.tfidf_similarity(model_answer, student_answer)
        concept_score = self.key_concept_score(model_answer, student_answer, answer_type)

        # Different weighting based on answer type
        if answer_type == "short":
            weights = {
                'semantic': 0.2,
                'context': 0.2,
                'tfidf': 0.2,
                'concept': 0.4
            }
        else:
            weights = {
                'semantic': 0.3,
                'context': 0.3,
                'tfidf': 0.2,
                'concept': 0.2
            }

        # Calculate weighted score
        final_score = (
                (semantic_similarity * weights['semantic']) +
                (deep_context_similarity * weights['context']) +
                (tfidf_score * weights['tfidf']) +
                (concept_score * weights['concept'])
        )

        # Additional safeguards for short answers
        if answer_type == "short":
            if concept_score < 0.3:
                final_score = min(final_score, 0.4)
            final_score = max(0, final_score)

        return {
            'score': final_score * 10,  # Scale to 10-point system
            'components': {
                'sbert': semantic_similarity * 10,
                'bert': deep_context_similarity * 10,
                'tfidf': tfidf_score * 10,
                'concepts': concept_score * 10
            }
        }

    def convert_pdf_to_images(self, pdf_file):
        """Convert PDF to list of PIL images"""
        try:
            return pdf2image.convert_from_bytes(
                pdf_file.read(),
                dpi=300,
                fmt='png',
                thread_count=4
            )
        except Exception as e:
            st.error(
                "PDF processing requires poppler-utils. Install via: Ubuntu/Debian: sudo apt-get install poppler-utils, MacOS: brew install poppler, Windows: Install Poppler and add to PATH from https://github.com/oschwartz10612/poppler-windows/releases")
            raise e

    def extract_text_with_gemini(self, image):
        """Extract text from image using Gemini Vision"""
        try:
            response = self.gemini_vision.generate_content([
                "Extract all text from this image exactly as written. Preserve formatting, equations, and special characters.",
                image
            ])
            return response.text if response else ""
        except Exception as e:
            st.error(f"Text extraction failed: {e}")
            return ""

    def process_file(self, uploaded_file):
        """Process PDF or image file using Gemini"""
        content = uploaded_file.read()
        results = []

        if uploaded_file.type == 'application/pdf':
            images = self.convert_pdf_to_images(io.BytesIO(content))
            for idx, img in enumerate(images, 1):
                text = self.extract_text_with_gemini(img)
                results.append({'page': idx, 'text': text, 'image': img})
        else:  # Image file
            img = Image.open(io.BytesIO(content))
            text = self.extract_text_with_gemini(img)
            results.append({'page': 1, 'text': text, 'image': img})

        return results

    def get_gemini_feedback(self, student_text, model_text, answer_type, max_marks):
        """Get comprehensive feedback from Gemini API"""
        prompt = f"""Analyze and grade student's answer against model answer with extreme detail:

        Model Answer: {model_text}
        Student Answer: {student_text}
        Answer Type: {'Short answer' if answer_type == 'short' else 'Paragraph answer'}
        Maximum Marks: {max_marks}

        Provide a comprehensive analysis with these components:
        1. Numerical marks awarded (0 to {max_marks}) - calculate based on content accuracy, structure, and key concepts
        2. Key strengths of the student's answer
        3. Key weaknesses and areas needing improvement
        4. Detailed feedback for improvement
        5. Letter grade (A-F) based on academic standards
        6. Specific actionable recommendations
        7. Study resources relevant to the question topic

        Format your response as follows:

        Marks Awarded: [marks] out of {max_marks}
        Grade: [letter grade]

        Strengths:
        - [strength 1]
        - [strength 2]

        Weaknesses:
        - [weakness 1]
        - [weakness 2]

        Feedback:
        [detailed paragraph feedback]

        Recommendations:
        - [recommendation 1]
        - [recommendation 2]

        Study Resources:
        - [resource 1]
        - [resource 2]"""

        try:
            response = self.gemini_text.generate_content(prompt)
            return self._parse_gemini_response(response.text, max_marks)
        except Exception as e:
            st.error(f"Gemini feedback generation failed: {e}")
            return {}

    def _parse_gemini_response(self, text: str, max_marks: int) -> dict:
        """Parse Gemini's response into structured data"""
        try:
            result = {
                'marks': 0,
                'max_marks': max_marks,
                'grade': 'F',
                'strengths': [],
                'weaknesses': [],
                'feedback': '',
                'recommendations': [],
                'resources': []
            }

            # Extract marks awarded
            marks_match = re.search(r'Marks Awarded:\s*(\d+(?:\.\d+)?)\s*out of', text)
            if marks_match:
                result['marks'] = float(marks_match.group(1))

            # Extract grade
            grade_match = re.search(r'Grade:\s*([A-F][+-]?)', text, re.IGNORECASE)
            if grade_match:
                result['grade'] = grade_match.group(1).upper()

            # Extract strengths
            strengths_section = re.search(r'Strengths:(.*?)(?=Weaknesses:|$)', text, re.DOTALL | re.IGNORECASE)
            if strengths_section:
                strengths = re.findall(r'-\s*(.*)', strengths_section.group(1))
                result['strengths'] = [s.strip() for s in strengths if s.strip()]

            # Extract weaknesses
            weaknesses_section = re.search(r'Weaknesses:(.*?)(?=Feedback:|$)', text, re.DOTALL | re.IGNORECASE)
            if weaknesses_section:
                weaknesses = re.findall(r'-\s*(.*)', weaknesses_section.group(1))
                result['weaknesses'] = [w.strip() for w in weaknesses if w.strip()]

            # Extract feedback
            feedback_section = re.search(r'Feedback:(.*?)(?=Recommendations:|$)', text, re.DOTALL | re.IGNORECASE)
            if feedback_section:
                result['feedback'] = feedback_section.group(1).strip()

            # Extract recommendations
            recommendations_section = re.search(r'Recommendations:(.*?)(?=Study Resources:|$)', text,
                                                re.DOTALL | re.IGNORECASE)
            if recommendations_section:
                recommendations = re.findall(r'-\s*(.*)', recommendations_section.group(1))
                result['recommendations'] = [r.strip() for r in recommendations if r.strip()]

            # Extract study resources
            resources_section = re.search(r'Study Resources:(.*)', text, re.DOTALL | re.IGNORECASE)
            if resources_section:
                resources = re.findall(r'-\s*(.*)', resources_section.group(1))
                result['resources'] = [r.strip() for r in resources if r.strip()]

            return result
        except Exception as e:
            st.error(f"Response parsing failed: {e}")
            return {'marks': 0, 'max_marks': max_marks, 'grade': 'F', 'feedback': text}

    def generate_learning_plan(self, question, model_answer, weaknesses):
        """Generate personalized learning plan using Gemini"""
        prompt = f"""Create a detailed, actionable 2-week learning plan based on:

        Question: {question}
        Model Answer: {model_answer}
        Student Weaknesses: {', '.join(weaknesses)}

        Provide:
        1. A structured 2-week study schedule with daily topics
        2. Recommended resources for each weakness (include links if possible)
        3. Practice exercises tailored to the weaknesses
        4. Self-assessment checkpoints with clear criteria
        5. Motivational tips for the student

        Format as:

        **Learning Plan Overview**
        [concise overview of the plan]

        **Week 1 Schedule**
        - Day 1: [topic] - [specific resources]
        - Day 2: [topic] - [specific resources]
        ...

        **Week 2 Schedule**
        - Day 1: [topic] - [specific resources]
        - Day 2: [topic] - [specific resources]
        ...

        **Practice Exercises**
        - [exercise 1 with instructions]
        - [exercise 2 with instructions]
        ...

        **Assessment Checkpoints**
        - [checkpoint 1 with success criteria]
        - [checkpoint 2 with success criteria]
        ...

        **Additional Tips**
        [motivational tips and study advice]"""

        try:
            response = self.gemini_text.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Learning plan generation failed: {e}")
            return "Could not generate learning plan at this time."

    def generate_quiz_questions(self, topic, difficulty="medium", num_questions=5):
        """Generate quiz questions on a given topic using Gemini"""
        prompt = f"""Generate {num_questions} {difficulty} difficulty quiz questions on the topic: {topic}

        For each question provide:
        1. The question text
        2. 4 multiple choice options (A-D)
        3. The correct answer
        4. A brief explanation of why it's correct
        5. The key concept being tested

        Format as:

        **Question 1**
        [question text]

        A) [option A]
        B) [option B]
        C) [option C]
        D) [option D]

        **Correct Answer:** [letter]
        **Explanation:** [brief explanation]
        **Key Concept:** [concept being tested]

        **Question 2**
        ..."""

        try:
            response = self.gemini_text.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Quiz generation failed: {e}")
            return "Could not generate quiz questions at this time."

    def analyze_writing_style(self, text):
        """Analyze writing style using Gemini"""
        prompt = f"""Analyze the following text for writing style characteristics:

        {text}

        Provide analysis of:
        1. Vocabulary level (basic, intermediate, advanced)
        2. Sentence structure (simple, compound, complex)
        3. Tone (formal, informal, academic, conversational)
        4. Readability score estimate
        5. Strengths in writing style
        6. Areas for improvement
        7. Specific suggestions for enhancing clarity and impact

        Format as:

        **Vocabulary:** [level] - [specific examples]
        **Sentence Structure:** [analysis with examples]
        **Tone:** [description with examples]
        **Readability:** [score/description]

        **Strengths:**
        - [strength 1 with example]
        - [strength 2 with example]

        **Areas for Improvement:**
        - [area 1 with specific example]
        - [area 2 with specific example]

        **Recommendations:**
        - [specific recommendation 1]
        - [specific recommendation 2]"""

        try:
            response = self.gemini_text.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Writing analysis failed: {e}")
            return "Could not analyze writing style at this time."


class AIGradingApp:
    def __init__(self):
        self.models = NLPGradingModels()
        self.results_df = pd.DataFrame(columns=[
            'Student Name', 'Question', 'Answer Type', 'Grade',
            'Marks', 'Max Marks', 'Percentage', 'Model Score',
            'Feedback', 'Timestamp', 'Subject', 'Assignment'
        ])
        if 'results_df' not in st.session_state:
            st.session_state.results_df = self.results_df
        if 'theme' not in st.session_state:
            st.session_state.theme = "Academic"
        if 'confetti' not in st.session_state:
            st.session_state.confetti = False
        if 'last_grading_data' not in st.session_state:
            st.session_state.last_grading_data = None

    def _create_confetti(self):
        """Creates confetti animation effect"""
        if st.session_state.confetti:
            js = """
            <script>
            function createConfetti() {
                const colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff'];
                const container = document.createElement('div');
                container.style.position = 'fixed';
                container.style.top = '0';
                container.style.left = '0';
                container.style.width = '100%';
                container.style.height = '100%';
                container.style.pointerEvents = 'none';
                container.style.zIndex = '1000';
                document.body.appendChild(container);

                for (let i = 0; i < 100; i++) {
                    const confetti = document.createElement('div');
                    confetti.style.position = 'absolute';
                    confetti.style.width = '10px';
                    confetti.style.height = '10px';
                    confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
                    confetti.style.borderRadius = '50%';

                    const startX = Math.random() * window.innerWidth;
                    const animationDuration = Math.random() * 3 + 2;

                    confetti.style.left = startX + 'px';
                    confetti.style.top = '-10px';

                    const animation = confetti.animate([
                        { top: '-10px', opacity: 1, transform: 'rotate(0deg)' },
                        { top: window.innerHeight + 'px', opacity: 0, transform: 'rotate(360deg)' }
                    ], {
                        duration: animationDuration * 1000,
                        easing: 'cubic-bezier(0.1, 0.8, 0.3, 1)'
                    });

                    animation.onfinish = () => confetti.remove();

                    container.appendChild(confetti);
                }

                setTimeout(() => container.remove(), 3000);
            }

            setTimeout(createConfetti, 100);
            </script>
            """
            st.components.v1.html(js, height=0)
            st.session_state.confetti = False

    def home_page(self):
        self._create_confetti()

        # Hero Section with full-width container
        with st.container():
            cols = st.columns([1, 2])
            with cols[0]:
                st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png",
                         width=200, use_column_width=False)
            with cols[1]:
                st.markdown("<h1 class='glow-text'>🎓 AI Grading Assistant</h1>",
                            unsafe_allow_html=True)
                st.markdown("""
                <h4 style='color:var(--secondary-color);'>Advanced Multi-Model Evaluation System</h4>
                <p style='font-size:1.1rem;'>Revolutionize your grading process with AI-powered assessment tools.</p>
                """, unsafe_allow_html=True)

                if st.button("🚀 Get Started Grading", use_container_width=True, type="primary"):
                    st.session_state.page = "Grade Submission"
                    st.rerun()

        st.markdown("---")

        # Features Section with improved layout and consistent boxes
        st.markdown("<h2 class='header'>✨ Key Features</h2>", unsafe_allow_html=True)

        features = [
            {"icon": "🤖", "title": "Multi-Model Evaluation",
             "desc": "Combines SBERT, BERT, TF-IDF and Key Concept analysis"},
            {"icon": "🔍", "title": "Answer-Type Aware", "desc": "Different grading for short vs paragraph answers"},
            {"icon": "📊", "title": "AI Consensus", "desc": "Cross-validates with Gemini API for reliable scoring"},
            {"icon": "📚", "title": "Learning Plans", "desc": "Generates personalized study recommendations"},
            {"icon": "✍️", "title": "Writing Analysis", "desc": "Evaluates vocabulary and sentence structure"},
            {"icon": "🧠", "title": "Quiz Generator", "desc": "Creates custom quizzes for any topic"}
        ]

        # Using container width for better responsiveness with consistent boxes
        with st.container():
            cols = st.columns(3)
            for idx, feature in enumerate(features):
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div class='feature-box'>
                        <div class='feature-box-icon'>{feature['icon']}</div>
                        <div class='feature-box-title'>{feature['title']}</div>
                        <div class='feature-box-desc'>{feature['desc']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # How It Works Section with diagram
        st.markdown("<h2 class='header'>📝 How It Works</h2>", unsafe_allow_html=True)

        with st.expander("See our grading process workflow", expanded=False):
            st.markdown("""
            <div class='diagram-container'>
                <h3 style='text-align:center;'>AI Grading Process</h3>
                <div style='text-align:center;'>
                    <p>1. Upload student and model answers</p>
                    <p>↓</p>
                    <p>2. AI analyzes with multiple NLP models</p>
                    <p>↓</p>
                    <p>3. Gemini provides comprehensive feedback</p>
                    <p>↓</p>
                    <p>4. Results combined for final assessment</p>
                    <p>↓</p>
                    <p>5. Detailed feedback and learning plan generated</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Recent Grades Section with improved layout
        if not st.session_state.results_df.empty:
            st.markdown("<h2 class='header'>📈 Recent Grades Overview</h2>", unsafe_allow_html=True)
            recent_grades = st.session_state.results_df.tail(5).sort_values('Timestamp', ascending=False)

            for _, row in recent_grades.iterrows():
                with st.container():
                    cols = st.columns([1, 1, 1, 2])
                    with cols[0]:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h4>Student</h4>
                            <h3>{row['Student Name']}</h3>
                            <p>{row['Subject'] if pd.notna(row['Subject']) else 'No subject'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[1]:
                        grade_class = f"grade-{row['Grade'][0]}" if pd.notna(row['Grade']) else ""
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h4>Grade</h4>
                            <h3 class='{grade_class}'>{row['Grade'] if pd.notna(row['Grade']) else 'N/A'}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[2]:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h4>Marks</h4>
                            <h3>{row['Marks']:.1f}/{row['Max Marks']}</h3>
                            <p>{row['Percentage']:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[3]:
                        percentage = row['Percentage']
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=percentage,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={'axis': {'range': [None, 100]},
                                   'bar': {'color': "#3498db"}}))
                        fig.update_layout(height=150, margin=dict(t=0, b=0, l=0, r=0))
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🌟 No submissions graded yet. Get started by grading some submissions!")

    def grade_submission_page(self):
        st.title("📝 Grade Submission")

        # Check if we have last grading data and display it
        if st.session_state.last_grading_data:
            with st.expander("↩️ Last Graded Submission", expanded=False):
                last_data = st.session_state.last_grading_data
                st.markdown(f"**Student:** {last_data['Student Name']}")
                st.markdown(f"**Question:** {last_data['Question'][:100]}...")
                st.markdown(f"**Grade:** {last_data['Grade']}")
                st.markdown(f"**Marks:** {last_data['Marks']:.1f}/{last_data['Max Marks']}")
                st.markdown(f"**Percentage:** {last_data['Percentage']:.1f}%")
                if st.button("Clear Last Submission", key="clear_last"):
                    st.session_state.last_grading_data = None
                    st.rerun()

        with st.expander("ℹ️ How to use this page", expanded=False):
            st.markdown("""
            - Upload student answers and model answers (PDF, image, or text)
            - Provide the question text or upload a PDF with questions
            - Enter the maximum marks for this question
            - Select the answer type (short or paragraph)
            - Click 'Grade Submission' to get AI-powered feedback
            """)

        with st.form("grading_form"):
            # File upload section
            cols = st.columns(2)
            with cols[0]:
                student_file = st.file_uploader("📤 Student Answer",
                                                type=['pdf', 'png', 'jpg', 'jpeg', 'txt'],
                                                help="Upload student's answer file")
            with cols[1]:
                model_file = st.file_uploader("📚 Model Answer",
                                              type=['pdf', 'png', 'jpg', 'jpeg', 'txt'],
                                              help="Upload model/correct answer file")

            # Student information
            cols = st.columns(2)
            with cols[0]:
                student_name = st.text_input("👤 Student Name",
                                             placeholder="Enter student's name")
            with cols[1]:
                subject = st.text_input("📚 Subject",
                                        placeholder="e.g., Mathematics, History")

            assignment_name = st.text_input("📝 Assignment Name",
                                            placeholder="e.g., Chapter 3 Quiz")

            # Question input with PDF upload option
            question_tab1, question_tab2 = st.tabs(["📝 Enter Question Text", "📄 Upload Question PDF"])
            question_text = ""

            with question_tab1:
                question_text = st.text_area("Question Text",
                                             placeholder="Enter the question",
                                             height=100)

            with question_tab2:
                question_file = st.file_uploader("Upload Question PDF",
                                                 type=['pdf'],
                                                 help="Upload a PDF containing the questions")
                if question_file:
                    with st.spinner("Extracting text from question PDF..."):
                        extracted_text = self._extract_text_from_file(question_file)
                        question_text = st.text_area("Extracted Question Text",
                                                     value=extracted_text,
                                                     height=100)

            # Marks input
            max_marks = st.number_input("🔢 Maximum Marks", min_value=1, max_value=100, value=10)

            answer_type = st.radio("🔤 Answer Type",
                                   ["Short Answer (single line/few words)", "Paragraph Answer"],
                                   index=1)

            submitted = st.form_submit_button("✨ Grade Submission",
                                              type="primary",
                                              use_container_width=True)

        if submitted:
            if all([student_file, model_file, student_name]) and (question_text or question_file):
                with st.spinner("🔍 Analyzing submissions..."):
                    try:
                        # Process files
                        student_text = self._extract_text_from_file(student_file)
                        model_text = self._extract_text_from_file(model_file)

                        if not student_text or not model_text:
                            st.error("Could not extract text from files")
                            return

                        answer_type_key = "short" if "Short Answer" in answer_type else "paragraph"

                        # Model-based evaluation
                        model_results = self.models.compute_model_based_score(
                            student_text, model_text, answer_type_key
                        )

                        # Gemini evaluation
                        gemini_results = self.models.get_gemini_feedback(
                            student_text, model_text, answer_type_key, max_marks
                        )

                        # Calculate percentage
                        percentage = (gemini_results['marks'] / max_marks) * 100
                        final_grade = gemini_results.get('grade', 'F')

                        # Display results
                        st.success("✅ Grading complete!")
                        st.balloons()
                        st.session_state.confetti = True
                        st.markdown("---")

                        # Results header
                        st.markdown(f"<h2 style='color:var(--secondary-color);'>📊 Results for {student_name}</h2>",
                                    unsafe_allow_html=True)
                        st.markdown(f"<span class='answer-type-badge'>{answer_type_key.capitalize()} Answer</span>",
                                    unsafe_allow_html=True)

                        # Metrics row
                        cols = st.columns(4)
                        with cols[0]:
                            grade_class = f"grade-{final_grade[0]}"
                            st.markdown(f"""
                            <div class='metric-card'>
                                <h3>Final Grade</h3>
                                <h1 class='{grade_class}'>{final_grade}</h1>
                                <p>Overall assessment</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with cols[1]:
                            st.markdown(f"""
                            <div class='metric-card'>
                                <h3>Marks Awarded</h3>
                                <h1>{gemini_results['marks']:.1f}/{max_marks}</h1>
                                <div class='progress-bar'>
                                    <div class='progress-fill' style='width:{percentage}%'></div>
                                </div>
                                <p>{percentage:.1f}% of total marks</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with cols[2]:
                            st.markdown(f"""
                            <div class='metric-card'>
                                <h3>Model Score</h3>
                                <h1>{model_results['score']:.1f}/10</h1>
                                <div class='progress-bar'>
                                    <div class='progress-fill' style='width:{model_results['score'] * 10}%'></div>
                                </div>
                                <p>NLP model evaluation</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with cols[3]:
                            st.markdown(f"""
                            <div class='metric-card'>
                                <h3>Percentage</h3>
                                <h1>{percentage:.1f}%</h1>
                                <div class='progress-bar'>
                                    <div class='progress-fill' style='width:{percentage}%'></div>
                                </div>
                                <p>Performance percentage</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Model components visualization
                        st.markdown("---")
                        st.markdown("<h3 class='subheader'>🤖 Model Evaluation Components</h3>", unsafe_allow_html=True)

                        components = model_results['components']
                        fig = px.bar(
                            x=list(components.keys()),
                            y=list(components.values()),
                            labels={'x': 'Model', 'y': 'Score (out of 10)'},
                            color=list(components.keys()),
                            color_discrete_map={
                                'sbert': '#3498db',
                                'bert': '#9b59b6',
                                'tfidf': '#e74c3c',
                                'concepts': '#2ecc71'
                            }
                        )
                        fig.update_layout(
                            showlegend=False,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Display badges for models used
                        st.markdown("""
                        <div style='margin-bottom: 20px;'>
                            <span class='model-badge sbert-badge'>SBERT</span>
                            <span class='model-badge bert-badge'>BERT</span>
                            <span class='model-badge tfidf-badge'>TF-IDF</span>
                            <span class='model-badge concept-badge'>Key Concepts</span>
                            <span class='model-badge gemini-badge'>Gemini</span>
                        </div>
                        """, unsafe_allow_html=True)

                        # Strengths and Weaknesses
                        st.markdown("---")
                        cols = st.columns(2)
                        with cols[0]:
                            st.markdown("""
                            <div class='feedback-card'>
                                <h3 style='color:#27ae60;'>✅ Strengths</h3>
                                <ul>
                            """, unsafe_allow_html=True)
                            for strength in gemini_results.get('strengths', []):
                                st.markdown(f"<li>{strength}</li>", unsafe_allow_html=True)
                            st.markdown("""
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        with cols[1]:
                            st.markdown("""
                            <div class='feedback-card'>
                                <h3 style='color:#e74c3c;'>❌ Areas for Improvement</h3>
                                <ul>
                            """, unsafe_allow_html=True)
                            for weakness in gemini_results.get('weaknesses', []):
                                st.markdown(f"<li>{weakness}</li>", unsafe_allow_html=True)
                            st.markdown("""
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)

                        # Detailed Feedback
                        st.markdown("---")
                        st.markdown("""
                        <div class='feedback-card'>
                            <h3 style='color:var(--primary-color);'>📝 Detailed Feedback</h3>
                            <p>{}</p>
                        </div>
                        """.format(gemini_results.get('feedback', 'No feedback available')), unsafe_allow_html=True)

                        # Writing Style Analysis for paragraph answers
                        if answer_type_key == "paragraph":
                            st.markdown("---")
                            with st.expander("✍️ Writing Style Analysis", expanded=False):
                                with st.spinner("Analyzing writing style..."):
                                    writing_analysis = self.models.analyze_writing_style(student_text)
                                    st.markdown(writing_analysis)

                        # Recommendations
                        if gemini_results.get('recommendations'):
                            st.markdown("---")
                            st.markdown("""
                            <div class='feedback-card'>
                                <h3 style='color:#f39c12;'>💡 Recommended Actions</h3>
                                <ul>
                            """, unsafe_allow_html=True)
                            for rec in gemini_results.get('recommendations', []):
                                st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
                            st.markdown("""
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)

                        # Study Resources
                        if gemini_results.get('resources'):
                            st.markdown("---")
                            st.markdown("""
                            <div class='feedback-card'>
                                <h3 style='color:#2c3e50;'>📚 Study Resources</h3>
                                <ul>
                            """, unsafe_allow_html=True)
                            for resource in gemini_results.get('resources', []):
                                st.markdown(f"<li>{resource}</li>", unsafe_allow_html=True)
                            st.markdown("""
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)

                        # Generate learning plan for low scores
                        if percentage < 70 and gemini_results.get('weaknesses'):
                            st.markdown("---")
                            with st.expander("🚀 Personalized Learning Plan", expanded=True):
                                with st.spinner("Generating learning plan..."):
                                    learning_plan = self.models.generate_learning_plan(
                                        question_text,
                                        model_text,
                                        gemini_results.get('weaknesses', [])
                                    )
                                    st.markdown(learning_plan)

                        # Store results
                        new_result = pd.DataFrame({
                            'Student Name': [student_name],
                            'Question': [question_text],
                            'Answer Type': [answer_type_key],
                            'Grade': [final_grade],
                            'Marks': [gemini_results['marks']],
                            'Max Marks': [max_marks],
                            'Percentage': [percentage],
                            'Model Score': [model_results['score']],
                            'Feedback': [gemini_results.get('feedback', '')],
                            'Timestamp': [pd.Timestamp.now()],
                            'Subject': [subject],
                            'Assignment': [assignment_name]
                        })
                        st.session_state.results_df = pd.concat(
                            [st.session_state.results_df, new_result],
                            ignore_index=True
                        )

                        # Store last grading data
                        st.session_state.last_grading_data = {
                            'Student Name': student_name,
                            'Question': question_text,
                            'Answer Type': answer_type_key,
                            'Grade': final_grade,
                            'Marks': gemini_results['marks'],
                            'Max Marks': max_marks,
                            'Percentage': percentage,
                            'Model Score': model_results['score'],
                            'Feedback': gemini_results.get('feedback', ''),
                            'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                            'Subject': subject,
                            'Assignment': assignment_name
                        }

                        # Show download button
                        st.markdown("---")
                        csv = st.session_state.results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download All Results",
                            data=csv,
                            file_name='grading_results.csv',
                            mime='text/csv',
                            use_container_width=True
                        )

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please provide all required information")

    def student_performance_page(self):
        st.title("📊 Student Performance Dashboard")

        if st.session_state.results_df.empty:
            st.info("ℹ️ No submission data available. Grade some submissions first!")
            return

        st.markdown("""
        <div style='background-color:rgba(232, 244, 252, 0.7); border-radius:10px; padding:15px; margin-bottom:20px;'>
            <p>Visual analytics of student performance across all graded submissions.</p>
        </div>
        """, unsafe_allow_html=True)

        # Ensure we have the required columns
        required_columns = ['Student Name', 'Question', 'Answer Type', 'Grade',
                            'Marks', 'Max Marks', 'Percentage', 'Model Score',
                            'Feedback', 'Timestamp', 'Subject', 'Assignment']

        # Add missing columns if they don't exist
        for col in required_columns:
            if col not in st.session_state.results_df.columns:
                st.session_state.results_df[col] = None

        # Convert Timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(st.session_state.results_df['Timestamp']):
            st.session_state.results_df['Timestamp'] = pd.to_datetime(st.session_state.results_df['Timestamp'])

        # Filters
        st.sidebar.header("📊 Dashboard Filters")

        # Get min and max dates from data
        min_date = st.session_state.results_df['Timestamp'].min().date()
        max_date = st.session_state.results_df['Timestamp'].max().date()

        date_range = st.sidebar.date_input(
            "Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )

        # Other filters
        min_percentage = st.sidebar.slider("Minimum Percentage", 0, 100, 0)
        answer_type_filter = st.sidebar.selectbox(
            "Answer Type",
            ["All"] + list(st.session_state.results_df['Answer Type'].dropna().unique())
        )
        subject_filter = st.sidebar.selectbox(
            "Subject",
            ["All"] + list(st.session_state.results_df['Subject'].dropna().unique())
        )
        student_filter = st.sidebar.selectbox(
            "Student",
            ["All"] + list(st.session_state.results_df['Student Name'].unique())
        )

        # Apply filters
        filtered_df = st.session_state.results_df.copy()

        # Date filter
        if len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['Timestamp'].dt.date >= date_range[0]) &
                (filtered_df['Timestamp'].dt.date <= date_range[1])
                ]

        # Other filters
        filtered_df = filtered_df[filtered_df['Percentage'] >= min_percentage]

        if answer_type_filter != "All":
            filtered_df = filtered_df[filtered_df['Answer Type'] == answer_type_filter]
        if subject_filter != "All":
            filtered_df = filtered_df[filtered_df['Subject'] == subject_filter]
        if student_filter != "All":
            filtered_df = filtered_df[filtered_df['Student Name'] == student_filter]

        # Check if any data remains after filtering
        if filtered_df.empty:
            st.warning("No data matches the selected filters. Please adjust your filter criteria.")
            return

        # Overall Metrics
        st.subheader("📈 Overall Performance Metrics")
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Submissions", len(filtered_df))
        with cols[1]:
            avg_percentage = filtered_df['Percentage'].mean()
            st.metric("Average Percentage", f"{avg_percentage:.1f}%")
        with cols[2]:
            avg_marks = filtered_df['Marks'].mean()
            avg_max_marks = filtered_df['Max Marks'].mean()
            st.metric("Average Marks", f"{avg_marks:.1f}/{avg_max_marks:.1f}")
        with cols[3]:
            pass_rate = len(filtered_df[filtered_df['Percentage'] >= 50]) / len(filtered_df) * 100
            st.metric("Pass Rate (≥50%)", f"{pass_rate:.1f}%")

        # Grade Distribution
        st.markdown("---")
        st.subheader("🎯 Grade Distribution")
        cols = st.columns([2, 1])
        with cols[0]:
            grade_counts = filtered_df['Grade'].value_counts().sort_index()
            fig_grades = px.pie(
                values=grade_counts.values,
                names=grade_counts.index,
                title="Grade Distribution",
                color=grade_counts.index,
                color_discrete_map={
                    'A': '#2ecc71', 'B': '#27ae60', 'C': '#f39c12',
                    'D': '#e67e22', 'F': '#e74c3c'
                }
            )
            fig_grades.update_traces(textposition='inside', textinfo='percent+label')
            fig_grades.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_grades, use_container_width=True)
        with cols[1]:
            st.dataframe(grade_counts.reset_index().rename(columns={'index': 'Grade', 'Grade': 'Count'}))

        # Performance Over Time
        if len(filtered_df) > 1:
            st.markdown("---")
            st.subheader("⏳ Performance Over Time")

            # Group by date if multiple submissions per day
            time_df = filtered_df.copy()
            time_df['Date'] = time_df['Timestamp'].dt.date
            time_df = time_df.sort_values('Timestamp')

            # Calculate daily averages
            daily_avg = time_df.groupby('Date').agg({
                'Percentage': 'mean',
                'Marks': 'mean',
                'Max Marks': 'first'
            }).reset_index()

            # Create figure
            fig_time = px.line(
                daily_avg,
                x='Date',
                y='Percentage',
                labels={'Percentage': 'Average Percentage (%)'},
                title="Performance Trend Over Time"
            )
            fig_time.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Date",
                yaxis_title="Average Percentage (%)"
            )
            st.plotly_chart(fig_time, use_container_width=True)

        # Subject Performance (if multiple subjects)
        if 'Subject' in filtered_df.columns and len(filtered_df['Subject'].dropna().unique()) > 1:
            st.markdown("---")
            st.subheader("📚 Subject Performance Comparison")

            subject_avg = filtered_df.groupby('Subject').agg({
                'Percentage': 'mean',
                'Marks': 'mean',
                'Max Marks': 'first'
            }).reset_index()

            fig_subject = px.bar(
                subject_avg,
                x='Subject',
                y='Percentage',
                color='Subject',
                text='Percentage',
                labels={'Percentage': 'Average Percentage (%)'},
                title="Average Performance by Subject"
            )
            fig_subject.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_subject.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                yaxis_title="Average Percentage (%)"
            )
            st.plotly_chart(fig_subject, use_container_width=True)

        # Student-wise Performance (if multiple students)
        if len(filtered_df['Student Name'].unique()) > 1:
            st.markdown("---")
            st.subheader("👥 Student-wise Performance")

            student_avg = filtered_df.groupby('Student Name').agg({
                'Percentage': 'mean',
                'Marks': 'mean',
                'Max Marks': 'first',
                'Grade': lambda x: x.value_counts().index[0]  # Most common grade
            }).reset_index()

            fig_student = px.bar(
                student_avg.sort_values('Percentage', ascending=False),
                x='Student Name',
                y='Percentage',
                color='Grade',
                color_discrete_map={
                    'A': '#2ecc71', 'B': '#27ae60', 'C': '#f39c12',
                    'D': '#e67e22', 'F': '#e74c3c'
                },
                text='Percentage',
                labels={'Percentage': 'Average Percentage (%)'},
                title="Student Performance Comparison"
            )
            fig_student.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_student.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Student",
                yaxis_title="Average Percentage (%)"
            )
            st.plotly_chart(fig_student, use_container_width=True)

        # Answer Type Comparison (if both types exist)
        if len(filtered_df['Answer Type'].unique()) > 1:
            st.markdown("---")
            st.subheader("🔤 Answer Type Comparison")

            type_avg = filtered_df.groupby('Answer Type').agg({
                'Percentage': 'mean',
                'Marks': 'mean',
                'Max Marks': 'first'
            }).reset_index()

            fig_type = px.bar(
                type_avg,
                x='Answer Type',
                y='Percentage',
                color='Answer Type',
                text='Percentage',
                labels={'Percentage': 'Average Percentage (%)'},
                title="Performance by Answer Type"
            )
            fig_type.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_type.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                yaxis_title="Average Percentage (%)"
            )
            st.plotly_chart(fig_type, use_container_width=True)

        # Detailed Data Table
        st.markdown("---")
        st.subheader("📋 Detailed Performance Data")
        st.dataframe(
            filtered_df.sort_values('Timestamp', ascending=False).drop(columns=['Feedback']),
            height=400
        )

        # Export options
        st.markdown("---")
        st.subheader("📤 Export Data")
        cols = st.columns(2)
        with cols[0]:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Results",
                data=csv,
                file_name='filtered_grading_results.csv',
                mime='text/csv',
                use_container_width=True
            )
        with cols[1]:
            full_csv = st.session_state.results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Results",
                data=full_csv,
                file_name='all_grading_results.csv',
                mime='text/csv',
                use_container_width=True
            )

    def quiz_generator_page(self):
        st.title("🧠 Quiz Generator")
        st.markdown("""
        <div style='background-color:rgba(232, 244, 252, 0.7); border-radius:10px; padding:15px; margin-bottom:20px;'>
            <p>Generate custom quizzes for any topic to test student understanding.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("quiz_form"):
            topic = st.text_input("📝 Topic", placeholder="Enter the topic for the quiz",
                                  help="e.g., Quadratic Equations, World War II, Photosynthesis")

            cols = st.columns(2)
            with cols[0]:
                difficulty = st.selectbox("📊 Difficulty", ["Easy", "Medium", "Hard"], index=1)
            with cols[1]:
                num_questions = st.slider("❓ Number of Questions", 1, 20, 5)

            submitted = st.form_submit_button("✨ Generate Quiz", type="primary", use_container_width=True)

        if submitted and topic:
            with st.spinner("Generating quiz questions..."):
                quiz = self.models.generate_quiz_questions(topic, difficulty.lower(), num_questions)
                st.markdown("---")
                st.markdown("### Generated Quiz")
                st.markdown(f"**Topic:** {topic}  \n**Difficulty:** {difficulty}  \n**Questions:** {num_questions}")
                st.markdown("---")
                st.markdown(quiz)

                # Add download button
                st.download_button(
                    label="📥 Download Quiz",
                    data=quiz,
                    file_name=f'quiz_{topic.replace(" ", "_")}.txt',
                    mime='text/plain',
                    use_container_width=True
                )

    def writing_analysis_page(self):
        st.title("✍️ Writing Style Analysis")
        st.markdown("""
        <div style='background-color:rgba(232, 244, 252, 0.7); border-radius:10px; padding:15px; margin-bottom:20px;'>
            <p>Analyze writing samples for vocabulary, sentence structure, and readability.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("writing_form"):
            writing_sample = st.text_area("📝 Writing Sample", height=300,
                                          placeholder="Paste the writing sample to analyze...")
            submitted = st.form_submit_button("🔍 Analyze Writing", type="primary", use_container_width=True)

        if submitted and writing_sample:
            with st.spinner("Analyzing writing style..."):
                analysis = self.models.analyze_writing_style(writing_sample)
                st.markdown("---")
                st.markdown("### Writing Analysis Results")
                st.markdown(analysis)

    def settings_page(self):
        st.title("⚙️ Settings")
        st.markdown("""
        <div style='background-color:rgba(232, 244, 252, 0.7); border-radius:10px; padding:15px; margin-bottom:20px;'>
            <p>Customize your grading assistant experience.</p>
        </div>
        """, unsafe_allow_html=True)

        # Theme selection
        st.subheader("🎨 Theme Customization")
        theme = st.selectbox("Select Theme", list(THEMES.keys()),
                             index=list(THEMES.keys()).index(st.session_state.theme))
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            if THEMES[theme]:
                set_background(THEMES[theme])
            else:
                st.markdown("""
                <style>
                .stApp {{
                    background-image: none;
                    background-color: #f5f5f5;
                }}
                </style>
                """, unsafe_allow_html=True)
            st.rerun()

        # Data management
        st.subheader("🗄️ Data Management")
        if st.button("Clear All Grading Data", use_container_width=True):
            st.session_state.results_df = pd.DataFrame(columns=[
                'Student Name', 'Question', 'Answer Type', 'Grade',
                'Marks', 'Max Marks', 'Percentage', 'Model Score',
                'Feedback', 'Timestamp', 'Subject', 'Assignment'
            ])
            st.success("All grading data has been cleared!")

        # Export all data
        st.download_button(
            label="📥 Export All Data",
            data=st.session_state.results_df.to_csv(index=False),
            file_name='ai_grading_assistant_data.csv',
            mime='text/csv',
            use_container_width=True
        )

        # Import data
        uploaded_file = st.file_uploader("📤 Import Data", type=['csv'])
        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)
                st.session_state.results_df = pd.concat([st.session_state.results_df, new_data], ignore_index=True)
                st.success("Data imported successfully!")
            except Exception as e:
                st.error(f"Error importing data: {e}")

    def run(self):
        """Main application runner"""
        # Apply theme
        if st.session_state.theme and THEMES[st.session_state.theme]:
            set_background(THEMES[st.session_state.theme])

        with st.sidebar:
            st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=80)
            st.title("AI Grading Assistant")

            selected = option_menu(
                menu_title=None,
                options=["Home", "Grade Submission", "Student Performance", "Quiz Generator", "Writing Analysis",
                         "Settings"],
                icons=["house", "file-earmark-text", "bar-chart", "question-square", "pencil-square", "gear"],
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "#2c3e50"},
                    "icon": {"color": "white", "font-size": "18px"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px",
                                 "--hover-color": "#34495e"},
                    "nav-link-selected": {"background-color": "#3498db"},
                }
            )

        if selected == "Home":
            self.home_page()
        elif selected == "Grade Submission":
            self.grade_submission_page()
        elif selected == "Student Performance":
            self.student_performance_page()
        elif selected == "Quiz Generator":
            self.quiz_generator_page()
        elif selected == "Writing Analysis":
            self.writing_analysis_page()
        elif selected == "Settings":
            self.settings_page()

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #7f8c8d; font-size: 0.9rem;'>
            <p>AI Grading Assistant v2.0 | © 2025 Educational Technology Inc.</p>
        </div>
        """, unsafe_allow_html=True)

    def _extract_text_from_file(self, file):
        """Extract text from various file types"""
        if file.type == 'text/plain':
            return file.read().decode('utf-8')
        elif file.type == 'application/pdf':
            images = self.models.convert_pdf_to_images(file)
            return "\n".join([self.models.extract_text_with_gemini(img) for img in images])
        elif file.type.startswith('image/'):
            img = Image.open(file)
            return self.models.extract_text_with_gemini(img)
        return ""


def main():
    # Initialize app
    app = AIGradingApp()

    # Run the app
    app.run()


if __name__ == "__main__":
    main()