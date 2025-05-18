import time
import requests
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def explain_with_groq(algorithm_name, job_description, resume_text, score):
    # Truncate inputs to reduce token usage
    resume_text = resume_text[:3000]
    job_description = job_description[:1000]

    prompt = f"""
        You are an AI resume evaluator. Your task is to analyze the similarity between a job description and a resume using the provided algorithm. Be strict in your evaluation: only mark a resume as a "Good Fit" if it clearly matches the key responsibilities, skills, and qualifications of the job. If the match is partial or weak, label it as a "Bad Fit."
        Job Description:{job_description},Resume:{resume_text} , Similarity Score: {score:.4f}, Algorithm Used: {algorithm_name} . In exactly 4 lines, explain why this resume is a Good Fit or Bad Fit for the job. Do not be lenient. Do not suggest improvements. Avoid bullet points. Prefix your response with either 'Good Fit:' or 'Bad Fit:'. """

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a strict and objective resume evaluator. Only mark a resume as a 'Good Fit' if it strongly matches the job description in terms of key skills, responsibilities, and qualifications. If the match is weak or partial, label it as a 'Bad Fit'. Never be lenient. Keep evaluations concise, exactly 4 lines, and clearly explain the reasoning."
},
            {"role": "user", "content": prompt}
        ]
    }


    # Retry logic
    for attempt in range(3):  # Try up to 3 times
        try:
            response = requests.post(url, headers=headers, json=payload)
            result = response.json()

            if 'error' in result:
                error_msg = result['error']['message']
                if "rate limit" in error_msg.lower() or "please try again" in error_msg.lower():
                    wait_time = 8 + attempt * 2
                    time.sleep(wait_time)
                    continue  # retry
                elif "request too large" in error_msg.lower():
                    return f"⚠️ Resume too long. Skipped to avoid Groq token limit."
                else:
                    return f"⚠️ Groq API error: {error_msg}"

            # Pause to avoid hitting TPM limits
            time.sleep(1.5)

            return result['choices'][0]['message']['content']

        except Exception as e:
            time.sleep(2)
            continue  # try again

    return "❌ Failed after multiple Groq attempts."

def batch_groq_fit_evaluation(algorithm_name, job_description, resume_score_data):
    good_fits = []
    bad_fits = []

    for file_name, score, resume_text in resume_score_data:
        with st.spinner(f"Evaluating: {file_name}"):
            explanation = explain_with_groq(algorithm_name, job_description, resume_text, score)

        if explanation.strip().lower().startswith("good fit") or "**Good Fit**" in explanation:
            good_fits.append((file_name, score, explanation))
        else:
            bad_fits.append((file_name, score, explanation))

    return good_fits, bad_fits
