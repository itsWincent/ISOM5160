"""
ISOM5160 Homework 2 — SMS Spam / Smishing Alert App
Course: ISOM5160-L3 Python Programming for Business Applications, Fall 2026 (PG ICSM)

INSTRUCTIONS (read carefully)
- Complete the marked STUDENT sections and the function bodies below.
- Keep these function names: load_model, load_test_samples, classify_text,
  compute_accuracy, main.
- Call main() at the bottom of this file (the starter already does this).
- In GitHub and on Streamlit Cloud, this file must be named app.py.
- Submit to Canvas: a zip file containing app.py, requirements.txt, and
  Homework2_<StudentID>_link.txt (Streamlit Cloud URL and GitHub URL).
  Put those three files at the root of the zip, not inside a subfolder.
- Do NOT fine-tune a model. Do NOT scrape the web.
- Do NOT change DATASET_ID, MODEL_ID, N_TEST, or SPLIT_SEED.
- When you deploy, open Advanced settings and set Python to 3.12.
  If the app is already deployed, delete it and redeploy — the Python
  version cannot be changed in place.
- If the app calls a Hugging Face model, store a fine-grained Hugging Face
  API token in Streamlit Cloud secrets as HF_TOKEN. Do not put the token
  in app.py, requirements.txt, or any public file.
- First run may download the model and dataset (needs network). CPU is OK.

SECRET PHRASE (must appear on the app page): ISOM5160-HW2-SMS
"""

# =============================================================================
# STUDENT ANSWERS BLOCK (fill in; keep this in the SAME .py file)
# =============================================================================
# Student ID: 21403267
#
# GenAI acknowledgment (required if you used GenAI; otherwise write "None"):
#   - Tools used: Gemini
#   - How you used them: Checking for error and debugging
#
# Q1. Who is the primary business user of this app?
#     (one short sentence; e.g., an employee or a security analyst reviewing
#      inbound SMS for spam or scam-like messages)
#   YOUR_ANSWER: everyday civilian trying to see if the message they obtained from unknown number is a scam
#
# Q2. Look at the accuracy from your 100-sample test. Is accuracy enough
#     for a business to rely on this app, and why or why not?
#     2–4 short sentences max. You MUST cite the accuracy figure from
#     YOUR app run. Comment on the cost of a false positive (blocking a
#     legitimate SMS, such as a bank or one-time code message) versus a
#     false negative (missing a spam or scam-like message).
#   YOUR_ANSWER: no. because when measuring the performance of a fraud detection such as this one, having just accuracy can be an issue if the dataset is highly imbalanced. 
#                for example, if the dataset has 99 true and one false, having a model that just returns "true" everytime will also get a 99% accuracy. 
#                other metrics are needed such as recall
# =============================================================================

import streamlit as st
from datasets import load_dataset
from transformers import pipeline

# Locked by the instructor — do not change these four values.
DATASET_ID = "ucirvine/sms_spam"
MODEL_ID = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
N_TEST = 100
SPLIT_SEED = 42

# Course identity (required)
STUDENT_ID = "YOUR_STUDENT_ID"  # STUDENT: replace with your real student ID
COURSE_SECRET = "ISOM5160-HW2-SMS"  # do not change


def map_label(raw_label):
    """Map model or dataset labels to the two display names used in this app."""
    text = str(raw_label).strip().lower()
    if text in {"ham", "legitimate", "label_0", "0"}:
        return "legitimate"
    if text in {"spam", "smishing", "label_1", "1"}:
        return "spam"
    return str(raw_label)


@st.cache_resource
def load_model(token):
    """Load the SMS text-classification model once and reuse it."""
    # This function is provided. Keep device="cpu" and
    # model_kwargs={"low_cpu_mem_usage": False} so Streamlit Cloud can load
    # the model on CPU.
    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        token=token,
        device="cpu",
        truncation=True,
        max_length=512,
        model_kwargs={"low_cpu_mem_usage": False},
    )
    return classifier


@st.cache_data
def load_test_samples():
    """
    Load the Hugging Face dataset and take a fixed set of 100 samples.

    The Hub copy has only a train split, so this homework holds out 100
    rows after a shuffle with SPLIT_SEED. Do not change the seed or N_TEST,
    so every student is tested on the same 100 messages.
    """
    raw = load_dataset(DATASET_ID, split="train")
    picked = raw.shuffle(seed=SPLIT_SEED).select(range(N_TEST))
    texts = list(picked["sms"])
    true_labels = [map_label(label) for label in picked["label"]]
    return texts, true_labels


def classify_text(classifier, text):
    """Return (display_label, confidence) for one SMS string."""
    # STUDENT: call the pipeline on this one text, map the raw label with
    # map_label(...), and return the display label and the confidence score.

    temp = classifier(text)[0]
    display_label = map_label(temp["label"])  # STUDENT: replace this
    score = float(temp["score"])  # STUDENT: replace this
    return display_label, score


def compute_accuracy(pred_labels, true_labels):
    """Return accuracy as a float between 0 and 1."""
    # STUDENT: accuracy = (number of matches) / (number of samples).

    if not true_labels:
        return 0.0
    else:
        matches = sum(1 for pred, true in zip(pred_labels, true_labels) if pred == true)
    accuracy = matches/len(true_labels)  # STUDENT: replace this
    return accuracy


def main():
    st.title("ISOM5160")
    st.write("Homework 2 — SMS Spam / Smishing Alert")
    # STUDENT: also show Student ID and COURSE_SECRET on the page.
    st.write(f"Student ID: {STUDENT_ID} | Secret: {COURSE_SECRET}")

    # Use a Hugging Face fine-grained token from Streamlit Cloud secrets
    # if one is set. Do not put the token in this file.
    hf_token = None
    if "HF_TOKEN" in st.secrets:
        hf_token = st.secrets["HF_TOKEN"]

    # STUDENT: load the model with load_model(hf_token)
    classifier = load_model(hf_token)

    alert_text = st.text_area(
        "SMS message",
        value=(
            "WINNER!! You have been selected to receive a $900 prize. "
            "Claim now at http://prize-alert.example or the offer expires."
        ),
        height=140,
    )

    if st.button("Classify SMS"):
        # STUDENT: if the text box is not empty, call classify_text,
        # then show the display label and the confidence.
        if alert_text.strip():
            pred_label, confidence = classify_text(classifier, alert_text.strip())
            st.write(f"**Prediction:** {pred_label}")
            st.write(f"**Confidence:** {confidence:.4f}")
        else:
            st.warning("Please enter your SMS message and try again.")
        pass

    st.subheader("Test on 100 Hugging Face samples")
    st.write(
        "Click the button to load 100 messages from the Hugging Face "
        "dataset and measure accuracy."
    )
    if st.button("Run 100-sample test"):
        # STUDENT: call load_test_samples(), classify each message with
        # classify_text, then compute_accuracy. Show the accuracy and a
        # results table with at least these columns:
        # # | Text (short) | Ground truth | Prediction | Confidence | Match

        texts, true_labels = load_test_samples()
        pred_labels = []
        rows = []
        for idx, (msg, truth) in enumerate(zip(texts, true_labels), start=1):
            pred, conf = classify_text(classifier, msg)
            pred_labels.append(pred)
            short_text = msg if len(msg) <= 60 else msg[:57] + "..."
            rows.append(
                {
                    "#": idx,
                    "Text (short)": short_text,
                    "Ground truth": truth,
                    "Prediction": pred,
                    "Confidence": round(conf, 4),
                    "Match": pred == truth,
                }
            )
        acc = compute_accuracy(pred_labels, true_labels)
        st.write(f"**Accuracy:** {acc:.4f} ({acc * 100:.2f}%)")
        st.dataframe(rows, use_container_width=True)
        pass


if __name__ == "__main__":
    main()
