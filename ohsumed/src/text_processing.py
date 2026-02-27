import re
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer

# --- Constants ---
RAW_NOISY_TAGS = {
    "Human", "Humans", "Male", "Female", "Animal", "Animals", "Adult", "Aged", 
    "Middle Age", "Adolescence", "Child", "Infant", "Pregnancy", "Young Adult", 
    "Aged, 80 and over", "Child, Preschool", "Infant, Newborn", "Mice", "Rats", 
    "Rats, Inbred Strains", "Dogs", "Rabbits", "Cattle", "Age Factors", 
    "Sex Factors", "Body Weight", "Case Report", "Comparative Study", 
    "English Abstract", "In Vitro", "Evaluation Studies", "Historical Article", 
    "Follow-Up Studies", "Clinical Trials", "Prospective Studies", 
    "Retrospective Studies", "Random Allocation", "Methods", "Double-Blind Method", 
    "Diagnosis, Differential", "Prognosis", "Risk", "Kinetics", "Time Factors", 
    "United States", "Cell Line", "Cells, Cultured", "Combined Modality Therapy", 
    "Recurrence", "Exertion", "Hemodynamics", "Support, Non-U.S. Gov't", 
    "Support, U.S. Gov't, P.H.S", "Support, U.S. Gov't, Non-P.H.S", 
    "Molecular Weight", "Acute Disease", "Chronic Disease", 
    "Monitoring, Physiologic", "Dose-Response Relationship, Drug"
}

MEDICAL_WEEDS = {
    'patient', 'patients', 'study', 'studies', 'clinical', 'result', 'results',
    'group', 'groups', 'treatment', 'treated', 'case', 'cases', 'associated',
    'found', 'showed', 'compared', 'using', 'used', 'however', 'also', 'within',
    'significantly', 'significant', 'effect', 'effects', 'increase', 'increased',
    'one', 'two', 'three', 'may', 'both', 'among', 'data','incidence', 'outcome', 
    'without', 'reported', 'report', 'demonstrated', 'shown', 'level', 'levels', 
    'total', 'year', 'years', 'analysis', 'method'
}

# --- Initialization ---
lemmatizer = WordNetLemmatizer()
tokenizer = WordPunctTokenizer()
stop_words = set(stopwords.words('english')).union(MEDICAL_WEEDS)

def get_fingerprint(text):
    if not text: return ""
    clean = re.sub(r'[^a-zA-Z0-9]', ' ', text).lower()
    return " ".join(clean.split())

NOISY_FINGERPRINTS = {get_fingerprint(tag) for tag in RAW_NOISY_TAGS}

def clean_mesh_smart(mesh_str):
    if not mesh_str: return []
    raw_terms =[t.split('/')[0].replace('*', '').strip() for t in mesh_str.split(';')]
    final_list =[]
    for term in raw_terms:
        f_print = get_fingerprint(term.rstrip('.'))
        if f_print not in NOISY_FINGERPRINTS and len(term) > 2:
            final_list.append(term.rstrip('.'))
    return final_list

def feature_cleaner(text):
    if not text: return ""
    text = text.lower()
    text = re.sub(r'[-/]', ' ', text)
    text = re.sub(r'\d+', ' [num] ', text) # Number handling
    text = re.sub(r'[^a-z\s\[\]]', ' ', text) # Punctuation
    
    tokens = tokenizer.tokenize(text)
    cleaned_tokens =[
        lemmatizer.lemmatize(w) for w in tokens 
        if w not in stop_words and len(w) >= 2
    ]
    return " ".join(cleaned_tokens)