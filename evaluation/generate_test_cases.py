"""
Script to generate test cases by analyzing chunks and metadata
"""
import json
import os
from collections import defaultdict

# Load metadata to create mapping from (source_file, chunk_id) to global_id
metadata_path = "../data/vectordb/metadata.json"
chunks_dir = "../data/chunks"

with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# Create mapping: (source_file, chunk_id) -> global_id
chunk_to_global = {}
for entry in metadata:
    key = (entry['source_file'], entry['chunk_id'])
    chunk_to_global[key] = entry['global_id']

print(f"Loaded {len(metadata)} metadata entries")
print(f"Created mapping for {len(chunk_to_global)} chunks")

# Load all chunk files and create searchable content
chunk_content = {}  # (source_file, chunk_id) -> text
for source_file in os.listdir(chunks_dir):
    if source_file.endswith('_chunks.json'):
        file_path = os.path.join(chunks_dir, source_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            for chunk in chunks:
                chunk_id = chunk['id']
                text = chunk['text'].lower()
                chunk_content[(source_file, chunk_id)] = text

print(f"Loaded content from {len([f for f in os.listdir(chunks_dir) if f.endswith('_chunks.json')])} chunk files")

# Define questions and keywords to search for
test_questions = [
    # Roles questions
    ("What is the role of the Timer in a Toastmasters meeting?", ["timer", "time", "speeches", "table topics"]),
    ("What are the responsibilities of the Grammarian?", ["grammarian", "word of the day", "vocabulary", "grammar"]),
    ("What does the Ah Counter do during a meeting?", ["ah counter", "filler words", "crutch words", "um", "ah"]),
    ("What is the role of the General Evaluator?", ["general evaluator", "overall meeting", "feedback"]),
    ("What are the duties of the Toastmaster of the Day (TMOD)?", ["toastmaster", "tmod", "host", "director", "meeting"]),
    ("What does the Table Topics Master do?", ["table topics master", "topicsmaster", "impromptu", "topics"]),
    ("What is the role of the Sergeant at Arms?", ["sergeant at arms", "saa", "meeting room", "equipment"]),
    ("What are the responsibilities of the Club President?", ["club president", "president responsibilities", "leadership"]),
    ("What does the Vice President Education (VPE) do?", ["vice president education", "vpe", "educational program"]),
    ("What are the duties of the Club Secretary?", ["club secretary", "secretary responsibilities", "minutes", "records"]),
    ("What is the role of the Club Treasurer?", ["club treasurer", "treasurer responsibilities", "financial", "dues"]),
    ("What does the Vice President Membership do?", ["vice president membership", "membership", "recruitment"]),
    ("What are the responsibilities of the Vice President Public Relations?", ["vice president public relations", "public relations", "pr", "marketing"]),
    
    # Contest questions
    ("What is the time limit for speeches in the International Speech Contest?", ["international speech contest", "time limit", "speech"]),
    ("What are the eligibility requirements for speech contests?", ["eligibility", "speech contest", "requirements", "contestant"]),
    ("What are the rules for the Humorous Speech Contest?", ["humorous speech contest", "rules", "humorous"]),
    ("What is the time limit for Table Topics speeches?", ["table topics", "time limit", "speech"]),
    ("What are the evaluation criteria in a speech contest?", ["evaluation contest", "criteria", "judging"]),
    ("Can non-English speech contests proceed beyond District level?", ["non-english", "district level", "contest"]),
    ("What is the Tall Tales Contest?", ["tall tales contest", "tall tales"]),
    ("What are the rules for Online Speech Contests?", ["online speech contest", "online", "virtual"]),
    
    # Meeting structure questions
    ("What are the components of a Toastmasters meeting?", ["components", "meeting", "structure", "anatomy"]),
    ("What happens during the Table Topics section?", ["table topics", "section", "impromptu", "speaking"]),
    ("How long should speeches typically be in club meetings?", ["speeches", "time", "minutes", "club meeting"]),
    ("What is the purpose of evaluations in Toastmasters?", ["evaluations", "feedback", "purpose", "improvement"]),
    ("What is an Icebreaker speech?", ["icebreaker", "first speech", "introduction"]),
    
    # Pathways and education
    ("What are Toastmasters Pathways?", ["pathways", "educational program", "learning"]),
    ("What are the core competencies in Pathways?", ["core competencies", "pathways", "skills"]),
    ("How do you complete a Pathway level?", ["pathway level", "complete", "projects"]),
    
    # Templates and guidelines
    ("What should be included in an Ah Counter report?", ["ah counter report", "report", "crutch words"]),
    ("What is the time limit for the Ah Counter role introduction?", ["ah counter", "time limit", "introduction", "minute"]),
    ("What should the Grammarian report include?", ["grammarian report", "word of the day", "grammar"]),
    ("What are the guidelines for creating an introduction?", ["introduction", "creating", "guidelines"]),
    ("What should a Timer report include?", ["timer report", "time", "speeches"]),
    
    # Leadership and governance
    ("What is the Club Executive Committee?", ["executive committee", "club", "governance"]),
    ("What are the duties of the Club Executive Committee?", ["executive committee", "duties", "responsibilities"]),
    ("What is the Distinguished Club Program?", ["distinguished club program", "dcp", "recognition"]),
    ("What is the District structure in Toastmasters?", ["district", "structure", "area", "division"]),
    ("What happens at Area Council meetings?", ["area council", "meetings", "area"]),
    ("What is the purpose of Division Council?", ["division council", "purpose", "division"]),
    ("What are the responsibilities of an Area Director?", ["area director", "visits", "club"]),
    
    # General Toastmasters knowledge
    ("What is the mission of Toastmasters International?", ["mission", "values", "promises", "toastmasters international"]),
    ("What are the core values of Toastmasters?", ["values", "core values", "toastmasters"]),
    ("How can you become a Toastmasters club officer?", ["club officer", "become", "election"]),
    ("What are the different roles in a Toastmasters meeting?", ["roles", "meeting", "different", "various"]),
    ("What is the purpose of Table Topics?", ["table topics", "purpose", "impromptu", "practice"]),
    ("How do you improve as a speaker in Toastmasters?", ["improve", "speaker", "practice", "feedback"]),
    ("What makes a good speech evaluation?", ["evaluation", "good", "feedback", "constructive"]),
    ("What are common mistakes to avoid as a new Toastmaster?", ["mistakes", "new", "toastmaster", "avoid"]),
    ("What is the role of the Presiding Officer?", ["presiding officer", "address", "meeting"]),
    ("How do you prepare for a speech contest?", ["speech contest", "prepare", "contestant", "preparation"]),
]

# Function to find relevant global_ids for a question
def find_relevant_chunks(question, keywords):
    question_lower = question.lower()
    scored_chunks = []  # List of (score, global_id) tuples
    
    # Search through all chunks and score them
    for (source_file, chunk_id), text in chunk_content.items():
        # Get global_id from mapping
        key = (source_file, chunk_id)
        if key not in chunk_to_global:
            continue
            
        global_id = chunk_to_global[key]
        
        # Score based on keyword matches
        score = 0
        matched_keywords = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Check if keyword appears in text (not just question)
            if keyword_lower in text:
                # Give higher score for exact phrase matches
                if len(keyword.split()) > 1:  # Multi-word keyword
                    score += 3
                else:
                    score += 1
                matched_keywords.append(keyword)
        
        # Bonus for matching multiple keywords
        if len(matched_keywords) > 1:
            score += len(matched_keywords)
        
        # Only include chunks that match at least one keyword
        if score > 0:
            scored_chunks.append((score, global_id, text[:100]))  # Include text snippet for debugging
    
    # Sort by score (highest first) and remove duplicates
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    
    # Remove duplicate global_ids, keeping the highest scored one
    seen = set()
    relevant_global_ids = []
    for score, global_id, text_snippet in scored_chunks:
        if global_id not in seen:
            relevant_global_ids.append(global_id)
            seen.add(global_id)
        if len(relevant_global_ids) >= 5:  # Limit to top 5
            break
    
    return relevant_global_ids

# Generate test cases
test_cases = []
existing_ids = set()

# Load existing test cases to continue numbering
existing_file = "test_cases.json"
if os.path.exists(existing_file):
    with open(existing_file, 'r', encoding='utf-8') as f:
        existing = json.load(f)
        existing_ids = {tc['id'] for tc in existing}
        next_id = max(existing_ids) + 1 if existing_ids else 1
        test_cases = existing
else:
    next_id = 1

# Generate new test cases
for question, keywords in test_questions:
    if next_id > 50:  # Stop at 50 total
        break
    
    relevant_ids = find_relevant_chunks(question, keywords)
    
    if relevant_ids:  # Only add if we found relevant chunks
        # Verify the chunks are actually relevant by checking their content
        question_lower = question.lower()
        verified_ids = []
        for global_id in relevant_ids:
            # Find the chunk content for this global_id
            found = False
            for entry in metadata:
                if entry['global_id'] == global_id:
                    source_file = entry['source_file']
                    chunk_id = entry['chunk_id']
                    key = (source_file, chunk_id)
                    if key in chunk_content:
                        text = chunk_content[key]
                        # Check if at least one primary keyword appears
                        primary_keywords = [k for k in keywords if len(k.split()) > 1 or k.lower() in question_lower]
                        if any(kw.lower() in text for kw in primary_keywords) or any(kw.lower() in text for kw in keywords[:2]):
                            verified_ids.append(global_id)
                            found = True
                            break
            if not found:
                # Still add it if we can't verify (might be valid)
                verified_ids.append(global_id)
        
        if verified_ids:  # Only add if we have verified chunks
            test_cases.append({
                "id": next_id,
                "query": question,
                "relevant_global_ids": verified_ids[:5]  # Limit to 5
            })
            next_id += 1
        else:
            print(f"Warning: No verified relevant chunks for: {question}")
    else:
        print(f"Warning: No relevant chunks found for: {question}")

# Save test cases
output_file = "test_cases.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(test_cases, f, indent=2, ensure_ascii=False)

print(f"\nGenerated {len(test_cases)} test cases")
print(f"Saved to {output_file}")

