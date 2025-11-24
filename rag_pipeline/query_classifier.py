# Keeping it simple for now. Classify based on the phrase
# To do: Pass it through LLM to decide for better accuracy

class BroadQuestionClassifier:

    BROAD_PATTERNS = {
        "roles": [
            "different roles",
            "roles in toastmasters",
            "list of roles",
            "toastmasters roles",
            "all roles"
        ],
        "pathways": [
            "different pathways",
            "list of pathways",
            "toastmasters pathways",
            "all pathways"
        ]
    }

    # Return core_knowledge key if broad question, else None
    def classify(self, query: str):
        q = query.lower()

        for key, patterns in self.BROAD_PATTERNS.items():
            for p in patterns:
                if p in q:
                    return key
        
        return None  # not a broad question

