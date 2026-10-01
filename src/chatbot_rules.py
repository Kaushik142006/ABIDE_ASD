
SYNONYM_MAP: dict = {
    
    "result": "prediction", "results": "prediction",
    "output": "prediction", "answer": "prediction",
    "classify": "prediction", "classified": "prediction",
    "classification": "prediction", "predicted": "prediction",
    "predicts": "prediction",
    "detected": "prediction", "detection": "prediction",
    "finding": "prediction", "findings": "prediction",
    "verdict": "prediction", "outcome": "prediction",
    "confidence": "probability", "certainty": "probability",
    "certain": "probability", "sure": "probability",
    "likelihood": "probability", "likely": "probability",
    "chances": "probability", "chance": "probability",
    "percentage": "probability", "percent": "probability",
    "score": "probability",
    "classifier": "model", "algorithm": "model",
    "method": "model",
    "edge": "feature", "edges": "features",
    "variable": "feature", "variables": "features",
    "input": "feature", "inputs": "features",
    "attribute": "feature", "attributes": "features",
    "dimension": "feature", "dimensions": "features",
    "workflow": "pipeline", "process": "pipeline",
    "steps": "pipeline", "procedure": "pipeline",
    "system": "pipeline", "architecture": "pipeline",
    "cleaning": "preprocessing", "clean": "preprocessing",
    "preparation": "preprocessing", "prepare": "preprocessing",
    "normalize": "preprocessing", "normalise": "preprocessing",
    "normalization": "preprocessing", "normalisation": "preprocessing",
    "autism": "asd", "autistic": "asd",
    "autism spectrum disorder": "asd",
    "autism spectrum": "asd",
    "typical": "control", "healthy": "control",
    "normal": "control", "non-asd": "control",
    "td": "control", "con": "control",
    "typical control": "control",
    "performance": "evaluation",
    "metric": "evaluation", "metrics": "evaluation",
    "diagnose": "diagnosis", "diagnosed": "diagnosis",
    "diagnostic": "diagnosis", "clinical": "diagnosis",
    "medical": "diagnosis",
}
INTENT_PATTERNS: dict = {
    "PREDICTION": {
        "keywords": {"prediction", "predict", "predicted", "result", "classify",
                      "classified", "classification", "class", "label", "output",
                      "outcome", "detection", "detected", "asd", "autism", "control"},
        "phrases": [
            "what did it predict", "what is the result", "is this asd",
            "is this control", "what class was predicted",
            "what did the model say", "what was predicted",
            "show me the prediction", "what is the prediction",
            "what did the model predict", "prediction result",
            "what was the outcome", "what is the output",
            "what did it detect", "is it asd", "is it autism",
            "is it control", "is it normal", "is it typical",
            "asd or control", "autism or control", "asd", "autism", "control",
        ],
        "weight": 1.0,
    },

    "PROBABILITY": {
        "keywords": {"probability", "probabilities", "confidence", "confident",
                      "certain", "certainty", "percentage", "percent", "chances",
                      "chance", "likelihood", "likely", "score"},
        "phrases": [
            "what is the asd probability", "control probability",
            "how certain is it", "what percentage", "what are the chances",
            "how confident is the model", "how sure is the model",
            "probability distribution", "probability of asd",
            "probability of autism", "probability of control",
            "what is the confidence", "confidence level",
            "how likely", "what are the probabilities",
            "asd probability", "asd percentage",
        ],
        "weight": 1.0,
    },

    "WHY_PREDICTION": {
        "keywords": {"why", "reason", "because", "cause", "explain",
                      "explanation", "justification", "rationale"},
        "phrases": [
            "why asd", "why control", "why did it say autism",
            "why is this asd", "why did the model classify this as asd",
            "how come it predicted autism", "what made it give asd",
            "what caused the prediction", "why did it predict",
            "why was this classified", "explain the prediction",
            "explain the result", "why this result",
            "what influenced the prediction", "what factors",
            "why did the model say", "reason for prediction",
            "why autism", "why not asd", "why not autism",
            "why typical control", "why control",
            "explain why", "how did it decide",
        ],
        "weight": 1.2,
    },

    "MODEL_INFO": {
        "keywords": {"model", "svm", "classifier", "algorithm"},
        "phrases": [
            "what model was used", "which classifier",
            "what is svm", "what is the model", "tell me about the model",
            "describe the model", "model used", "which model",
            "type of model", "what kind of model",
            "support vector machine", "what algorithm",
        ],
        "weight": 1.0,
    },

    "MODEL_PARAMS": {
        "keywords": {"hyperparameter", "hyperparameters", "parameter",
                      "parameters", "gamma", "kernel", "rbf"},
        "phrases": [
            "what is c 10", "what is c=10", "what does c=10 mean",
            "what is gamma", "what is rbf", "why rbf",
            "what is the kernel", "radial basis function",
            "why rbf-svm", "why rbf svm", "what is rbf kernel",
            "balanced class weights", "why balanced",
            "what are the hyperparameters", "model parameters",
            "c parameter", "gamma parameter", "kernel parameter",
            "what does gamma mean", "why c equals 10",
        ],
        "weight": 1.1,
    },

    
    "FEATURES": {
        "keywords": {"features", "feature", "edges", "edge", "selected",
                      "selection"},
        "phrases": [
            "how many features", "why 600", "why 600 features",
            "what are the 19900 edges", "what is an edge",
            "what is a selected feature", "which features were important",
            "important features", "feature selection",
            "how are features selected", "what features are used",
            "number of features", "how many edges",
            "what are edges", "selected edges",
            "feature importance", "most important features",
            "which features matter", "feature count",
            "19900 edges", "600 features",
        ],
        "weight": 1.0,
    },

    "PIPELINE": {
        "keywords": {"pipeline", "workflow", "process", "steps",
                      "procedure", "system", "architecture"},
        "phrases": [
            "how does the system work", "how does it work",
            "what is the pipeline", "describe the pipeline",
            "explain the workflow", "processing steps",
            "what happens to the image", "what steps are taken",
            "end to end process", "end-to-end",
            "how does the model work", "system overview",
            "how does prediction work",
        ],
        "weight": 0.9,
    },

    "ROI_INFO": {
        "keywords": {"roi", "rois", "atlas", "cc200", "parcellation",
                      "region", "regions"},
        "phrases": [
            "what is cc200", "what is an roi", "what are rois",
            "region of interest", "regions of interest",
            "brain regions", "what atlas", "craddock atlas",
            "cc200 atlas", "brain parcellation",
            "how many rois", "200 rois", "200 regions",
        ],
        "weight": 1.0,
    },

    
    "FC_INFO": {
        "keywords": {"connectivity", "fc", "pearson", "correlation",
                      "fisher", "arctanh"},
        "phrases": [
            "what is functional connectivity",
            "functional connectivity matrix",
            "what is pearson correlation", "pearson correlation",
            "what is fisher transformation", "fisher z transform",
            "r to z", "r-to-z transformation",
            "what is fc", "connectivity matrix",
            "how is connectivity computed",
            "brain connectivity", "correlation matrix",
        ],
        "weight": 1.0,
    },

    
    "PREPROCESSING": {
        "keywords": {"preprocessing", "standardization", "standardize",
                      "normalize", "normalization", "cleaning", "clipping",
                      "percentile", "scaler"},
        "phrases": [
            "why preprocessing", "what preprocessing was performed",
            "what is standardization", "standard scaler",
            "what is cleaning", "how is data cleaned",
            "percentile clipping", "5th and 95th percentile",
            "outlier removal", "data preparation",
            "how is data preprocessed",
        ],
        "weight": 0.9,
    },

    
    "TTEST_INFO": {
        "keywords": {"welch", "ttest", "t-test"},
        "phrases": [
            "what is welch t-test", "welch test",
            "what is t-test", "statistical test",
            "how are features selected",
            "feature selection method",
            "welch two sample", "welch t test",
        ],
        "weight": 1.0,
    },

   
    "EVALUATION": {
        "keywords": {"accuracy", "sensitivity", "specificity", "f1",
                      "roc", "auc", "mcc", "evaluation", "performance",
                      "metric", "metrics", "recall", "precision"},
        "phrases": [
            "what is accuracy", "what is sensitivity",
            "what is specificity", "what is f1",
            "what is roc-auc", "what is roc auc", "what is mcc",
            "what is matthews correlation",
            "model performance", "evaluation metrics",
            "how well does the model perform",
            "how accurate is the model",
            "evaluation results", "performance metrics",
            "what is recall", "what is precision",
            "what is balanced accuracy",
        ],
        "weight": 1.0,
    },

    
    "CONFUSION_MATRIX": {
        "keywords": {"confusion"},
        "phrases": [
            "what is the confusion matrix", "confusion matrix",
            "what is a false positive", "false positive",
            "what is a false negative", "false negative",
            "what is a true positive", "true positive",
            "what is a true negative", "true negative",
            "type 1 error", "type 2 error",
            "type i error", "type ii error",
        ],
        "weight": 1.0,
    },

  
    "LIMITATIONS": {
        "keywords": {"limitations", "limitation", "wrong", "reliable",
                      "trustworthy", "guarantee", "error"},
        "phrases": [
            "can the model be wrong", "is this reliable",
            "what are the limitations", "model limitations",
            "can this guarantee", "how reliable",
            "can it make mistakes", "is it always correct",
            "is the model perfect", "what if its wrong",
            "what are the weaknesses",
        ],
        "weight": 1.0,
    },

   
    "MEDICAL_DISCLAIMER": {
        "keywords": {"diagnosis", "clinical", "diagnostic", "medical",
                      "doctor", "diagnose", "diagnosed", "treatment",
                      "therapy", "psychiatrist", "psychologist"},
        "phrases": [
            "does this mean i have autism", "do i have autism",
            "am i autistic", "is this a diagnosis",
            "can this be used clinically", "clinical diagnosis",
            "medical diagnosis", "should i see a doctor",
            "does this confirm autism", "is this a medical test",
            "can this replace a doctor", "clinical use",
            "why am i autistic",
        ],
        "weight": 1.3,
    },

    
    "FILE_INFO": {
        "keywords": {"file", "upload", "uploaded", "filename", "scan",
                      "image", "nii", "jpg", "png"},
        "phrases": [
            "what file was uploaded", "uploaded file",
            "what is the filename", "file information",
            "what format", "file type", "file dimensions",
            "tell me about the file", "about the upload",
            "what did i upload", "which file",
        ],
        "weight": 0.9,
    },

    "GREETING": {
        "keywords": {"hello", "hi", "hey", "greetings", "howdy",
                      "morning", "afternoon", "evening"},
        "phrases": [
            "good morning", "good afternoon", "good evening",
            "hello there", "hi there", "hey there",
        ],
        "weight": 0.5,
    },

    
    "HELP": {
        "keywords": {"help", "commands", "capabilities", "options"},
        "phrases": [
            "what can you do", "what can i ask",
            "help me", "show help", "list commands",
            "what questions can i ask", "what do you know",
            "what topics", "how can you help",
        ],
        "weight": 0.8,
    },
}



FOLLOWUP_TRIGGERS = {
    "why",
    "how",
    "explain",
    "elaborate",
    "more",
    "details",
    "clarify",
}

FOLLOWUP_REFERENCE_WORDS = {
    "that", "this", "it", "its", "the", "those", "these",
}

FOLLOWUP_QUALIFIER_PHRASES = [
    "is that high", "is that low", "is that good", "is that bad",
    "is it high", "is it low", "is it good", "is it bad",
    "is that normal", "is that typical", "is that significant",
    "what does that mean", "what does it mean",
    "tell me more", "can you explain", "go on",
    "why is that", "how come",
]
