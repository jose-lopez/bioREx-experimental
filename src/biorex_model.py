# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("fill-mask", model="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")

# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
model = AutoModelForMaskedLM.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")

