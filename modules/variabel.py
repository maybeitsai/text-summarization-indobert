import torch_directml

# Inisialisasi device untuk DirectML
DEVICE = torch_directml.device()

# Model 
MODEL = "cahya/bert2gpt-indonesian-summarization"

MIN_LENGTH = 64
MAX_LENGTH = 512