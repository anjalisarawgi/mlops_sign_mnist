import torch
from models.model import SignLanguageMNISTModel

# NOTE: After scripting the model, comment that chunck of code 

# load regular model
model_checkpoint = "../models/sign_language_mnist_model.pth"
model = SignLanguageMNISTModel()
model.load_state_dict(torch.load(model_checkpoint))

# Script your model
scripted_model = torch.jit.script(model)

# Save the scripted model if needed
scripted_model.save("models/scripted_model.pt")

