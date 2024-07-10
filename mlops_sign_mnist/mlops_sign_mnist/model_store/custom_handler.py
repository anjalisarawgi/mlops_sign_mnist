# Custom handler for the local deployment with torchserve

import json
import torch
from ts.torch_handler.base_handler import BaseHandler

ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


class MyHandler(BaseHandler):
    def initialize(self, context):
        self.model = torch.jit.load('models/scripted_model.pt"')
        self.model.eval()

# data can be fed into model as json (that is in fact a tensor)
    def preprocess(self, data):
        # Read the file and parse the JSON content
        data_content = data[0].get('body')
        input_json = json.loads(data_content.decode('utf-8'))
        print("the data content is ")
        print(data_content)
        print("the input json is ")
        print(input_json)
        # Convert JSON data to tensor
        tensor = torch.tensor(input_json['input']).float()
        return tensor

    def inference(self, inputs):
        # Run inference on the preprocessed data
        with torch.no_grad():
            output = self.model(inputs)
            correct = (output.argmax(dim=1))
            prediction = ALPHABET[correct.item()]
        return prediction

    def postprocess(self, inference_output):
        # Postprocess the inference output
        result = inference_output.numpy().tolist()
        return [json.dumps({'output': result})]

