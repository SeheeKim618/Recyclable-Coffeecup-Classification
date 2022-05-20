import os
import torch

def save_model(model_path, state, file_name):
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    torch.save(state, model_path + file_name)
    print('Model Saved!')

def get_reload_weight(model_path, model):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dicts'])
    return model
