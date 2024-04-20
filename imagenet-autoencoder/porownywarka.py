
import json
import torch
import torch.nn.functional as F

def read_vector_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        vector = data['popularity']
        return torch.tensor(vector)

popularity_vector = read_vector_from_json('the_popularity_vector.json')

with open('embedding_all_img_list.json', 'r') as datafile:
    dict = json.load(datafile)



for i in range(100):
    img_vec = dict[f"{i}.jpg"]
    img_vec = torch.tensor(img_vec)
    similarity = F.cosine_similarity(img_vec.unsqueeze(0), popularity_vector.unsqueeze(0), dim=1)
    print(f"Obrazek nr {i}, popularnosc = {similarity}")

