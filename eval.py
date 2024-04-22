import json
from bert_score import score as b_score

# Load the inference data
with open('./transfomer_pt/white/s:True_epos:False/inference.json', 'r') as f:
    inference_data = json.load(f)

# Initialize lists to store BERTScores and manual evaluation results
bert_scores = []

for pid, data in inference_data.items():
    reference = data['ground_truth']
    prediction = data['prediction']

    # Calculate BERTScore
    _, _, bert_score = b_score([prediction], [reference], lang='en')
    bert_scores.append(bert_score.item())

# Calculate average BERTScore
avg_bert = sum(bert_scores) / len(bert_scores)

# Print the average BERTScore
print(f'Average BERT score: {avg_bert}')

# Saving the average BERT score to a file
with open('evaluation_scores.json', 'w') as f:
    json.dump({'avg_bert': avg_bert}, f)
