import json
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

def exact_match(predictions):
    total = len(predictions)
    exact_matches = sum(1 for pred in predictions if pred['ground_truth'] == pred['prediction'])
    return exact_matches / total * 100

def token_level_evaluation(predictions):
    total_precision = total_recall = total_f1 = 0
    for pred in predictions:
        ground_truth_tokens = pred['ground_truth'].split()
        prediction_tokens = pred['prediction'].split()
        common_tokens = set(ground_truth_tokens) & set(prediction_tokens)
        precision = len(common_tokens) / len(prediction_tokens)
        recall = len(common_tokens) / len(ground_truth_tokens)
        f1 = 2 * ((precision * recall) / (precision + recall + 1e-8))
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    avg_precision = total_precision / len(predictions)
    avg_recall = total_recall / len(predictions)
    avg_f1 = total_f1 / len(predictions)
    return avg_precision, avg_recall, avg_f1

def calculate_bleu(predictions):
    references = [[pred['ground_truth'].split() for pred in predictions]]
    hypotheses = [pred['prediction'].split() for pred in predictions]
    return corpus_bleu(references, hypotheses)

def calculate_rouge(predictions):
    rouge = Rouge()
    total_scores = {'rouge-1': {'p': 0, 'r': 0, 'f': 0},
                    'rouge-2': {'p': 0, 'r': 0, 'f': 0},
                    'rouge-l': {'p': 0, 'r': 0, 'f': 0}}
    for pred in predictions:
        scores = rouge.get_scores(pred['prediction'], pred['ground_truth'])[0]
        for metric in total_scores.keys():
            for score_type in ['p', 'r', 'f']:
                total_scores[metric][score_type] += scores[metric][score_type]
    for metric in total_scores.keys():
        for score_type in ['p', 'r', 'f']:
            total_scores[metric][score_type] /= len(predictions)
    return total_scores

def main():
    with open('./transfomer_pt/white/s:True_epos:False/inference.json', 'r') as file:
        predictions = json.load(file)
    
    em = exact_match(predictions)
    avg_precision, avg_recall, avg_f1 = token_level_evaluation(predictions)
    bleu = calculate_bleu(predictions)
    rouge = calculate_rouge(predictions)
    
    print("Exact Match (EM): {:.2f}%".format(em))
    print("Token-level Evaluation:")
    print("  Precision: {:.4f}".format(avg_precision))
    print("  Recall: {:.4f}".format(avg_recall))
    print("  F1-score: {:.4f}".format(avg_f1))
    print("BLEU Score: {:.4f}".format(bleu))
    print("ROUGE Score:")
    for metric, scores in rouge.items():
        print("  {}: Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}".format(metric, scores['p'], scores['r'], scores['f']))

if __name__ == "__main__":
    main()
