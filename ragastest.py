from datasets import Dataset 
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

data_samples = {
    'question': ['When was the first super bowl?', 'When was the first super bowl?'],
    'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'contexts' : [['The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'], 
    ['The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,']],
    'ground_truth': ['The first superbowl was held on Jan 15, 1967', 'The first superbowl was held on Jan 15, 1967']
}
dataset = Dataset.from_dict(data_samples)
# relevancy_score = evaluate(dataset,metrics=[answer_relevancy])
# faithfulness_score = evaluate(dataset,metrics=[faithfulness])
# relevancy_score.to_pandas()
# faithfulness_score.to_pandas()
# print(relevancy_score)
# print(faithfulness_score)
results = {}

for metric in [answer_relevancy, faithfulness, context_recall, context_precision]:
    result = evaluate(dataset,metrics=[metric])
    num = result.to_pandas()
    results[metric.name] = num[metric.name].to_dict()
    print(result)
print(results)

#
