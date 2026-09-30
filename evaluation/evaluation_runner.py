from evaluation.retrieval_evaluator import RetrievalEvaluator
from evaluation.test_dataset import RetrievealTestDataset

class EvaluationRunner:
    def __init__(self, retriever):
        self.retriever = retriever
        self.evaluator = RetrievalEvaluator()
        self.dataset = RetrievealTestDataset()

    def run(self):
        tests = self.dataset.get_tests()
        

        precision_scores = []
        recall_scores = []
        mrr_scores = []
        
        for test in tests:
            question = test["question"]
            relevant_ids = test["relevant_ids"]
            result = self.retriever.search(query=question, candidate_k=10, final_k=3)
            retrieved_id = [doc.metadata.get("chunk_id", "N/A") for doc in result]
            
            precision = self.evaluator.precision_at_k(retrieved_id, relevant_ids)
            recall = self.evaluator.recall_at_k(retrieved_id, relevant_ids)
            mrr = self.evaluator.mmr(retrieved_id, relevant_ids)

            precision_scores.append(precision)
            recall_scores.append(recall)
            mrr_scores.append(mrr)
            
        print("\nQuestion:", question)
        print("Precision:", precision_scores)
        print("Recall:", recall_scores)
        print("MRR:", mrr_scores)
        
        print("\n=== Overall Evaluation Results ===")
        print("Average Precision:", sum(precision_scores) / len(precision_scores))
        print("Average Recall:", sum(recall_scores) / len(recall_scores))
        print("Average MRR:", sum(mrr_scores) / len(mrr_scores))    