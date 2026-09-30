class RetrievelComparison:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        
    def evaluate(self, retriever, tests):
        precision_scores = []
        recall_scores =  []
        mrr_scores = []
        for test in tests:
            result = retriever.search(query=test["question"])
            retrieved_id = [doc.metadata.get("chunk_id", "N/A") for doc in result]
            relevant_ids = test["relevant_ids"]
            
            precision = self.evaluator.precision_at_k(retrieved_id, relevant_ids)
            recall = self.evaluator.recall_at_k(retrieved_id, relevant_ids)
            mrr = self.evaluator.mmr(retrieved_id, relevant_ids) 
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            mrr_scores.append(mrr) 
            
        return {
            "precision": sum(precision_scores)/len(precision_scores), 
            "recall" : sum(recall_scores)/len(recall_scores),
            "mrr" : sum(mrr_scores)/ len (mrr_scores)
        }   
            