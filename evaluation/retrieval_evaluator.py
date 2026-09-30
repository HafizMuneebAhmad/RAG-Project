from sympy import re


class RetrievalEvaluator:
    
    def precision_at_k(self, retrieved_ids, relevant_ids):
         if not retrieved_ids:
            return 0
         relevant_count = sum(
             1
            for doc_id in retrieved_ids
                if doc_id in relevant_ids
            )
         return relevant_count/ len(retrieved_ids)
     
    def recall_at_k(self, retrieved_ids, relevant_ids):
             if not relevant_ids:
                return 0
             relevant_count = sum(
                 1
                for doc_id in retrieved_ids
                    if doc_id in relevant_ids
                )
             return relevant_count/ len(relevant_ids) 
         
    def mmr(self, retrieved_ids, relevant_ids):
             for rank, doc_id in enumerate(
                 retrieved_ids,
                 start=1):
                 if doc_id in relevant_ids:
                     return 1/ rank
                 return 0.0
                         