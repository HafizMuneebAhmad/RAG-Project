class RRFFussion:
    def __init__(self, constant=60):
        self.constant = constant
        
    def combined(self, vector_search, bm25_search, k=3):
            scores = {}
            documents = {}
            for rank, document in enumerate(vector_search, start=1):
                doc_id = document.page_content
                documents[doc_id] = document
             
                scores[doc_id] = scores.get(doc_id, 0) + (1/(self.constant + rank))   
             
             
            for rank, document in enumerate(bm25_search, start=1):
                    doc_id = document.page_content
                    documents[doc_id] = document
                     
                    scores[doc_id] = scores.get(doc_id, 0) + (1/(self.constant + rank)) 
         
            ranked_ids = sorted(scores,
                            key=scores.get,
                            reverse=True) 
            return [documents[doc_id] for doc_id in ranked_ids[:k]]          