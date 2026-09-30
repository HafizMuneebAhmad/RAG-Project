class RetrievealTestDataset:
    def __init__(self):
        self.test = [
            {
                "question": "what is the Topic of the document",
                "relevant_ids": {"chunk_0", "chunks_1"}
            },
            {
                "question": "Explain 7 layers of OSI model",
                "relevant_ids": {"chunk_2"}
            },
            {
                "question": "What is the purpose of the OSI model?",
                "relevant_ids": {"chunk_3"}
            }
        ]
    def get_tests(self):
        return self.test  