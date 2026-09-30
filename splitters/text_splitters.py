from langchain_text_splitters import RecursiveCharacterTextSplitter

class text_splitters:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    def split(self, documents):
        chunks = self.splitter.split_documents(documents)  
        return chunks  