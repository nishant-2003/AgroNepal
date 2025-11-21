from utils.datasetloader.loader import Datasetloader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

current_script_directory=Path(__file__).parent
PROJECT_ROOT=current_script_directory.parent.parent 
dataset_path=PROJECT_ROOT / "Datasets" / "files"

file_directory=Datasetloader(dataset_path)
loaded_files=file_directory.smart_loader()

class SplittingData:
    """This is a smart splitter using recursive chunker for pdf files and just adding metadata for csv files"""
    def __init__(self, files_list):
        self.files_list = files_list

    def splitter(self):
        pdf_splits = []
        csv_splits = []
        pdf_string = ".pdf"

        for file_doc in self.files_list:
            #print(f"\nProcessing file: {file_doc.metadata.get('source', 'Unknown')}")
            
            
            if (pdf_string in file_doc.metadata["source"]):
                #print("-> Applying Recursive Character Splitting (PDF logic)")
                
                recursive_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, #just for testing small chunk size
                    chunk_overlap=20,
                    separators=["\n\n", "\n", " ", ""]
                )
                
                
                recursive_chunks = recursive_splitter.split_documents([file_doc])
                
                
                pdf_splits.extend(recursive_chunks)
                
            else:
                #print("-> Applying Direct Chunking (CSV logic)")
                
                csv_splits.append(file_doc)
                
        
        all_docs_split = pdf_splits + csv_splits
        #print(f"\nSummary: {len(pdf_splits)} PDF chunks created, {len(csv_splits)} CSV chunks added.")
        return all_docs_split
#if __name__=="__main__":
    #loader=SplittingData(loaded_files)
    #list=loader.splitter()
    #print(list)
