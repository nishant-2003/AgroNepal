from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
filepaths=[]
all_docs=[]
file_metadata={}
class Datasetloader:
    """This constructor can be used for Loading Datasets for NepAgri Rag chain
    where it is an effiecient way to organize the data in a single list It Supports pdf
    and csv file"""
    def __init__(self,filedirectory):
        self.filedirectory=filedirectory
    
    def check_library(self):
        """To check whether the module is installed properly or not"""
        try:
            import langchain_community
            
            return True
        except Exception as e:
            print(f"Seems like this is the error {e}")
            return False
    def loader(self):
        """To Loade csv files"""
        test=self.check_library()
        if test==True:
            loader=Path(self.filedirectory)
            return loader

    def smart_loader(self):
        """It automatically writes simple metadata based on title of your csv file"""
        loader=self.loader()
        ## global filepaths
        ## global all_docs
        files=[str(file) for file in loader.glob("*")]
        #files_check=Path(files)
        csv_paths=[]
        #file_paths=[str(file) for file in loader.glob("*.csv")]
        doc_type="dataset"
        version=1.0
        for file_path_str in files:
            files_check=Path(file_path_str)
            #for filer in files_check:
            if files_check.suffix==".csv":
                files_check=Path(file_path_str)
                key=files_check.name
                title=files_check.stem
                file_metadata[key]={
                'path':file_path_str,
                'metadata':{
                    'doctype':doc_type,
                    'version':version,
                    'title':title
                }
            }
        
        all_docs=[]
        #print(len(files))
        print(files)
        for file_path_str in files:
            file_path_obj=Path(file_path_str)
            print(file_path_obj)
            if file_path_obj.suffix==".csv":
                key=file_path_obj.name
                metadata=file_metadata[key]['metadata']
                loader=CSVLoader(file_path=file_path_str)
                docs=loader.load()
               # print(docs[0].metadata)
            elif file_path_obj.suffix==".pdf":
                loader=PyPDFLoader(file_path_str)
                docs=loader.load()
            for doc in docs:
                    if hasattr(doc,'metadata') and isinstance(doc.metadata,dict):
                        doc.metadata.update(metadata)
                    else:
                        doc.metadata['doctype']="pdf"
                        doc.metadata=metadata        
                    all_docs.append(doc)
        print(all_docs)
        return all_docs
            
    
##checking code working or not
#test=Datasetloader("C:\\Users\\KIIT\\Downloads\\dataset")
#var1=test.csv_metadata()
#print(var1)



