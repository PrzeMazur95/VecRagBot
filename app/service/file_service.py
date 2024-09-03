import os

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

class FileService():
    
    def __init__(self):
        self.CURRENT_DIRECTORY = os.path.dirname(__file__)
        self.APP_DIRECTORY = Path(__file__).parents[1]
        self.UPLOADED_FILES_DIRECTORY = self.APP_DIRECTORY / "files"
        self.ALLOWED_EXTENSIONS = {'.pdf', '.txt'}
    
    def allowed_file_extension(self, filename: str):
        return self.get_file_extension(filename) in self.ALLOWED_EXTENSIONS
    
    
    def get_uploaded_files_directory(self, filename: str) -> str:
        return self.APP_DIRECTORY + self.UPLOADED_FILES_DIRECTORY + filename
    
    def load_pdf_content(self, filename: str):
        try:
            file_path = self.UPLOADED_FILES_DIRECTORY / filename
            loader = PyPDFLoader(file_path)
            return loader.load() 
        except Exception as e:
            print(f'An error occurred while loading the PDF file: {e}')
            return None
    
    def save_file(self, file):
        try:
            file_path = self.UPLOADED_FILES_DIRECTORY / file.filename
            file.save(file_path)
            return True
        except Exception as e:
            print(f'An error occurred while saving the file: {e}')
            return False
    
    def get_file_extension(self, filename: str) -> str:
        return os.path.splitext(filename)[1].lower()
    
    def load_txt_content(self, filename: str):
        file_path = self.UPLOADED_FILES_DIRECTORY / filename
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return Document(page_content=content)
        except Exception as e:
            print(f'An error occurred while loading the TXT file: {e}')
            return None
        
    def get_base_file_name(self, filename: str) -> str:
            base_name = filename.split('.')[0]
            return base_name