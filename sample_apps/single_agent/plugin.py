import PyPDF2
import os
from pathlib import Path
from semantic_kernel.functions import kernel_function


class FileReaderPlugin:
    def __init__(self):
        current_dir = Path(__file__).parent.absolute()
        self.data_dir = current_dir.parent.parent / 'data'
        self.pdf_content = {}
        self._load_pdf_files()

    def _load_pdf_files(self):
        """Load all PDF files from the data directory"""
        if self.data_dir.exists():
            for pdf_file in self.data_dir.glob('*.pdf'):
                self.pdf_content[pdf_file.name] = None  # Lazy loading

    @kernel_function(description="List available PDF files in the data directory")
    async def list_pdfs(self) -> str:
        """List all available PDF files"""
        files = list(self.pdf_content.keys())
        return "Available PDF files:\n" + "\n".join(files) if files else "No PDF files found"

    @kernel_function(description="Read and search content from PDF files")
    async def read_pdf(self, file_name: str = None) -> str:
        """Read content from a specific PDF file or return error if file not found"""
        if not file_name:
            return await self.list_pdfs()
        
        if file_name not in self.pdf_content:
            return f"File {file_name} not found. {await self.list_pdfs()}"
            
        if self.pdf_content[file_name] is None:  # Lazy load content
            try:
                file_path = os.path.join(self.data_dir, file_name)
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    self.pdf_content[file_name] = text
            except Exception as e:
                return f"Error reading {file_name}: {str(e)}"
        
        return self.pdf_content[file_name]