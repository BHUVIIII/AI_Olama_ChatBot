import tkinter as tk
from tkinter import scrolledtext, filedialog
import requests
import json
from typing import List
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader
)
import numpy as np
from typing import List, Any
import warnings
warnings.filterwarnings('ignore')

# Define color scheme (keeping your dark theme)
DARK_BG = "#1E1E1E"
DARKER_BG = "#252526"
TEXT_COLOR = "#4AF626"
BUTTON_BG = "#2C2C2C"
BUTTON_FG = "#4AF626"
ENTRY_BG = "#2D2D2D"
ENTRY_FG = "#4AF626"

class SimpleEmbeddings:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
        self.fitted = False
    
    def fit_vectorizer(self, texts):
        if not self.fitted:
            self.vectorizer.fit(texts)
            self.fitted = True
    
    def __call__(self, texts):
        """Make the class callable - this is required by FAISS"""
        if isinstance(texts, str):
            return self.embed_query(texts)
        return self.embed_documents(texts)
    
    '''def embed_documents(self, texts):
        self.fit_vectorizer(texts)
        vectors = self.vectorizer.transform(texts).toarray()
        return vectors.astype('float32')'''
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search documents."""
        try:
            if isinstance(texts[0], str):
                # If input is list of strings
                documents = texts
            else:
                # If input is list of Document objects
                documents = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in texts]
                
            self.fit_vectorizer(documents)
            vectors = self.vectorizer.transform(documents).toarray()
            return vectors.astype('float32')
        except Exception as e:
            print(f"Error in embed_documents: {e}")
            return [[0.0] * 1000] * len(texts)  # Return zero vectors as fallback
    
    '''def embed_query(self, text):
        if not self.fitted:
            return np.zeros(1000, dtype='float32')
        vector = self.vectorizer.transform([text]).toarray()
        return vector[0].astype('float32')'''
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        try:
            if not self.fitted:
                return [0.0] * 1000
            vector = self.vectorizer.transform([text]).toarray()
            return vector[0].astype('float32')
        except Exception as e:
            print(f"Error in embed_query: {e}")
            return [0.0] * 1000  # Return zero vector as fallback

class DocumentStore:
    def __init__(self):
        self.embeddings = SimpleEmbeddings()
        self.vector_store = None
        self.supported_extensions = {
            'txt': TextLoader,
            'pdf': PyPDFLoader,
            'docx': Docx2txtLoader,
            'doc': UnstructuredWordDocumentLoader,
            'csv': CSVLoader,
            'json': JSONLoader,
            'md': UnstructuredMarkdownLoader,
            'html': UnstructuredHTMLLoader,
            'htm': UnstructuredHTMLLoader,
            'pptx': UnstructuredPowerPointLoader,
            'ppt': UnstructuredPowerPointLoader,
            'xlsx': UnstructuredExcelLoader,
            'xls': UnstructuredExcelLoader
        }

    def get_loader_for_file(self, file_path: str):
        """Return appropriate loader based on file extension"""
        extension = file_path.lower().split('.')[-1]
        loader_class = self.supported_extensions.get(extension)
        
        if loader_class is None:
            print(f"Warning: Unsupported file type for {file_path}")
            return None
            
        return loader_class

    '''def load_documents(self, directory_path: str):
        """Load documents from a directory"""
        try:
            documents = []
            loaded_files = 0
            skipped_files = 0
            
            # Walk through directory and process files
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        # Get appropriate loader
                        loader_class = self.get_loader_for_file(file_path)
                        if loader_class is None:
                            skipped_files += 1
                            continue
                            
                        loader = loader_class(file_path)
                        docs = loader.load()
                        documents.extend(docs)
                        loaded_files += 1
                        print(f"Successfully loaded: {file_path}")
                        
                    except Exception as e:
                        print(f"Error loading {file_path}: {str(e)}")
                        skipped_files += 1
                        continue

            if not documents:
                raise ValueError("No documents were successfully loaded")

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            texts = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
            return len(texts), loaded_files, skipped_files
            
        except Exception as e:
            print(f"Error in load_documents: {e}")
            raise'''
    def load_documents(self, directory_path: str):
        """Load documents from a directory"""
        try:
            documents = []
            loaded_files = 0
            skipped_files = 0
            
            # Walk through directory and process files
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        # Get appropriate loader
                        loader_class = self.get_loader_for_file(file_path)
                        if loader_class is None:
                            print(f"Skipping unsupported file: {file_path}")
                            skipped_files += 1
                            continue
                        
                        # Special handling for PDF files
                        if loader_class == PyPDFLoader:
                            try:
                                loader = PyPDFLoader(file_path)
                                docs = loader.load()
                            except Exception as pdf_error:
                                print(f"Error loading PDF {file_path}: {str(pdf_error)}")
                                try:
                                    with open(file_path, 'rb') as pdf_file:
                                        import PyPDF2
                                        reader = PyPDF2.PdfReader(pdf_file)
                                        text = ""
                                        for page in reader.pages:
                                            text += page.extract_text() + "\n"
                                        if text.strip():
                                            from langchain.schema import Document
                                            docs = [Document(page_content=text)]
                                        else:
                                            raise ValueError("No text could be extracted from PDF")
                                except Exception as fallback_error:
                                    print(f"Fallback PDF extraction failed: {str(fallback_error)}")
                                    skipped_files += 1
                                    continue
                        else:
                            loader = loader_class(file_path)
                            docs = loader.load()
                        
                        if docs:
                            documents.extend(docs)
                            loaded_files += 1
                            print(f"Successfully loaded: {file_path}")
                        else:
                            print(f"No content extracted from: {file_path}")
                            skipped_files += 1
                            
                    except Exception as e:
                        print(f"Error loading {file_path}: {str(e)}")
                        skipped_files += 1
                        continue

            if not documents:
                raise ValueError("No documents were successfully loaded")

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            texts = text_splitter.split_documents(documents)
            
            # Create vector store with explicit embedding function
            self.vector_store = FAISS.from_documents(
                documents=texts,
                embedding=self.embeddings
            )
            return len(texts), loaded_files, skipped_files
            
        except Exception as e:
            print(f"Error in load_documents: {e}")
            raise
    
    def search(self, query: str, k: int = 3) -> List[str]:
        """Search for relevant documents"""
        if not self.vector_store:
            return []
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"Error in search: {e}")
            return []

class OllamaRAGChat:
    def __init__(self, root):
        self.root = root
        self.doc_store = DocumentStore()
        self.models = ['phi3', 'openchat', 'infosys/nt-java']
        self.model_var = tk.StringVar(value='phi3')  # Set default model
        self.setup_gui()
        self.show_supported_formats()
    
    def show_supported_formats(self):
        supported_formats = ", ".join(f"*.{ext}" for ext in self.doc_store.supported_extensions.keys())
        message = f"Supported file formats: {supported_formats}\n\n"
        self.chat_history.insert(tk.END, message)
        self.chat_history.see(tk.END)

    def clear_chat(self):
        """Clear chat history and reset document store"""
        # Clear chat history
        self.chat_history.delete('1.0', tk.END)
    
        # Reset document store
        self.doc_store = DocumentStore()
    
        # Show supported formats again
        self.show_supported_formats()
    
        # Update UI to show cleared state
        self.chat_history.insert(tk.END, "Chat and documents cleared. Ready for new conversation.\n\n")
        self.chat_history.see(tk.END)


    def setup_gui(self):
        # Model selector frame
        self.model_frame = tk.Frame(self.root, bg=DARK_BG)
        self.model_frame.pack(pady=5)
        
        # Model selector
        self.models = ['phi3', 'openchat', 'infosys/nt-java']
        self.model_var = tk.StringVar(value='phi3')
        self.setup_model_selector()

        # Button frame for document controls
        self.button_frame = tk.Frame(self.root, bg=DARK_BG)
        self.button_frame.pack(pady=5)
        
        # Load documents button
        self.load_docs_button = tk.Button(
            self.button_frame,
            text="Give Context",
            command=self.load_documents,
            bg=BUTTON_BG,
            fg=TEXT_COLOR,
            font=("Arial", 10, "bold")
        )
        self.load_docs_button.pack(side='left',padx=5)

         # Clear chat button
        self.clear_button = tk.Button(
            self.button_frame,
            text="Clear Chat",
            command=self.clear_chat,
            bg=BUTTON_BG,
            fg=TEXT_COLOR,
            font=("Arial", 10, "bold")
        )
        self.clear_button.pack(side='left', padx=5)
        
        # Chat history
        self.chat_history = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            width=60,
            height=30,
            font=("Consolas", 10),
            bg=DARKER_BG,
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR
        )
        self.chat_history.pack(padx=10, pady=10, expand=True, fill='both')
        
        # Input frame
        self.setup_input_frame()

    def update_model_display(self, *args):
        """Update the display showing which model is selected"""
        selected_model = self.model_var.get()
        self.model_label.config(text=f"Current Model: {selected_model}")
        # Add to chat history
        self.chat_history.insert(tk.END, f"Switched to model: {selected_model}\n\n")
        self.chat_history.see(tk.END)
        
    def setup_model_selector(self):
        # Create a sub-frame for model selection controls
        model_controls = tk.Frame(self.model_frame, bg=DARK_BG)
        model_controls.pack(fill='x')
        tk.Label(
            model_controls,
            text="Select Model:",
            font=("Arial", 10, "bold"),
            bg=DARK_BG,
            fg=TEXT_COLOR
        ).pack(side='left', padx=5)
        
        model_menu = tk.OptionMenu(model_controls, self.model_var, *self.models)
        model_menu.configure(
            font=("Arial", 10),
            bg=BUTTON_BG,
            fg=TEXT_COLOR,
            activebackground=DARKER_BG,
            activeforeground=TEXT_COLOR
        )
        model_menu.pack(side='left')
        # Create label to display current model
        self.model_label = tk.Label(
            self.model_frame,
            text=f"Current Model: {self.model_var.get()}",
            font=("Arial", 10, "bold"),
            bg=DARK_BG,
            fg=TEXT_COLOR
        )
        self.model_label.pack(pady=5)
        # Trace the variable to update the display when changed
        self.model_var.trace('w', self.update_model_display)
        
    def setup_input_frame(self):
        input_frame = tk.Frame(self.root, bg=DARK_BG)
        input_frame.pack(padx=10, pady=(0, 10), fill='x')
        
        self.entry = tk.Entry(
            input_frame,
            width=50,
            font=("Consolas", 10),
            bg=ENTRY_BG,
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR
        )
        self.entry.pack(side='left', expand=True, fill='x', padx=(0, 10))
        self.entry.bind('<Return>', lambda e: self.send_message())
        
        self.send_button = tk.Button(
            input_frame,
            text="Send",
            command=self.send_message,
            bg=BUTTON_BG,
            fg=TEXT_COLOR,
            font=("Arial", 10, "bold")
        )
        self.send_button.pack(side='right')
        

    def load_documents(self):
        filetypes = [
        ('All Supported Files', (
            '*.txt', '*.pdf', '*.docx', '*.doc', 
            '*.csv', '*.json', '*.md', '*.html', 
            '*.htm', '*.pptx', '*.ppt', '*.xlsx', '*.xls'
        )),
        ('Text Files', '*.txt'),
        ('PDF Files', '*.pdf'),
        ('Word Documents', ('*.docx', '*.doc')),
        ('CSV Files', '*.csv'),
        ('JSON Files', '*.json'),
        ('Markdown Files', '*.md'),
        ('HTML Files', ('*.html', '*.htm')),
        ('PowerPoint Files', ('*.pptx', '*.ppt')),
        ('Excel Files', ('*.xlsx', '*.xls')),
        ('All Files', '*.*')
        ]

        # Use askopenfilenames instead of askdirectory to select multiple files
        files = filedialog.askopenfilenames(
            title="Select Documents",
            filetypes=filetypes
        )

        if files:
            try:
                # Create a temporary directory to store selected files
                import tempfile
                import shutil
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Copy selected files to temporary directory
                    for file_path in files:
                        shutil.copy2(file_path, temp_dir)
                
                    # Process the files
                    num_chunks, loaded_files, skipped_files = self.doc_store.load_documents(temp_dir)
                
                    # Show status message
                    status_message = (
                        f"Processing complete:\n"
                        f"- Selected {len(files)} files\n"
                        f"- Successfully loaded {loaded_files} files\n"
                        f"- Skipped {skipped_files} unsupported/error files\n"
                        f"- Created {num_chunks} text chunks for processing\n\n"
                        f"Loaded files:\n"
                    )
                
                    # Add list of processed files
                    for file_path in files:
                        status_message += f"- {os.path.basename(file_path)}\n"
                
                    status_message += "\n"
                    self.chat_history.insert(tk.END, status_message)
                    self.chat_history.see(tk.END)
                
            except Exception as e:
                error_message = f"Error loading documents: {str(e)}\n\n"
                self.chat_history.insert(tk.END, error_message)
                self.chat_history.see(tk.END)
                
    def send_message(self):
        user_input = self.entry.get()
        if not user_input.strip():
            return
            
        self.chat_history.insert(tk.END, "You: " + user_input + "\n")
        self.send_button.config(state='disabled', text='Sending...')
        self.root.update()
        
        try:
            # Get relevant context from document store
            relevant_docs = self.doc_store.search(user_input)
            context = "\n".join(relevant_docs)
            
            # Construct prompt with context
            prompt = f"""Context information is below.
            ---------------------
            {context}
            ---------------------
            Given the context information and not prior knowledge, answer the question: {user_input}
            """
            
            # Send to Ollama
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': self.model_var.get(),
                    'prompt': prompt,
                    'stream': False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_response = result.get('response', '')
                self.chat_history.insert(tk.END, "Assistant: " + assistant_response + "\n\n")
                self.chat_history.see(tk.END)
            else:
                self.chat_history.insert(tk.END, f"Error: HTTP {response.status_code}\n\n")
                
        except Exception as e:
            self.chat_history.insert(tk.END, f"Error: {str(e)}\n\n")
        finally:
            self.send_button.config(state='normal', text='Send')
            self.entry.delete(0, tk.END)

# Create and run the application
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Ollama RAG Chat")
    root.geometry("600x800")
    root.configure(bg=DARK_BG)
    
    app = OllamaRAGChat(root)
    root.mainloop()
