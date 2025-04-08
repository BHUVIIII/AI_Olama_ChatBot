import tkinter as tk
from tkinter import scrolledtext, font
import requests
import json
import time
import threading
import re

# Define color scheme
DARK_BG = "#1E1E1E"        # Dark background
DARKER_BG = "#252526"      # Slightly darker background for contrast
TEXT_COLOR = "#4AF626"     # Matrix-style green
BUTTON_BG = "#2C2C2C"      # Button background
BUTTON_FG = "#4AF626"      # Button text color
ENTRY_BG = "#2D2D2D"       # Input field background
ENTRY_FG = "#4AF626"       # Input field text color
CODE_BG = "#1C1C1C"        # Code block background
CODE_FG = "#00FFFF"        # Code text color
KEYWORD_COLOR = "#FF69B4"   # Programming keyword color

def clear_chat():
    chat_history.delete(1.0, tk.END)

def change_model(event):
    selected_model = model_var.get()
    chat_history.insert(tk.END, f"Switched to model: {selected_model}\n\n")

def configure_tags():
    chat_history.tag_configure("code", foreground=CODE_FG, background=CODE_BG, font=("Consolas", 10))
    chat_history.tag_configure("normal", foreground=TEXT_COLOR, font=("Consolas", 10))
    chat_history.tag_configure("keyword", foreground=KEYWORD_COLOR, font=("Consolas", 10, "bold"))
    chat_history.tag_configure("assistant", foreground="#00FF00", font=("Consolas", 10, "bold"))

def highlight_keywords(start_index, end_index):
    keywords = [
        "def", "class", "import", "from", "return", "if", "else", "elif", "while", "for",
        "try", "except", "finally", "with", "as", "in", "is", "not", "and", "or",
        "True", "False", "None", "async", "await", "lambda"
    ]
    
    for keyword in keywords:
        start = start_index
        while True:
            start = chat_history.search(r'\m' + keyword + r'\M', start, end_index, regexp=True)
            if not start:
                break
            end = f"{start}+{len(keyword)}c"
            chat_history.tag_add("keyword", start, end)
            start = end

def stream_response(response_text):
    chat_history.insert(tk.END, "Assistant: ", "assistant")
    
    in_code_block = False
    code_block_content = ""
    current_line = ""
    
    lines = response_text.split('\n')
    
    for line in lines:
        if line.startswith('```'):
            if in_code_block:
                if code_block_content:
                    start_index = chat_history.index("end-1c linestart")
                    chat_history.insert(tk.END, code_block_content + '\n', "code")
                    end_index = chat_history.index("end-1c")
                    highlight_keywords(start_index, end_index)
                    code_block_content = ""
                in_code_block = False
            else:
                in_code_block = True
            continue
        
        if in_code_block:
            code_block_content += line + '\n'
        else:
            for char in line + '\n':
                chat_history.insert(tk.END, char, "normal")
                chat_history.see(tk.END)
                chat_history.update()
                time.sleep(0.01)
    
    chat_history.insert(tk.END, "\n")
    chat_history.see(tk.END)

def send_message():
    user_input = entry.get()
    if not user_input.strip():
        return
    
    chat_history.insert(tk.END, "You: " + user_input + "\n")
    chat_history.see(tk.END)
    
    send_button.config(state='disabled', text='Sending...')
    entry.delete(0, tk.END)
    
    def process_response():
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model_var.get(),
                    'prompt': user_input,
                    'stream': False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_response = result.get('response', '')
                root.after(0, lambda: stream_response(assistant_response))
            else:
                chat_history.insert(tk.END, f"Error: HTTP {response.status_code}\n\n")
                
        except Exception as e:
            chat_history.insert(tk.END, f"Error: {str(e)}\n\n")
        finally:
            root.after(0, lambda: send_button.config(state='normal', text='Send'))
    
    threading.Thread(target=process_response, daemon=True).start()

# Create main window
root = tk.Tk()
root.title("Local Ollama Chat")
root.geometry("600x800")
root.configure(bg=DARK_BG)

# Create model selector
models = ['phi3', 'openchat', 'infosys/nt-java']
model_var = tk.StringVar(value='phi3')
model_frame = tk.Frame(root, bg=DARK_BG)
model_frame.pack(pady=5)

tk.Label(
    model_frame, 
    text="Select Model:", 
    font=("Arial", 10, "bold"),
    bg=DARK_BG,
    fg=TEXT_COLOR
).pack(side='left', padx=5)

model_menu = tk.OptionMenu(model_frame, model_var, *models, command=change_model)
model_menu.configure(
    font=("Arial", 10),
    bg=BUTTON_BG,
    fg=TEXT_COLOR,
    activebackground=DARKER_BG,
    activeforeground=TEXT_COLOR,
    highlightbackground=DARK_BG,
    highlightcolor=TEXT_COLOR
)
model_menu["menu"].configure(
    bg=BUTTON_BG,
    fg=TEXT_COLOR,
    activebackground=DARKER_BG,
    activeforeground=TEXT_COLOR
)
model_menu.pack(side='left')

# Create chat history display
chat_history = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    width=60,
    height=30,
    font=("Consolas", 10),
    bg=DARKER_BG,
    fg=TEXT_COLOR,
    insertbackground=TEXT_COLOR,
    selectbackground=TEXT_COLOR,
    selectforeground=DARKER_BG
)
chat_history.pack(padx=10, pady=10, expand=True, fill='both')

# Create input frame
input_frame = tk.Frame(root, bg=DARK_BG)
input_frame.pack(padx=10, pady=(0, 10), fill='x')

# Create input field
entry = tk.Entry(
    input_frame,
    width=50,
    font=("Consolas", 10),
    bg=ENTRY_BG,
    fg=TEXT_COLOR,
    insertbackground=TEXT_COLOR,
    relief="flat"
)
entry.pack(side='left', expand=True, fill='x', padx=(0, 10))

# Create send button
send_button = tk.Button(
    input_frame,
    text="Send",
    command=send_message,
    width=10,
    bg=BUTTON_BG,
    fg=TEXT_COLOR,
    font=("Arial", 10, "bold"),
    relief="flat",
    activebackground=DARKER_BG,
    activeforeground=TEXT_COLOR
)
send_button.pack(side='right')

# Create clear chat button
clear_button = tk.Button(
    root,
    text="Clear Chat",
    command=clear_chat,
    bg=BUTTON_BG,
    fg=TEXT_COLOR,
    font=("Arial", 10, "bold"),
    width=10,
    relief="flat",
    activebackground=DARKER_BG,
    activeforeground=TEXT_COLOR
)
clear_button.pack(pady=(0, 10))

# Bind Enter key to send message
entry.bind('<Return>', lambda event: send_message())

# Configure hover effects
def on_enter(e):
    e.widget['background'] = DARKER_BG

def on_leave(e):
    e.widget['background'] = BUTTON_BG

# Bind hover events to buttons
for button in [send_button, clear_button]:
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)

# Configure text tags
configure_tags()

# Initial welcome message
chat_history.insert(tk.END, "Welcome to Ollama Chat! Select a model and start chatting.\n", "normal")
chat_history.insert(tk.END, "Code blocks will be highlighted automatically.\n\n", "normal")

# Start the GUI
root.mainloop()