import tkinter as tk
from tkinter import filedialog, messagebox
import time
import os
import webbrowser
from search_engine.loader import load_all_data
from search_engine.engine import search_query

# Load dữ liệu một lần
vocab, metadata, tfidf_data, inverted_index = load_all_data()

class SearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tìm kiếm văn bản")
        self.root.geometry("600x300")
        self.file_path = None

        self.label = tk.Label(root, text="Chọn file truy vấn:")
        self.label.pack(pady=10)

        self.choose_btn = tk.Button(root, text="Chọn file", command=self.choose_file)
        self.choose_btn.pack()

        self.search_btn = tk.Button(root, text="Tìm kiếm", command=self.search)
        self.search_btn.pack(pady=10)

        self.result_label = tk.Label(root, text="", fg="blue")
        self.result_label.pack()

        self.result_links = []
        for _ in range(3):
            lbl = tk.Label(root, text="", fg="blue", cursor="hand2")
            lbl.pack()
            lbl.bind("<Button-1>", self.open_file)
            self.result_links.append(lbl)

    def choose_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if self.file_path:
            self.label.config(text=f"Đã chọn: {self.file_path}")

    def search(self):
        if not self.file_path:
            messagebox.showerror("Lỗi", "Chưa chọn file!")
            return

        start = time.time()
        top_results = search_query(self.file_path, vocab, tfidf_data, inverted_index)
        duration = time.time() - start

        self.result_label.config(text=f"Thời gian tìm kiếm: {duration:.2f} giây")

        for i, lbl in enumerate(self.result_links):
            if i < len(top_results):
                doc_id = top_results[i][0]
                file_path = os.path.abspath("Text/" + metadata[doc_id]["filename"])
                lbl.config(text=file_path)
                lbl.file_path = file_path
            else:
                lbl.config(text="")
                lbl.file_path = None

    def open_file(self, event):
        file_path = event.widget.file_path
        if file_path and os.path.exists(file_path):
            webbrowser.open(file_path)
        else:
            messagebox.showerror("Lỗi", "Không thể mở file.")

if __name__ == "__main__":
    root = tk.Tk()
    app = SearchApp(root)
    root.mainloop()
