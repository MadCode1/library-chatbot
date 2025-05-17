from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests
from bs4 import BeautifulSoup
import urllib.parse

app = Flask(__name__)

# Load DialoGPT model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Internal library data

book_data = {
    # Custom / Team data
    "austin": "He is the one who coded me bruh.",
    "team": "Austin Masila, Mansoor S, Varshita, Dheekshitha",

    # General Fiction / Self-help
    "the silent patient": "The Silent Patient is a psychological thriller by Alex Michaelides about a woman who stops speaking after committing a shocking act of violence.",
    "it ends with us": "It Ends With Us is a romance novel by Colleen Hoover that explores the complexities of love and domestic abuse.",
    "project hail mary": "Project Hail Mary is a science fiction novel by Andy Weir about a man who wakes up alone on a spaceship with the mission to save humanity.",
    "verity": "Verity is a psychological thriller by Colleen Hoover involving a ghostwriter uncovering disturbing secrets.",
    "psychology of money": "In The Psychology of Money, Morgan Housel shares stories about how people think about money.",

    # Engineering & CS Books
    "introduction to algorithms": "Also known as CLRS, this book is a comprehensive guide to algorithms and data structures, widely used in universities worldwide.",
    "clean code": "Clean Code by Robert C. Martin teaches the principles of writing readable, maintainable, and efficient software.",
    "design patterns": "Design Patterns: Elements of Reusable Object-Oriented Software is a foundational book that introduces 23 classic software design patterns.",
    "structure and interpretation of computer programs": "SICP is a classic computer science textbook that focuses on the principles of computation using Scheme.",
    "computer networks": "Computer Networks by Andrew S. Tanenbaum is a foundational book that covers the basics of networking from physical to application layers.",
    "artificial intelligence: a modern approach": "Often referred to as AIMA, this book by Russell and Norvig is the most widely used AI textbook.",
    "deep learning": "Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville covers the theory and practice of neural networks and AI systems.",
    "hands-on machine learning": "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron teaches ML with practical coding projects.",
    "the elements of statistical learning": "This book provides an in-depth introduction to statistical modeling and machine learning techniques.",
    "pattern recognition and machine learning": "Christopher Bishop's textbook focuses on probabilistic models for machine learning and pattern recognition.",
    "introduction to the theory of computation": "Michael Sipser's book introduces theoretical computer science, including automata, Turing machines, and complexity.",
    "modern operating systems": "Written by Andrew Tanenbaum, this book covers OS concepts such as memory management, processes, threads, and security.",
    "computer organization and design": "This book by David A. Patterson and John L. Hennessy explains how computer hardware and software interact.",
    "introduction to linear algebra": "Gilbert Strang's popular book explains linear algebra, matrices, and vector spaces with applications in engineering.",
    "numerical methods for engineers": "A book by Chapra and Canale that teaches how to use numerical techniques in engineering problem-solving.",
    "signal processing and linear systems": "This book by B.P. Lathi covers the fundamentals of analog and digital signal processing.",
    "control systems engineering": "A widely-used book by Norman S. Nise that covers system modeling, feedback, and stability in engineering.",
    "data structures and algorithms in python": "This book explains basic to advanced data structures and algorithms using Python code examples.",
    "the pragmatic programmer": "This is a book of software development best practices that help engineers write better, more maintainable code.",
    "introduction to machine learning with python": "A beginner-friendly book by Andreas Müller and Sarah Guido on using Scikit-learn to build ML models.",
}

# Search book info from OpenLibrary
def get_openlibrary_book_data(query):
    try:
        encoded_query = urllib.parse.quote(query)
        search_url = f"https://openlibrary.org/search.json?q={encoded_query}"
        resp = requests.get(search_url, timeout=5)
        data = resp.json()

        if data.get('docs'):
            book = data['docs'][0]
            title = book.get('title', 'N/A')
            author = ', '.join(book.get('author_name', [])) or "Unknown"
            publish_year = book.get('first_publish_year', 'N/A')
            return f"📘 Title: {title}\n✍️ Author(s): {author}\n📅 First Published: {publish_year}"
    except Exception as e:
        print(f"Error with OpenLibrary: {e}")
    return None

# Web search fallback using DuckDuckGo
def search_duckduckgo(query, max_results=3):
    try:
        search_url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(search_url, headers=headers, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.find_all("a", class_="result__a", limit=max_results)

        snippets = []
        for r in results:
            href = r.get("href")
            if href:
                try:
                    page_resp = requests.get(href, headers=headers, timeout=5)
                    page_soup = BeautifulSoup(page_resp.text, "html.parser")
                    paragraphs = page_soup.find_all("p")
                    text = ' '.join(p.get_text() for p in paragraphs[:2])
                    if text:
                        snippets.append(text)
                except Exception:
                    continue
        return " ".join(snippets) if snippets else None
    except Exception:
        return None

# DialoGPT fallback
def get_dialo_response(user_input):
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        do_sample=True
    )
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Master response logic
def get_response(user_input):
    lower_input = user_input.lower()

    # 1. Internal library data
    for key in book_data:
        if key in lower_input:
            return book_data[key]

    # 2. OpenLibrary dynamic lookup
    openlib_info = get_openlibrary_book_data(user_input)
    if openlib_info:
        return openlib_info

    # 3. Web search fallback
    web_text = search_duckduckgo(user_input)
    if web_text:
        return web_text

    # 4. DialoGPT as final fallback
    return get_dialo_response(user_input)

# Web interface
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form.get("user_input")
        response = get_response(user_input)
        return render_template("index.html", user_input=user_input, response=response)
    return render_template("index.html")

# API endpoint
@app.route("/api/ask", methods=["POST"])
def ask_api():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Invalid request, 'message' key is required."}), 400
    user_input = data['message']
    response = get_response(user_input)
    return jsonify({"response": response})

# Run app
if __name__ == "__main__":
    app.run(debug=True)
