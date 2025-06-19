
import os
import shutil
import tempfile
import uuid

import gradio as gr
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import FARMReader, TransformersReader, PreProcessor, TextConverter, PDFToTextConverter, DocxToTextConverter, FileTypeClassifier
from haystack.nodes import BM25Retriever
from haystack.pipelines import ExtractiveQAPipeline

# Stockage temporaire des sessions utilisateurs
SESSIONS_DIR = os.path.join(tempfile.gettempdir(), "haystack_user_sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

# Initialisation des composants Haystack
document_store = InMemoryDocumentStore()
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever)

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="word",
    split_length=200,
    split_respect_sentence_boundary=True,
)

converters = {
    ".pdf": PDFToTextConverter(remove_numeric_tables=True),
    ".txt": TextConverter(),
    ".docx": DocxToTextConverter(),
}

def get_user_dir(user_id):
    user_dir = os.path.join(SESSIONS_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def add_files(user_id, files):
    user_dir = get_user_dir(user_id)
    for file in files:
        dest_path = os.path.join(user_dir, os.path.basename(file.name))
        with open(dest_path, "wb") as f:
            f.write(file.read())
    return f"{len(files)} fichier(s) ajouté(s)."

def delete_files(user_id):
    user_dir = get_user_dir(user_id)
    shutil.rmtree(user_dir)
    os.makedirs(user_dir)
    return "Tous les fichiers ont été supprimés."

def run_search(user_id, query):
    user_dir = get_user_dir(user_id)
    all_docs = []

    for root, _, files in os.walk(user_dir):
        for file in files:
            filepath = os.path.join(root, file)
            ext = os.path.splitext(file)[-1].lower()
            converter = converters.get(ext)
            if converter:
                doc = converter.convert(file_path=filepath, meta={"name": file, "path": filepath})
                docs = preprocessor.process([doc])
                all_docs.extend(docs)

    document_store.delete_documents()
    document_store.write_documents(all_docs)

    prediction = pipeline.run(query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})
    answers = prediction["answers"]

    results = []
    for ans in answers:
        context = ans.context.strip().replace("\n", " ")
        result = f"**Extrait :** {context}\n\n**Fichier :** {ans.meta.get('name')}\n**Chemin :** {ans.meta.get('path')}"
        results.append(result)

    return "\n---\n".join(results) if results else "Aucun passage trouvé."

with gr.Blocks() as demo:
    gr.Markdown("# 🔍 Recherche intelligente avec Haystack")
    user_id = str(uuid.uuid4())

    with gr.Row():
        upload = gr.File(file_types=[".pdf", ".txt", ".docx"], file_count="multiple", label="Déposez vos fichiers")
        upload_button = gr.Button("Ajouter les fichiers")
        delete_button = gr.Button("Supprimer tous les fichiers")

    with gr.Row():
        query = gr.Textbox(label="Mot-clé ou question", placeholder="Entrez un mot-clé…")
        search_button = gr.Button("Rechercher")

    output = gr.Markdown()

    upload_button.click(fn=add_files, inputs=[gr.State(user_id), upload], outputs=output)
    delete_button.click(fn=delete_files, inputs=gr.State(user_id), outputs=output)
    search_button.click(fn=run_search, inputs=[gr.State(user_id), query], outputs=output)

if __name__ == "__main__":
    demo.launch()
