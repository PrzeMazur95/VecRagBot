import os
from flask import Flask, render_template, request, redirect, flash, url_for, session
from app.service.open_ai_service import OpenAiService
from app.service.file_service import FileService
from app.models.qdrant_connector import QdrantConnection
from app.main import Retriever

app = Flask(__name__)
app.secret_key = '_5#y2L"F4Q8z\n\xec]/'

FILE_SERVICE = FileService()
RETRIEVER = Retriever()
QDRANT_CONNECTOR = QdrantConnection()

@app.route("/index")
def index():
    print(request.cookies.get('session'))
    return render_template("index.html")

@app.route("/key_check", methods=['POST'])
def key_check():
    if request.form.get('username'):
        session['username'] = request.form.get('username')
    if request.form.get('api_key'):
        is_key_valid = OpenAiService.check_if_api_key_is_valid(request.form.get('api_key'))
        if not is_key_valid:
            flash("Invalid API key!", 'error')
            return render_template("key_check.html", is_valid=is_key_valid)

        else:
            flash("Api key valid!", 'success')
            session[session['username'] + 'api_key']=request.form.get('api_key')
            return redirect("/upload")
    else:
        return render_template("key_check.html")

@app.route("/upload", methods=['POST', 'GET'])
def upload():
    if len(request.files) == 0:
        return render_template('/choose_file.html')
    else:
        file = request.files['file']
        if file and FILE_SERVICE.allowed_file_extension(file.filename):
            if FILE_SERVICE.save_file(file):
                collection_name = session['username'] + FILE_SERVICE.get_base_file_name(file.filename)
                #this is where specific user collection name will be stored, like tom_collection_name
                session[session['username'] + "_collection_name"] = collection_name
                QDRANT_CONNECTOR.run(collection_name = collection_name, file_name=file.filename)
                flash("File upladed succesfully!", 'success')
                return redirect(url_for("chat", show_chat=True, filename=file.filename))
            else:
                flash("File has not been saved, try again!")
                return redirect("/upload")
        else:
            flash("Invalid file type. Please upload a PDF or TXT file.", 'error')
            return render_template('/choose_file.html')
        

@app.route("/chat", methods=['POST', 'GET'])
def chat():
    show_chat = request.args.get('show_chat', 'false').lower() == 'true'
    filename = request.args.get('filename', 'false')
    return render_template("/chat.html", show_chat=show_chat, filename=filename)

@app.route("/process_query", methods=['POST', 'GET'])
def process_query():
    filename = request.form.get("filename")
    if not request.form.get("nextQuestion"):
        query = request.form.get("query")
        answer = RETRIEVER.main(filename=filename, query=query)
        return render_template("/chat.html", answer=answer, question_answered=True, filename=filename)
    else :
        chat_url = url_for('chat', filename=filename, show_chat=True)
        return redirect(chat_url)
        

if __name__ == "__main__":
    app.run(debug=True)