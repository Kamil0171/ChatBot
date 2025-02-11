from flask import Flask, render_template, request, jsonify
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

app = Flask(__name__)

llm = OllamaLLM(model="llama3", base_url="http://localhost:11434")

chat_prompt = ChatPromptTemplate(
    messages=[HumanMessagePromptTemplate.from_template("User: {input}\nBot:")],
    input_variables=["input"]
)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    if request.is_json:
        data = request.get_json()
        user_message = data.get("message", "")
    else:
        user_message = request.form.get("message", "")

    prompt = chat_prompt.format(input=user_message)

    bot_response = llm(prompt)

    return jsonify({"response": bot_response})


if __name__ == "__main__":
    app.run(debug=True)
