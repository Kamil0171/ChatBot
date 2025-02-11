from flask import Flask, render_template, request, jsonify
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# Inicjalizacja aplikacji Flask
app = Flask(__name__)

# Inicjalizacja modelu Llama3 przy użyciu OllamaLLM.
# Upewnij się, że serwer Ollama jest uruchomiony na wskazanym adresie.
llm = OllamaLLM(model="llama3", base_url="http://localhost:11434")

# Konfiguracja szablonu prompta do komunikacji czatu.
# Używamy szablonu, gdzie wiadomość użytkownika wstawiana jest w miejsce {input}.
chat_prompt = ChatPromptTemplate(
    messages=[HumanMessagePromptTemplate.from_template("User: {input}\nBot:")],
    input_variables=["input"]
)


# Główna strona aplikacji - zwraca szablon index.html
@app.route('/')
def index():
    return render_template('index.html')


# Endpoint do obsługi zapytań czatu metodą POST
@app.route('/chat', methods=['POST'])
def chat():
    # Sprawdzenie, czy dane zostały przesłane w formacie JSON
    if request.is_json:
        data = request.get_json()
        user_message = data.get("message", "")
    else:
        # Obsługa przesyłania danych przez form-data
        user_message = request.form.get("message", "")

    # Formatowanie prompta z wiadomością użytkownika
    prompt = chat_prompt.format(input=user_message)

    # Wysłanie prompta do modelu i otrzymanie odpowiedzi
    bot_response = llm(prompt)

    # Zwracanie odpowiedzi w formacie JSON
    return jsonify({"response": bot_response})


# Uruchomienie aplikacji Flask w trybie debugowania
if __name__ == "__main__":
    app.run(debug=True)
