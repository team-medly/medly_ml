import os
import requests
from lingua import LanguageDetectorBuilder
from deep_translator import GoogleTranslator
import re
from flask import Flask, request, jsonify  


# Load environment variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AI_SEARCH_ENDPOINT = os.getenv("AI_SEARCH_ENDPOINT")
AI_SEARCH_SEMANTIC = os.getenv("AI_SEARCH_SEMANTIC")
AI_SEARCH_KEY = os.getenv("AI_SEARCH_KEY")
AI_SEARCH_INDEX = os.getenv("AI_SEARCH_INDEX")

# Initialize language detector
detector = LanguageDetectorBuilder.from_all_languages().build()

# Flask 앱 생성
app = Flask(__name__)

# 최근 대화 히스토리를 추출하는 함수
def get_history_messages(histories):
    history_list = []
    history_length = 5  # 최근 5개의 메시지만 사용

    for i, history in enumerate(histories[:history_length]):
        history_list.append({"role": "assistant", "content": history[0]})
        history_list.append({"role": "assistant", "content": history[1]})

    return history_list

# GPT API 요청 함수
def request_gpt(prompt, history_list, detected_lan):
    headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_API_KEY}

    message_list = [
        {
            "role": "system",
            "content": f"You are an assistant for medical professionals. Always answer in {detected_lan}. If you cannot, switch to English. When asked about a disease, search the 'disease' section of the provided data. If found, answer based on the document."
        }
    ]

    message_list.extend(history_list)
    message_list.append({"role": "user", "content": prompt})

    payload = {
        "messages": message_list,
        "temperature": 0.1,
        "top_p": 0.6,
        "max_tokens": 800,
        "data_sources": [
            {
                "type": "azure_search",
                "parameters": {
                    "endpoint": AI_SEARCH_ENDPOINT,
                    "semantic_configuration": AI_SEARCH_SEMANTIC,
                    "query_type": "semantic",
                    "strictness": 5,
                    "top_n_documents": 5,
                    "key": AI_SEARCH_KEY,
                    "indexName": AI_SEARCH_INDEX
                }
            }
        ]
    }

    response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=payload)

    if response.status_code == 200:
        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]
        if content == "The requested information is not available in the retrieved data. Please try another query or topic.":
            return content, None
        # Check if there are any citations in the response
        if response_json["choices"][0]["message"]["context"]:
            citations = response_json["choices"][0]["message"]["context"]["citations"]
            formatted_citation_list = list()
            i = 0
            for c in citations:
                i += 1
                temp = f"<details><summary>Doc{i}</summary><ul>{c['content']}</ul></details>"
                formatted_citation_list.append(temp)
                
        else:
            formatted_citation_list = list() # No citations
            
        text = "".join(formatted_citation_list)
      

        # Extract chunk and disease values using regular expressions
        chunk_match = re.findall(r'"chunk"\s*:\s*"([^"]+)"', text)
        disease_match = re.findall(r'"disease"\s*:\s*"([^"]+)"', text)
        source_match = re.findall(r'"source"\s*:\s*"([^"]+)"', text)

        citation_text = []

        for idx, (chunk, disease, source) in enumerate(zip(chunk_match, disease_match, source_match), start=1):
            # Set it to expand when clicked
            citation_t = f"""
            <details>
                <summary>Doc{idx}</summary>
                <h3>Original Text</h3>
                <span>{chunk}</span> 
                <h3>Data Sources</h3>
                <span><b>disease</b>: {disease}, <b>source</b>: {source}</span>                
                </details>
                <br>
            """

            citation_text.append(citation_t)

        citation_html = "\n".join(citation_text)

        return content, citation_html


    else:
        return f"{response.status_code}, {response.text}", ""

# 언어 감지 함수
def detect_language(text):
    if re.search(r"[가-힣]", text):
        return "korean"
    detected_language = detector.detect_language_of(text)
    return detected_language.name.lower() if detected_language else "unknown"

# 영어로 번역하는 함수
def translate_to_english(text):
    try:
        translator = GoogleTranslator(source="auto", target="en")
        return translator.translate(text)
    except Exception:
        return text  # 번역 실패 시 원문 반환

# Flask 엔드포인트 정의
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        histories = data.get("histories", [])

        history_list = get_history_messages(histories)
        detected_lan = detect_language(prompt)
        trans_prompt = translate_to_english(prompt)

        response_text, citation_html = request_gpt(trans_prompt, history_list, detected_lan)
        histories.append((prompt, response_text))

        return jsonify({"histories": histories, "response": response_text, "citations": citation_html})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask 실행 (개발 환경)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080)) 
    app.run(host="0.0.0.0", port=port)
