import os
import requests
from flask import Flask, request, jsonify
import re

# 환경 변수 로드
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AI_SEARCH_ENDPOINT = os.getenv("AI_SEARCH_ENDPOINT")
AI_SEARCH_SEMANTIC = os.getenv("AI_SEARCH_SEMANTIC")
AI_SEARCH_KEY = os.getenv("AI_SEARCH_KEY")
AI_SEARCH_INDEX = os.getenv("AI_SEARCH_INDEX")
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL")

# Flask 앱 생성
app = Flask(__name__)

# 정규식을 활용한 환자 정보 비식별화
def anonymize_text(user_input):
    warning_message = None
    if re.search(r'\b\d{6}[-]?\d{7}\b', user_input):
        user_input = re.sub(r'\b(\d{6})[-]?\d{7}\b', r'\1-*******', user_input)
        warning_message = "⚠️ 개인정보 보호를 위해 일부 정보가 비식별화되었습니다."
    if re.search(r'\b01[016789][-]?\d{3,4}[-]?\d{4}\b', user_input):
        user_input = re.sub(r'\b(01[016789])[-]?\d{3,4}[-]?\d{4}\b', r'\1-****-****', user_input)
        warning_message = "⚠️ 개인정보 보호를 위해 일부 정보가 비식별화되었습니다."
    if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', user_input):
        user_input = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '***@***.com', user_input)
        warning_message = "⚠️ 개인정보 보호를 위해 일부 정보가 비식별화되었습니다."
    return user_input, warning_message

# 질문 정제 함수
def refine_query(user_input):
    medical_keywords = ["진료기록", "의료법", "법적", "효력", "합병증", "처방전", "회복", "기간", "진료", "통증", "주의사항", "건강보험"]
    remove_patterns = [
        r"(이란\?|무엇인가요\?|의 목적이 무엇인가요\?|이 궁금합니다\?|무엇이죠\?|뭐야\?|무엇이에요\?)",
        r"(을|를|의|에 대한|에 대해|에 관하여|에 대하여|을 알고 싶어요)",
        r"(어떤가요\?|어떻게 하나요\?|알려주세요\?|정의는\?|목적은\?)"
    ]
    refined_query = user_input.strip()
    for pattern in remove_patterns:
        refined_query = re.sub(pattern, "", refined_query)
    refined_query_tokens = refined_query.split()
    return " ".join([word for word in refined_query_tokens if word in medical_keywords or len(word) > 2])

# 과거 대화 기록 저장 함수
def get_history_messages(histories):
    history_list = []
    history_length = 5
    for history in histories[:history_length]:
        history_list.append({"role": "user", "content": history[0]})
        history_list.append({"role": "assistant", "content": history[1]})
    return history_list

# RAG 기반 문서 검색 (Azure Search 사용)
def search_rag(query):
    search_payload = {
        "messages": [{"role": "user", "content": query}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 800,
        "data_sources": [
            {
                "type": "azure_search",
                "parameters": {
                    "endpoint": AI_SEARCH_ENDPOINT,
                    "semantic_configuration": AI_SEARCH_SEMANTIC,
                    "query_type": "semantic",
                    "strictness": 1,
                    "top_n_documents": 10,
                    "key": AI_SEARCH_KEY,
                    "indexName": AI_SEARCH_INDEX
                }
            }
        ]
    }
    response = requests.post(AZURE_OPENAI_ENDPOINT, headers={"Content-Type": "application/json", "api-key": AZURE_OPENAI_API_KEY}, json=search_payload)
    return response.json() if response.status_code == 200 else None

# Fine-tuned GPT-4o 호출
def request_gpt(prompt, history_list, search_results):
    headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_API_KEY}
    retrieved_docs = "검색된 문서가 없습니다."
    doc_map = {}
    if search_results and "value" in search_results:
        retrieved_docs = "\n\n".join([
            f"[doc{idx+1}] {doc.get('title', '문서 제목 없음')}\n{doc.get('content', '문서 내용 없음')}"
            for idx, doc in enumerate(search_results["value"])
        ])
        doc_map = {f"doc{idx+1}": doc for idx, doc in enumerate(search_results["value"])}
    
    message_list = [
        {"role": "system", "content": "당신은 환자를 위한 의료 챗봇입니다. 사용자의 질문에 대해 RAG 검색된 문서를 반드시 활용하여 답변을 생성하세요."},
        {"role": "system", "content": f"🔎 다음은 검색된 문서들입니다. \n{retrieved_docs}\n\n사용자 질문: {prompt}"}
    ]
    message_list.extend(history_list)
    message_list.append({"role": "user", "content": f"{prompt}\n\n친절한 의사 말투로 설명해 주세요."})
    
    payload = {"messages": message_list, "model": FINE_TUNED_MODEL, "temperature": 0, "top_p": 0.6, "max_tokens": 600}
    response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=payload)
    return response.json() if response.status_code == 200 else None

# Flask 엔드포인트 정의
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        histories = data.get("histories", [])
        
        anonymized_prompt, warning_message = anonymize_text(prompt)
        refined_prompt = refine_query(anonymized_prompt)
        history_list = get_history_messages(histories)
        search_results = search_rag(refined_prompt)
        response_json = request_gpt(refined_prompt, history_list, search_results)
        
        if response_json:
            response_text = response_json["choices"][0]["message"]["content"]
        else:
            response_text = "응답을 생성할 수 없습니다."
        
        histories.append((prompt, response_text))
        return jsonify({"histories": histories, "response": response_text, "citations": ""})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)