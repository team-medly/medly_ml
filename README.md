# 🏥 PatientCareGPT - PatientCare Chatbot API Documentation

## **1. 개요**

🏥 PatientCareGPT는 환자의 의료 관련 질문에 대해 친절하고 이해하기 쉬운 답변을 제공하는 챗봇 API입니다.
이 API는 **Azure OpenAI Service**의 Fine-tuned GPT-4o 모델과 **Azure Cognitive Search 기반의 RAG(Retrieval-Augmented Generation) 시스템**을 결합하여, 신뢰할 수 있는 의료 정보를 제공합니다.

---

## **2. Chat API (GPT 기반 대화 엔드포인트)**

### **URL**: `/chat`

### **Method**: `POST`

### **Request Body (JSON)**:

```json
{
  "prompt": "환자 기록 열람이 가능한 대상은 누구인가요?",
  "histories": [
    [
      "수술 기록은 보호자가 볼 수 있나요?",
      "환자의 동의가 필요한 경우가 많습니다."
    ],
    [
      "의료법에 따르면 진료기록을 열람할 수 있는 대상은?",
      "의료법에 따라 환자 본인과 일부 지정된 관계자만 열람 가능합니다."
    ]
  ]
}
```
```json
{
  "prompt": "복강경담낭절제술의 목적이 무엇인가요?",
  "histories": [
    [
      "담낭 제거 수술은 언제 필요한가요?",
      "담낭에 염증이나 담석이 심한 경우 필요할 수 있습니다."
    ],
    [
      "수술 후 회복 기간은 얼마나 걸리나요?",
      "대부분의 경우 몇 주 이내에 일상 생활로 복귀할 수 있습니다."
    ]
  ]
}
```
- `prompt` (string): 사용자가 입력한 질문
- `histories` (array of [string, string], optional): 이전 대화 기록

### **Response (JSON)**:

```json
{
  "histories": [
    [
      "수술 기록은 보호자가 볼 수 있나요?",
      "환자의 동의가 필요한 경우가 많습니다."
    ],
    [
      "의료법에 따르면 진료기록을 열람할 수 있는 대상은?",
      "의료법에 따라 환자 본인과 일부 지정된 관계자만 열람 가능합니다."
    ],
    [
      "환자 기록 열람이 가능한 대상은 누구인가요?",
      "환자 본인, 친족, 의료기관 등이 법령에 따라 열람할 수 있습니다."
    ]
  ],
  "response": "환자 본인, 친족, 의료기관 등이 법령에 따라 열람할 수 있습니다.",
  "citations": ""
}
```
```json
{
  "histories": [
    [
      "담낭 제거 수술은 언제 필요한가요?",
      "담낭에 염증이나 담석이 심한 경우 필요할 수 있습니다."
    ],
    [
      "수술 후 회복 기간은 얼마나 걸리나요?",
      "대부분의 경우 몇 주 이내에 일상 생활로 복귀할 수 있습니다."
    ],
    [
      "복강경담낭절제술의 목적이 무엇인가요?",
      "복강경담낭절제술은 담낭(쓸개)을 제거하는 수술입니다. 이 수술은 담석증, 담낭염, 용종 등의 질환을 치료하기 위해 시행됩니다."
    ]
  ],
  "response": "복강경담낭절제술은 담낭(쓸개)을 제거하는 수술입니다. 이 수술은 담석증, 담낭염, 용종 등의 질환을 치료하기 위해 시행됩니다.",
  "citations": ""
}
```

---

## **3. 기능 개요 (상세 설명)**

### ✅ **환자 정보 비식별화 (`anonymize_text` 함수)**

- 사용자의 입력에서 개인정보(주민등록번호, 전화번호, 이메일)를 자동으로 감지하여 마스킹 처리합니다.
    - **Input**: `user_input` (string) - 사용자가 입력한 질문
    - **Output**: 비식별화된 `user_input` (string), `warning_message` (string, 선택적 경고 메시지)

### ✅ **질문 정제 (`refine_query` 함수)**

- 질문에서 불필요한 표현을 제거하고 핵심 의료 키워드만 남깁니다.
    - **Input**: `user_input` (string) - 사용자의 질문
    - **Output**: 정제된 질문 `refined_query` (string)

### ✅ **과거 대화 기록 저장 (`get_history_messages` 함수)**

- 최근 5개의 대화 기록을 유지하여 문맥을 고려한 응답을 생성합니다.
    - **Input**: `histories` (list of tuples: [(user_query, bot_response)])
    - **Output**: 최근 5개의 대화 목록 `history_list` (list of JSON objects)

### ✅ **RAG 기반 문서 검색 (`search_rag` 함수)**

- Azure Cognitive Search를 활용하여 사용자의 질문과 연관된 문서를 검색하고, 챗봇 응답을 형성하는 데 활용합니다.
    - **Input**: `query` (string) - 정제된 사용자 질문
    - **Output**: 검색된 문서 목록 `search_results` (JSON)

### ✅ **Fine-tuned GPT-4o 호출 (`request_gpt` 함수)**

- Azure OpenAI Service의 Fine-tuned GPT-4o 모델을 사용하여 자연어 응답을 생성합니다.
  - **검색된 문서(RAG)를 기반으로 GPT가 응답을 형성**하여 정확도를 높입니다.
  - 환자 친화적인 답변을 제공하기 위해 시스템 메시지를 활용하여 설명을 더욱 부드럽고 이해하기 쉽게 만듭니다.
- **Input**:
  - `prompt` (string): 사용자 질문
  - `history_list` (list): 최근 대화 기록
  - `search_results` (JSON): 검색된 의료 문서
- **Output**:
  - `response` (string): 환자 친화적인 의사 말투로 생성된 응답
  - `doc_map` (dict): 참조된 문서 목록

---

## **4. 환경 변수**

API 실행을 위해 아래의 환경 변수를 설정해야 합니다:

- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI 서비스 엔드포인트
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API 키
- `AI_SEARCH_ENDPOINT`: Azure Cognitive Search 엔드포인트
- `AI_SEARCH_SEMANTIC`: Azure Search의 시맨틱 검색 설정
- `AI_SEARCH_KEY`: Azure Search API 키
- `AI_SEARCH_INDEX`: 검색할 인덱스명
- `FINE_TUNED_MODEL`: Fine-tuned GPT-4o 모델 ID

---

## **5. 실행 방법**

```bash
export FLASK_APP=app.py
export PORT=8080
flask run --host=0.0.0.0 --port=$PORT
```

또는

```bash
python app.py
```
