import json
import re
import time
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI(api_key="myapikey")

MAX_MESSAGES = 50
MAX_CONTEXT_LENGTH = 1024

def preprocess_text(text):
    return re.sub(r'\s+', ' ', text.strip())

def manage_conversation(messages):
    if len(messages) > MAX_MESSAGES:
        archived_messages = messages[:-MAX_MESSAGES]
        save_conversation(archived_messages, filename="conversation_archive.json")
        return messages[-MAX_MESSAGES:]
    return messages

def save_conversation(messages, filename="conversation_history_3.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def load_conversation():
    try:
        with open('conversation_history_3.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return [{"role": "system", "content": "You are a helpful assistant."}]

def retrieve_relevant_info(query, messages, k=3):
    vectorizer = TfidfVectorizer()
    corpus = [preprocess_text(m['content']) for m in messages if m['role'] != 'system']
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([preprocess_text(query)])
    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [messages[i]['content'] for i in top_k_indices]

def limit_context_length(context):
    if len(context) > MAX_CONTEXT_LENGTH:
        return context[:MAX_CONTEXT_LENGTH]
    return context

def evaluate_response(user_input, ai_response, relevant_context):
    metrics = {}

    metrics['faithfulness'] = 1 if ai_response in relevant_context else 0

    metrics['answer_relevancy'] = cosine_similarity(
        TfidfVectorizer().fit_transform([user_input, ai_response])
    )[0, 1]

    precision_terms = len(set(relevant_context.split()) & set(ai_response.split()))
    total_terms = len(set(relevant_context.split()))
    metrics['context_precision'] = precision_terms / total_terms if total_terms > 0 else 0

    recall_terms = len(set(relevant_context.split()) & set(ai_response.split()))
    total_response_terms = len(set(ai_response.split()))
    metrics['context_recall'] = recall_terms / total_response_terms if total_response_terms > 0 else 0

    return metrics

def openai_request_with_retry(request_func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return request_func()
        except Exception as e:
            print(f"Retrying ({attempt + 1}/{max_retries}) due to error: {e}")
            time.sleep(2)
    raise Exception("OpenAI request failed after retries.")

messages = load_conversation()

print("AI 어시스턴트와 대화를 시작합니다. 종료하려면 'quit'을 입력하세요.")

while True:
    user_input = input("You: ")

    if user_input.lower() == 'quit':
        print("대화를 종료합니다.")
        save_conversation(messages)

        user_response = {"conversation": messages}
        json_data = json.dumps(user_response)

        response = openai_request_with_retry(lambda: client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "사용자의 말 중 유용한 정보만 텍스트 추출하여 올바른 정보 문서로 작성하시오."},
                {"role": "user", "content": json_data}
            ]
        ))

        summarized_document = response.choices[0].message.content
        with open('summarized_document_3.txt', 'w', encoding='utf-8') as file:
            file.write(summarized_document)

        print("대화 내용이 요약되어 'summarized_document_3.txt' 파일로 저장되었습니다.")
        break

    messages.append({"role": "user", "content": user_input})
    messages = manage_conversation(messages)

    relevant_info = retrieve_relevant_info(user_input, messages)
    context = limit_context_length("\n".join(relevant_info))

    response = openai_request_with_retry(lambda: client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use the following context to answer the user's question."},
            {"role": "assistant", "content": context},
            {"role": "user", "content": user_input}
        ]
    ))

    ai_response = response.choices[0].message.content
    print("AI:", ai_response)

    
    metrics = evaluate_response(user_input, ai_response, context)
    print("Evaluation Metrics:", metrics)

    messages.append({"role": "assistant", "content": ai_response})
    save_conversation(messages)
