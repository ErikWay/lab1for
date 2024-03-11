from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_response(prompt):
    model_name = "EleutherAI/gpt-neo-2.7B"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Генерация ответа
    output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Пример использования
user_prompt = "Какая будет погода завтра?"
bot_response = generate_response(user_prompt)
print(bot_response)
