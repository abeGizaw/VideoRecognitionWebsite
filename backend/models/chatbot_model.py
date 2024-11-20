
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration

def large_chatbot_model(result):
    # When actually using this model, the model and tokenizer should be loaded once and reused for all requests (outside of the function)   
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl").to("cuda")

    prompt = f"""
    You are part of a website where a user can record themselves doing some action, and a separate model classifies the action.
    The model passes the top 5 in this format: "Most confident: 'category' I think it is at least one of these 5: *lists 5 categories with confidence*."
    Send a response appropriate to the most confident category. For example:
    - If they wave, wave back.
    - If they are playing a game, say you love the game.
    - If unsure, state some fun facts.
    Here is the result: {result}
    """
    simplified_prompt = f"The user is most likely {result}. Respond appropriately."

    test = "Respone appropriately. fixing hair"

    inputs = tokenizer(simplified_prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
    output = model.generate(
        **inputs,
        max_length=200,
        do_sample=True,
        temperature=0.7,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


def small_chatbot_model(result):
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", device_map="auto")

    # Ensure the tensors are on the same device as the model
    device = next(model.parameters()).device

    simplified_prompt = f"The user is most likely {result}. Give a fun fact relating to {result}"
    # test = f"Respond appropriately. {result}"

    # Move the input tensors to the same device as the model
    input_ids = tokenizer(simplified_prompt, return_tensors="pt").input_ids.to(device)

    # Generate response
    outputs = model.generate(input_ids)

    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

