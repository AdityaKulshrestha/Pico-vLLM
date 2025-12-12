import time
from model import SmolLM, SmolLMConfig
from safetensors.torch import load_file
from transformers import AutoTokenizer

def main():
    config = SmolLMConfig(
        seq_len=2048,
        d_model=576,
        n_heads=9,
        n_kv_heads=3,
        d_ff=1536,
        vocab_size=49152,
        num_layers=30,
        eps=1e-05,
    )

    # Load safetensors and load it into the model
    tokenizer = AutoTokenizer.from_pretrained("smollm")
    model = SmolLM(config)
    state_dict = load_file("smollm/model.safetensors")

    # Remove the model prefix from the state dict keys
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    # To load the model, the model state dict and loaded should match
    # Leaving the strict as False for the last layer for weight tie
    model.load_state_dict(state_dict, strict=False)


    # Generate text
    text = "Once upon a time"
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    print("Token Shape: ", input_ids.shape)
    print("*"*30)

    # Without KV cache
    print("\n")
    outputs = model.generate(input_ids, max_new_tokens=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    # With KV cache
    print("\nWith KV Cache\n")
    outputs = model.generate(input_ids, max_new_tokens=50, use_cache=True)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))




if __name__ == "__main__":
    main()