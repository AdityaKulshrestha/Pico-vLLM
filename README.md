# Pico-vLLM

This repository is created to understand the vLLM inference engine. It has been divived into easy understandable commits for the user to follow and understand the code base of their own. 

## Installation
1. sudo apt-get install pkg-config


## Structure
```
├── picovllm/                 
│   ├── __init__.py           
│   └── engine        # LLM Engine

```

## Components



# Setup
1. Clone the repository
2. Create the env and install the dependencies
    `uv sync`
3. Install torch
    `uv pip install torch==2.6.0 --torch-backend=cpu`
    `uv pip install intel-extension-for-pytorch==2.6.0`
4. Activate the environment
    `source .venv\bin\activate`
5. Download the model
    `hf download  Qwen/Qwen3-0.6B   --local-dir ~/huggingface/Qwen3-0.6B/`
6. Run the main.py 
    `python main.py`


## References 
1. https://www.aleksagordic.com/blog/vllm#cpt2
2. https://code2tutorial.com/tutorial/6249f206-9aa2-400f-a854-c19e7e335491/index.md
3. https://d2l.ai/chapter_computational-performance/hybridize.html


## AMX Resources
1. https://gemini.google.com/share/67d7baa13934
2. https://github.com/copilot/c/bd58e188-46a5-49b2-9abd-ea90e27be460



## Additional Resources
1. https://flashinfer.ai/
2. https://zhuanlan.zhihu.com/p/17186885141
3. https://www.youtube.com/watch?v=UUIKnca31Ao&list=PL_lsbAsL_o2DsybRNydPRukT4LLkl2buy&index=3


## Step by step learning

1. Learn about torch distributed APIs
    - https://docs.pytorch.org/tutorials/beginner/dist_overview.html
    


### Torch Compile Resources
1. https://pytorch.org/blog/torch-compile-everything/
3. https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/
4. https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/
5. https://youtu.be/zn0Pm2Pv3O0?si=HYs5Ia7BTm20fS7d
6. https://youtu.be/CVVbFlnP0m0?si=uPvgfziJxe9BUpqw