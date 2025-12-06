import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp


from picovllm.config import Config
from picovllm.engine.model_runner import ModelRunner
from picovllm.engine.scheduler import Scheduler
from picovllm.engine.sequence import Sequence
from picovllm.sampling_params import SamplingParams

class LLMEngine:
    def __init__(self, model, **kwargs):

        # Setup configs
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.ps = []
        self.events = []

        ctx = mp.get_context("spawn")                                                   # Create context for handling tensor parallelism. Creates multiple event loop for each GPU for TP.
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()                                                         # Creates event loop for each tensor parallelism   
            process = ctx.Process(target=ModelRunner, args=(config, i, event))          # ModelRunner is the actual executeable class which executes the input requests on hardware
            process.start()
            self.ps.append(process)                                                     # For tracking and IPC (inter process) synchronization, later termination
            self.events.append(event)                                                   # For signalling between the main and worker process

        self.model_runner = ModelRunner(config, 0, self.events)                         ## TODO: Instantiates the ModelRunner for main process
        
        # Configuring tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fase=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)                                              # Responsible for scheduling the input requests and preprocessing
        atexit.register(self.exit)                                                      # Registers the self.exit method to be called automatically when the program exits. Ensure proper cleanup of resources and subprocesses; refer to the exit method in the class.

    ## TODO: ModelRunner and Scheduler

    def exit(self):
        """
        Exits the main and children process
        """
        self.model_runner.call("exit")      # Triggers the model_runner to terminate and exit
        del self.model_runner
        for p in self.ps:
            p.join()                        # Main program pauses here until the process finishes


    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)


    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True, 
    ) -> list[str]:
        if use_tqdm: 
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}

        prefill_throughput = decode_throughput = 0.

        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()

            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s"
                })

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]

        if use_tqdm:
            pbar.close()
        return outputs