"""






"""
import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp


class LLMEngine:
    def __init__(self, model, **kwargs):

        # Setup configs
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.ps = []
        self.events = []

        ctx = mp.get_context("spawn")           # Create context for handling tensor parallelism. Creates multiple event loop for each GPU for TP.
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()            # Creates event loop for each tensor parallelism   
            process = ctx.Process(target=ModelRunner, args=(config, i, event))          # ModelRunner is the actual executeable class which executes the input requests on hardware
            process.start()
            self.ps.append(process)     # For tracking and IPC (inter process) synchronization, later termination
            self.events.append(event)   # For signalling between the main and worker process

        self.model_runner = ModelRunner(config, 0, self.events)         ## TODO:    ; Instantiates the ModelRunner for main process
        
        # Configuring tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fase=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)          # Responsible for scheduling the input requests and preprocessing
        atexit.register(self.exit)                  # Registers the self.exit method to be called automatically when the program exits. Ensure proper cleanup of resources and subprocesses; refer to the exit method in the class.

    ## TODO: ModelRunner and Scheduler

    def exit(self):
        """
        Exits the main and children process
        """
        self.model_runner.call("exit")      # Triggers the model_runner to terminate and exit
        del self.model_runner
        for p in self.ps:
            p.join()                        # Main program pauses here until the process finishes


    def add_request(self, ):

        pass


    def step(self, ):

        pass


    def generate(self, ):


        pass  