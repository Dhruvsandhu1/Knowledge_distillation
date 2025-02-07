import os
import yaml
import torch
from datasets import load_dataset, IterableDataset, Dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, get_scheduler
from accelerate import Accelerator
from huggingface_hub import HfFolder, create_repo, upload_folder
import wandb
import time
import torch.nn.functional as F
from galore_torch import GaLoreAdamW8bit
import gc
from transformers import TrainerCallback
from itertools import islice
from huggingface_hub import login

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def setup_environment(config):
    os.environ['WANDB_PROJECT'] = config["wandb"]["wandb_project"]
    os.environ['WANDB_ENTITY'] = config["wandb"]["wandb_entity"]
    wandb.init(project=config["wandb"]["wandb_project"], entity=config["wandb"]["wandb_entity"])
    os.environ['WANDB_DISABLED'] = 'false'
    return Accelerator()

def load_and_preprocess_dataset(config, student_tokenizer):
    def tokenize_function(examples):
        return student_tokenizer(examples["text"], truncation=True, max_length=config["tokenizer"]["max_length"], padding="max_length")

    datasets = []
    for subset in config["dataset"]["subsets"]:
        # Load the dataset as an IterableDataset
        dataset = load_dataset(
            config["dataset"]["name"],
            subset['name'],
            split=subset['split'],
            streaming=True
        )
        
        # Keep only the 'text' column for all subsets
        if 'text' in dataset.column_names:
            dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'text'])
        else:
            raise ValueError(f"The 'text' column is missing in the {subset['name']} subset.")
        
        datasets.append(dataset)

    # Concatenate all datasets
    full_dataset = concatenate_datasets(datasets)

    # Create evaluation dataset (first N examples)
    eval_dataset = Dataset.from_list(list(islice(full_dataset, config["dataset"]["eval_samples"])))
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    # Create training dataset (skip first N examples)
    def generate_train_examples():
        for i, example in enumerate(full_dataset):
            if i >= config["dataset"]["eval_samples"]:
                yield example

    train_dataset = IterableDataset.from_generator(generate_train_examples)
    train_dataset = train_dataset.map(
        tokenize_function,
        remove_columns=train_dataset.column_names
    )

    return train_dataset, eval_dataset

def load_models_and_tokenizers(config):
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if config["model_config"]["use_flash_attention"]:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    print(f"model_kwargs: {model_kwargs}")

    teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"], add_eos_token=True)
    student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"], add_eos_token=True)

    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
        print(f"Setting pad_token to eos_token: {student_tokenizer.pad_token}")

    teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"], **model_kwargs)
    student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"], **model_kwargs)

    teacher_model.eval() # setting teacher model to evaluation mode

    return teacher_model, student_model, teacher_tokenizer, student_tokenizer

def pad_logits(student_logits, teacher_logits):
    #Ensure that the logits are padded to the same size
    #We know the logits for the teacher model are in the shape (4,1024,49152) ie. (batch_size, seq_len, vocab_size)
    student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
    if student_size != teacher_size:
        pad_size = abs(student_size - teacher_size)
        pad_tensor = torch.zeros((*teacher_logits.shape[:-1], pad_size), dtype=teacher_logits.dtype, device=teacher_logits.device)
        return (torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits) if student_size < teacher_size else (student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1))
    return student_logits, teacher_logits

class DistillationTrainer(SFTTrainer):

    def __init__(self, *args, **kwargs):
        self.config = kwargs.pop('config', None)
        self.teacher_model = kwargs.pop('teacher_model', None)
        super().__init__(*args, **kwargs)
        
        # Ensure teacher model is on the same device as the student model
        if self.teacher_model.device != self.model.device:
            self.teacher_model = self.teacher_model.to(self.model.device)
        
        # Ensure teacher model is in eval mode
        self.teacher_model.eval()

    #Overwriting the default loss function of the SFTTrainer Class
    def compute_loss(self, model, inputs, return_outputs=False):
        if hasattr(model, 'module'):
            device = model.module.device
        else:
            device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        student_outputs = model(**inputs)
        
        # Get teacher outputs (logits only for distillation)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        # Check if 'labels' are in the inputs, if not, use 'input_ids' as labels
        labels = inputs.get('labels', inputs.get('input_ids'))
        
        if labels is None:
            raise ValueError("Neither 'labels' nor 'input_ids' found in inputs. Cannot compute loss.")

        #Loss function is a combination of cross-entropy and KL divergence
        # Compute distillation loss (combination of cross-entropy and KL loss)
        custom_loss = self.distillation_loss(student_outputs, teacher_outputs, labels)
        
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, student_outputs, teacher_outputs, labels):
        # Get the logits for distillation
        student_logits = student_outputs.logits
        # print(student_logits.size())
        teacher_logits = teacher_outputs.logits
        # print(teacher_logits.size())
        
        # Ensure logits are padded if necessary
        student_logits, teacher_logits = pad_logits(student_logits, teacher_logits)
        
        #Using forward divergence loss in our case, You can try it with different losses as well
        kl_loss = self.forward_kl_divergence(student_logits, teacher_logits)

        # Check if alpha is set to allow cross-entropy loss (alpha != 1) ie if alpha==0 use only cross-entropy loss elif alpha==1 use only KL loss
        alpha = self.config["distillation"]["alpha"]
        
        if alpha != 1:
            # If the model has already computed cross-entropy loss, use that
            if 'loss' in student_outputs:
                original_loss = student_outputs['loss']
            else:
                # Shift the labels to the right for causal language modeling
                # Check whether it is done or not and then apply this 
                shifted_labels = labels[:, 1:].contiguous()  # Remove the first token
                shifted_logits = student_logits[:, :-1, :].contiguous()  # Ignore the last token for logits

                # Compute cross-entropy loss manually with shifted labels
                original_loss = F.cross_entropy(
                    shifted_logits.view(-1, shifted_logits.size(-1)), 
                    shifted_labels.view(-1), 
                    ignore_index=-100
                )
        else:
            # Redundant value, but required for the combined loss
            original_loss = 0

        # Combine the distillation loss (KL loss) and cross-entropy loss
        combined_loss = alpha * kl_loss + (1 - alpha) * original_loss
        
        return combined_loss

    def forward_kl_divergence(self, student_logits, teacher_logits):
        temperature = self.config["distillation"]["temperature"]

        # Compute log-softmax for both student and teacher logits
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)

        # Compute the KL divergence between the student and teacher logits
        kl_div = F.kl_div(student_log_probs, teacher_log_probs, reduction='batchmean', log_target=True)

        # Scale the KL divergence by the temperature and max sequence length
        return kl_div * (temperature ** 2) / self.config["tokenizer"]["max_length"]
    
    def backward_kl_divergence(self, student_logits, teacher_logits):
        temperature = self.config["distillation"]["temperature"]

        # Compute softmax for both student and teacher logits
        student_probs = F.softmax(student_logits / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)

        # Compute the KL divergence (teacher || student)
        kl_div = F.kl_div(teacher_log_probs, student_probs, reduction='batchmean', log_target=False)

        # Scale the KL divergence by temperature squared and max sequence length
        return kl_div * (temperature ** 2) / self.config["tokenizer"]["max_length"]
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        
        eval_loss = 0.0
        num_examples = 0
        chunk_size = 4  # Adjust this value based on your GPU memory

        for step, inputs in enumerate(dataloader):
            for i in range(0, inputs["input_ids"].size(0), chunk_size):
                chunk_inputs = {k: v[i:i+chunk_size] for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                loss = self.compute_loss(self.model, chunk_inputs)
                eval_loss += loss.detach().float() * len(chunk_inputs["input_ids"])
                num_examples += len(chunk_inputs["input_ids"])
        
        eval_loss /= num_examples
        output.metrics[f"{metric_key_prefix}_loss"] = eval_loss.item()
        return output
    
def print_memory_stats():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
    print(f"Max Reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f}GB")

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

class MemoryTracker(TrainerCallback):
    def __init__(self, print_every=100):
        self.print_every = print_every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.print_every == 0:
            print(f"Step {state.global_step}:")
            print_memory_stats()
            clear_memory()

def get_custom_lr_scheduler(optimizer, num_warmup_steps, num_training_steps, initial_phase_steps):
    #Linear Warmup --> Constant learning rate --> Linear Annealing
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return current_step / num_warmup_steps  # Linear warmup
        elif current_step < initial_phase_steps:
            return 1.0  # Constant learning rate for initial phase
        else:
            # Linear annealing for the remaining steps
            return 1.0 - ((current_step - initial_phase_steps) / (num_training_steps - initial_phase_steps))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def main(config_path):
    config = load_config(config_path)
    accelerator = setup_environment(config)
    
    teacher_model, student_model, teacher_tokenizer, student_tokenizer = load_models_and_tokenizers(config)
    
    print(f"Student model: {student_model}")
    
    print("Memory after loading models:")
    print_memory_stats()
    clear_memory()

    train_dataset, eval_dataset = load_and_preprocess_dataset(config, student_tokenizer)
    
    # Ensure train_dataset is iterable and eval_dataset is a regular dataset
    # assert isinstance(train_dataset, IterableDataset)
    # assert isinstance(eval_dataset, Dataset)
    
    # Calculate max_steps
    total_samples = config["dataset"]["total_train_samples"] + config["dataset"]["eval_samples"]
    batch_size = config["training"]["per_device_train_batch_size"]
    grad_accum_steps = config["training"]["gradient_accumulation_steps"]
    num_gpus = torch.cuda.device_count()
    num_epochs = config["training_aux"]["num_train_epochs"]
    
    max_steps = int((total_samples / (batch_size * grad_accum_steps * num_gpus)) * num_epochs)
    
    # Ensure max_steps is a positive integer
    max_steps = max(1, max_steps)

    initial_phase_steps = int(max_steps * (1 - config["training_aux"]["annealing_phase_fraction"]))

    # Calculate save_steps, logging_steps, and eval_steps
    save_steps = max(1, int(max_steps * config["training_aux"]["save_steps_fraction"]))
    logging_steps = max(1, int(max_steps * config["training_aux"]["logging_steps_fraction"]))
    eval_steps = max(1, int(max_steps * config["training_aux"]["eval_steps_fraction"]))
    
    # Calculate warmup_steps if using warmup
    warmup_steps = int(max_steps * config["training"]["warmup_ratio"]) if config["training"]["warmup_ratio"] > 0 else 0

    print(f"Running with max_steps: {max_steps}, will start annealing at step: {initial_phase_steps}")

    run_name = f'v{config["models"]["student"].split("/")[-1]}_lr_{config["training"]["learning_rate"]}_rows_{total_samples}'

    training_args = TrainingArguments(
        **config["training"],
        max_steps=3000,  # Explicitly set max_steps
        num_train_epochs=config["training_aux"]["num_train_epochs"],  # Set to None when using max_steps
        run_name=run_name,
        logging_dir=f"./logs/{run_name}",
        save_steps=save_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        warmup_steps=warmup_steps,
        # Default optimizer
        # optim="adamw_torch",
        
        # # Galore optimizer, uses 80%+ less memory than adamw_torch
        optim="galore_adamw_8bit",
        optim_target_modules=["mlp.down_proj","mlp.up_proj","mlp.gate_proj","self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj"],
        
        ddp_find_unused_parameters=False,
    )

    # Print out the values to verify
    print(f"max_steps: {max_steps}")
    print(f"num_train_epochs: {training_args.num_train_epochs}")

    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # This is now a regular Dataset, not IterableDataset
        tokenizer=student_tokenizer,
        config=config,  # This is your custom config, not SFTConfig
        dataset_text_field="text",
        max_seq_length=config["tokenizer"]["max_length"],
        packing=True,
    )
    
    if config.get("gradient_checkpointing", False)==True:
        # Disable caching for gradient checkpointing compatibility
        trainer.model.config.use_cache = False
    
    # Prepare the trainer, models, and datasets
    #Accelerator will automatically set the correct device and prepare the model and it is used for distributed training.
    trainer, teacher_model, train_dataset, eval_dataset = accelerator.prepare(
        trainer, teacher_model, train_dataset, eval_dataset
    )
    
    # Update the teacher model and datasets in the trainer
    trainer.teacher_model = teacher_model
    trainer.train_dataset = train_dataset
    trainer.eval_dataset = eval_dataset

    # Add custom scheduler
    optimizer = trainer.create_optimizer()
    scheduler = get_custom_lr_scheduler(optimizer, warmup_steps, max_steps, initial_phase_steps)
    trainer.lr_scheduler = scheduler

    trainer.add_callback(MemoryTracker())
    
    print("Starting knowledge distillation with evaluation...")
    try:
        trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])
    except RuntimeError as e:
        print(f"An error occurred during training: {e}")
        print("Please check that your GPU has enough memory and that all tensors are on the same device.")
        raise
    finally:
        print("Final memory stats:")
        print_memory_stats()
    
    print(f"Distillation completed. Saving model to {config['training']['output_dir']}")
    trainer.save_model(config['training']['output_dir'])
    
    trainer.push_to_hub()

if __name__ == "__main__":
    import torch
    print("GPU Available:", torch.cuda.is_available())
    print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    main("config_v10.yaml")
