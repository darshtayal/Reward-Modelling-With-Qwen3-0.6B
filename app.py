#install all the libraries needed
!pip install transformers datasets trl 

#import all nessaccary libraries
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from trl import RewardConfig, RewardTrainer
import torch 

#load the model
model = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen3-0.6B", num_labels=1, attn_implementation="eager")

#load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

#load the raw dataset
dataset_raw = load_dataset("Dahoas/full-hh-rlhf")

#get a glimpse of our dataset
print("prompt:", dataset_raw['train']['prompt'][99])
print("response:", dataset_raw['train']['response'][99])
print("chosen:", dataset_raw['train']['chosen'][99])
print("rejected:", dataset_raw['train']['rejected'][99])

#select a resonable length of our dataset
dataset_pre_formatted_train = dataset_raw['train'].select(range(3000)) 
dataset_pre_formatted_test = dataset_raw['test'].select(range(500)) 

#check the length of our dataset to get an idea of 'max_length' hyperparameter
def check_dataset_length(dataset, tokenizer):
    max_len = 0
    for example in dataset:
        text_chosen = example['prompt'] + example['chosen']
        text_rejected = example['prompt'] + example['rejected']

        len_chosen = len(tokenizer(text_chosen)['input_ids'])
        len_rejected = len(tokenizer(text_rejected)['input_ids'])

        max_len = max(max_len, len_chosen, len_rejected)
    
    return max_len

#run the function  
check_dataset_length(dataset_pre_formatted_train, tokenizer)

#tokenize the dataset
def format_dataset(example):
  chosen = tokenizer(example['prompt'] + example['chosen'], padding='max_length', truncation=True, max_length=600, return_tensors='pt')
  rejected = tokenizer(example['prompt'] + example['rejected'], padding='max_length', truncation=True, max_length=600, return_tensors='pt')
  return {
      "input_ids_chosen": chosen['input_ids'],
      "attention_mask_chosen": chosen['attention_mask'],
      "input_ids_rejected": rejected['input_ids'],
      "attention_mask_rejected": rejected['attention_mask']
  }

#Convert the input_ids and attention_mask to pytorch tensors for extra safety
def convert_dataset(example):
  return {
      "input_ids_chosen": torch.tensor(example["input_ids_chosen"][0]),
      "attention_mask_chosen": torch.tensor(example["attention_mask_chosen"][0]),
      "input_ids_rejected": torch.tensor(example["input_ids_rejected"][0]),
      "attention_mask_rejected": torch.tensor(example["attention_mask_rejected"][0])
  }

formatted_dataset_train = dataset_pre_formatted_train.map(format_dataset, batched=False)
formatted_dataset_test = dataset_pre_formatted_test.map(format_dataset, batched = False)

#remove unnessacary columns
formatted_dataset_train = formatted_dataset_train.remove_columns(['prompt', 'chosen', 'rejected', 'response'])
formatted_dataset_test = formatted_dataset_test.remove_columns(['prompt', 'chosen', 'rejected', 'response'])

formatted_dataset_train = formatted_dataset_train.map(convert_dataset, batched=False)
formatted_dataset_test = formatted_dataset_test.map(convert_dataset, batched=False)

#training arguments
trainer_args = RewardConfig(
    output_dir = 'trained_reward_model',
    num_train_epochs = 1,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 2,
    logging_steps = 1,
    learning_rate = 3e-5,
    eval_steps = 50,
    report_to = 'none'
)

#Initialize the trainer
trainer = RewardTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = formatted_dataset_train,
    eval_dataset = formatted_dataset_test,
    args = trainer_args
)

#get the data from the trainer to check if everything is right
dl = trainer.get_train_dataloader()
dl_iter = iter(dl)
batch = next(dl_iter)
print(tokenizer.decode(batch['input_ids_chosen'][0]))

#train the reward model
trainer.train()

#Test our model
print("___ INFERENCE TIME ___")
prompt = "Why is the sky blue?"
good = "Because it has refraction"
bad = "Because it is not green"

textgood = f"User:\n{prompt}\nAssistant:\n{good}"
textbad  = f"User:\n{prompt}\nAssistant:\n{bad}"

# Tokenize with batch dimension
token_good = tokenizer(textgood, return_tensors="pt").to(model.device)
token_bad  = tokenizer(textbad,  return_tensors="pt").to(model.device)

with torch.inference_mode():
    # Pass tensors correctly
    good_reward = model(**token_good).logits.squeeze(-1)
    bad_reward  = model(**token_bad).logits.squeeze(-1)

if good_reward > bad_reward:
    print("The reward for the good prompt is higher.")
elif good_reward < bad_reward:
    print("The reward for the bad prompt is higher.")
else:
    print("The reward for both the prompts was same.")

