# DAPF
Officical Implementation of: "Domain Adaptation via Prompt Learning for Alzheimer’s Detection", accepeted in EMNLP-Findings 2024.
<img width="926" alt="image" src="https://github.com/user-attachments/assets/9c0e01d5-2a28-4f44-b375-97b8e079ec05">

#Install


Run "pip install -r /path/to/requirements.txt" 


#Domain-Adaptive Prompt based Finetuning Command
After downloading the DAPF repository and running requirements.txt to install packages, you can run the following commands in the parent directory of DAPF directory. We have used OpenPrompt framework for prompt finetune the PLMs. I have downloaded the code of OpenPrompt code directly instead of installing the packages and made some changes in some files for experiment purpose. 

Before running the run_prompt_finetune.py or run_prompt_finetune_test.py in the following instruction, you'll have to define the project_root, logs_root, off_line_model_dir, data_dir configurations in your scripts. These configuration should be set to 1) the parent directory of your prompt_ad_code folder; 2) the directory to store your output (model or results); 3) the directory you store pre-trained model downloaded from huggingface; 4) the directory you store ADReSS data (csv file), respectively.
--project_root /parent/directory/DAPF \
--logs_root /directory/to/store/your/output \
--off_line_model_dir /model \      
--data_dir /directory/you/store/ADReSS/data \



