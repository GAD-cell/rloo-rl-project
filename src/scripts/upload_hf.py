from huggingface_hub import HfApi, create_repo, login

login() 
api = HfApi()

create_repo(
    repo_id="GAD-cell/sft_commongen",
    repo_type="model",
    private=False, 
    exist_ok=True   
)

api.upload_folder(
    folder_path="../models/sft_model",
    repo_id="GAD-cell/sft_commongen",
    repo_type="model",
)