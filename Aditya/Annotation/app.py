from huggingface_hub import snapshot_download

models = [
    "Qwen/Qwen2.5-7B-Instruct",
    "teknium/OpenHermes-2.5-Mistral-7B",
    "microsoft/Phi-4-mini-instruct",
]

for model_id in models:
    print(f"⬇️ Downloading {model_id} ...")
    snapshot_download(
        repo_id=model_id,
        local_dir=f"./models/{model_id.replace('/', '__')}",
        resume_download=True,
    )
    print(f"✅ Done: {model_id}\n")
