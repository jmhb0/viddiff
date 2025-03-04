from huggingface_hub import hf_hub_download
import zipfile
import os


def download_and_extract_videos(dataset_name, output_dir, domain):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Download the zip file
    # print(f"Downloading {zip_filename} from the dataset...")
    zip_path = hf_hub_download(
        repo_id=dataset_name,
        filename=f"data/videos_{domain}.zip",
        repo_type="dataset",
        local_dir=output_dir,
    )

    # Extract the zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            extracted_path = os.path.join(output_dir, file_info.filename)
            if file_info.filename.endswith("/"):
                os.makedirs(extracted_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
                with zip_ref.open(file_info) as source, open(
                    extracted_path, "wb"
                ) as target:
                    target.write(source.read())

    print(f"Videos extracted to {output_dir}")


# Usage
dataset_name = "jmhb/ViddiffBench"
output_dir = "."
domains = ("surgery", "fitness")
for domain in domains:
    download_and_extract_videos(dataset_name, output_dir, domain)
