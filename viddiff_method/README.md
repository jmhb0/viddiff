# Viddiff method 
The viddiff method has 3 components: 
- Proposer: in '`open' eval it calls an LLM to propose possible differences strings between actions based on the input action string. In open and closed eval, it also calls an LLM to create strings that are used by the retrieval model 
- Retriever
- FrameDifferenceer

Results to LLM calls and CLIP calls are automatically cached (in `cache` directory). 

# Run the VidDiff method
The method depends on a CLIP server you have to start locally and the OpenAI API. 

## Set API keys
Set $OPENAI_API_KEY environment variable, which is used by the Proposer and FrameDifferenceer modules

Run the VidDiff method with:
```
python -m ipdb viddiff_method/run_viddiff.py --config viddiff_method/configs/config.yaml --name viddiff_easy --split easy --eval_mode closed --subset_mode 0
```
If the results are not cached, you first 


## Run CLIP model as a server
The `retriever` module calls CLIP. To avoid loading CLIP each time, we run it as a server. We tested this on a6000 hardware. 

Run `python apis/clip_server.py &`, which uses [OpenClip](https://github.com/mlfoundations/open_clip) like this: 
```
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained="laion2b_s39b_b160k")
```
Then the code will call 

Creates `tmp` directory which saves images for the CLIP server. This is not the fastes way to do this, but for a smaller dataset it's manageable. Also the function automatically does embedding caching into ``
