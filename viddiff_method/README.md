# Viddiff method overivew
The viddiff method has 3 components: 
- `stage1_proposer.py`: in 'open' eval it calls an LLM to propose possible differences strings between actions based on the action description. In open and closed eval, it does some work that is later usedby the 'retriever' module: first it proposes a list of subactions, then creates retrieval strings for each of those subactions, and finally it links the subactions to the proposed difference strings. 
- `stage2_retriever.py`: for each difference and each video, this retrieves the most relevant frames. To do this, it calls CLIP to compute text-image similarity for the retrieval strings (from last stage), then does temporal segmentation of the subactions using the `utils_retriever.py` module. 
- `stage3_differencer.py`: given the retrieval frames and difference strings, query a VLM for whether the difference applies more for video A or B or neither.

The results of calls to LLM APIs and to CLIP are automatically cached (in `cache` directory). 

## Run the VidDiff method
First get the dataset from https://huggingface.co/datasets/jmhb/VidDiffBench.

Then run the VidDiff method with:
```
python viddiff_method/run_viddiff.py --config viddiff_method/configs/config.yaml --name viddiff_easy --split easy --eval_mode closed --subset_mode 0
```
This should work out of the box because we have implemented caching of all LLM and CLIP calls and included the cache in the repo. 

But if you change the code or data at all, then you'll need to set up the api key `$OPENAI_API_KEY`, and run a CLIP server. We used a CLIP server because it saves loading the CLIP model for each run of the viddiff method. We tested this on an a6000. Run `python apis/clip_server.py &`, and the server is running when you see this (takes about a minute):
```
 * Serving Flask app 'clip_server'
 * Debug mode: off
INFO:werkzeug:WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8090
 * Running on http://10.79.12.235:8090
INFO:werkzeug:Press CTRL+C to quit
```

This code uses [OpenClip](https://github.com/mlfoundations/open_clip) like this: 
```
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained="laion2b_s39b_b160k")
```
This creates `tmp` directory which saves images for the CLIP server. This is not the fastes way to do this, but for a smaller dataset it's manageable. Also the function automatically does embedding caching. 


