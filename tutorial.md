# Generating Videos from images with Wan 2.1 running on Koyeb

When I first tried out image prompting with DALL-E, it absolutely blew my mind. I couldn't believe that you could create complete images based on just a couple of words in a matter of seconds.

![](/IMG_1927.PNG)
![](/IMG_1932.PNG)
_I must have been hungry when I was first trying this out_

Only a couple years later, there are now a variety of diffusion models that can create images and even videos, with advancements coming faster all the time as companies and researchers race to create the next big thing. Just two years ago, the YouTube channel Corridor Crew sought to push the bounds of what is possible by creating animations from generated images. Making video from images isn't as simple as generating multiple images and stringing them together, as it requires memory of the previous frame and how things were positioned, along with the ability to accurately predict what the next movement should be. AI has come a long way since that YouTube video from just two years ago, and now generating video from images is possible for developers like us!

In this tutorial, we explore usage of Wan2.1, a comprehensive and open suite of video foundation models that seek to push the boundaries of video generation. We'll generate images from video with Wan2.1 by accessing it through HuggingFace's diffusers library and serving it on Koyeb through a FastAPI interface. For demonstration purposes, I've also included a React frontend running on Vite.

## Getting the code

To get the code for the video generation server, you can clone [this repo](https://github.com/jenperson/wan2.1-fastapi). You don't need to download the code directly unless you want to reference it, as you have the option to deploy directly from the repo using the "Deploy to Koyeb" button: [![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?name=wan2-1-fastapi&repository=jenperson%2Fwan2.1-fastapi&branch=main&builder=dockerfile&instance_type=gpu-nvidia-a100&regions=na&env%5BHF_HUB_ENABLE_HF_TRANSFER%5D=1)

To get the code for the frontend application, clone [this repo](https://github.com/jenperson/wan2.1-fastapi-frontend). You can run this app locally, or you have the option to deploy to Koyeb: [![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?name=wan2-1-fastapi-frontend&repository=jenperson%2Fwan2.1-fastapi-frontend&branch=main&run_command=npm+run+serve&instance_type=medium&regions=was)

## Viewing the video generation code

All of the functionality for the serverless app can be found in the `app.py` file. The majority of the code comes from Hugging Face's documentation on Wan Image to Video generation using the Hugging Face Diffusers library.

### Implementing video generation in app.py

First, import the required packages:

```python
import os
import uuid


import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import CLIPVisionModel
```
Then set up the model path and the directory where generated videos will be published, and initialize your FastAPI instance:
```python
model_path = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"


# Add directory to store generated videos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "generated_videos")
STATUS_DIR = os.path.join(BASE_DIR, "job_status")
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(STATUS_DIR, exist_ok=True)


app = FastAPI()
app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],  # Or restrict to your client app URL for tighter security
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)


app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")
```
Note that CORS les are also included to allow the client app to access the API.

Next, load the Wan2.1 model:
```python
# Load the model from the container
image_encoder = CLIPVisionModel.from_pretrained(
   model_path, subfolder="image_encoder", torch_dtype=torch.float32
)
vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(
   model_path,
   vae=vae,
   image_encoder=image_encoder,
   torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
print("Model loaded!")
```
Next, define the API request schema:

```
# Define the request schema
class GenerationRequest(BaseModel):
   prompt: str
   negative_prompt: str = ""
   num_frames: int = 33
   image_url: str = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
```


Use a function called `run_video_job` to kick off the video generation process:

```python
def run_video_job(job_id: str, request: GenerationRequest):
   try:
       image = load_image(request.image_url)


       max_area = 480 * 832
       aspect_ratio = image.height / image.width
       mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
       height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
       width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
       image = image.resize((width, height))


       frames = pipe(
           image=image,
           prompt=request.prompt,
           negative_prompt=request.negative_prompt,
           num_frames=request.num_frames,
           guidance_scale=5.0,
       ).frames[0]


       video_filename = os.path.join(VIDEO_DIR, f"{job_id}.mp4")
       export_to_video(frames, video_filename, fps=16)


       # Mark job as complete
       with open(os.path.join(STATUS_DIR, f"{job_id}.done"), "w") as f:
           f.write("done")


   except Exception as e:
       with open(os.path.join(STATUS_DIR, f"{job_id}.error"), "w") as f:
           f.write(str(e))

```


Declare an endpoint called `generate-video` that kicks off the `run_video_job` function that kicks off video generation as a background task. This sends a response to the client while continuing to complete the job, providing the client a job ID that can be used to check the status of video generation and reference the location of the video once loaded.

```python
@app.post("/generate-video")
def generate_video(request: GenerationRequest, background_tasks: BackgroundTasks):
   print(f"request created: {request.prompt}, {request.negative_prompt}, {request.num_frames}, {request.image_url}")
   job_id = str(uuid.uuid4())


   background_tasks.add_task(run_video_job, job_id, request)


   return {"job_id": job_id, "status_url": f"/status/{job_id}", "video_url": f"/videos/{job_id}.mp4"}

```

Declare another endpoint called `status` that is used to poll the task and determine if it has completed:

```python
@app.get("/status/{job_id}")
def get_status(job_id: str):
   if os.path.exists(os.path.join(STATUS_DIR, f"{job_id}.done")):
       return {"status": "completed", "video_url": f"/videos/{job_id}.mp4"}
   elif os.path.exists(os.path.join(STATUS_DIR, f"{job_id}.error")):
       with open(os.path.join(STATUS_DIR, f"{job_id}.error"), "r") as f:
           return {"status": "error", "detail": f.read()}
   else:
       return {"status": "processing"}
```

### Creating the Docker container

To easily build and scale your applications, Koyeb uses container orchestration through Docker.

The following code comprises the Dockerfile for the diffuser server:

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
   python3-pip \
   git \
   git-lfs && \
   rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# This is only working from a separate install right now
RUN pip install git+https://github.com/huggingface/diffusers

# Copy the model in chunks
COPY model/model_index.json /app/model/
COPY model/image_encoder /app/model/image_encoder/
COPY model/image_processor /app/model/image_processor/
COPY model/scheduler /app/model/scheduler/
COPY model/text_encoder /app/model/text_encoder/
COPY model/tokenizer /app/model/tokenizer/
COPY model/transformer /app/model/transformer/
COPY model/vae /app/model/vae/


# Ensure the model is detected properly in diffusers
ENV HUGGINGFACE_HUB_CACHE="/app/model"

# Copy the API code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

```

Notice the multiple operations in the following format:

```
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', allow_patterns=['model_index.json'])"
```

In order to speed up download of the model, we download it in chunks. This leads to faster builds, which is especially important as you scale. You can abstract this functionality for easy reuse, but it is shown here for demonstration purposes.

## Viewing the frontend code

The important functionality of the React frontend can be found in `App.tsx`. The `handleGenerate()` function calls to the FastAPI:

```tsx
 const handleGenerate = async () => {
   setLoading(true);
   const endpoint = "/generate-video";


   if (!imageUrl || !prompt || prompt === "") {
     alert("Please enter an image URL and a prompt.");
     setLoading(false);
     return;
   }


   try {
     const response = await fetch(`${apiUrl}${endpoint}`, {
       method: "POST",
       headers: { "Content-Type": "application/json" },
       body: JSON.stringify({
         prompt,
         image_url: imageUrl,
         negative_prompt: negPrompt,
         num_frames: frameCount,
       }),
     });


     if (!response.ok) {
       alert("Failed to start video generation job");
       throw new Error("Failed to start job");
     }


     const data = await response.json();
     const jobId = data.job_id;


     // Poll for job status
     const pollInterval = 5000; // 5 seconds
     const maxAttempts = 120; // 10 minutes
     let attempts = 0;


     const poll = async () => {
       attempts++;
       const statusResponse = await fetch(`${apiUrl}/status/${jobId}`);
       const statusData = await statusResponse.json();


       if (
         statusResponse.ok &&
         statusData.status === "completed" &&
         statusData.video_url
       ) {
         setVideoUrl(`${apiUrl}${statusData.video_url}`);
         setLoading(false);
       } else if (statusData.status === "error") {
         setLoading(false);
         alert("Error generating video");
       } else if (attempts < maxAttempts) {
         setTimeout(poll, pollInterval);
       } else {
         setLoading(false);
         alert("Timed out waiting for video generation.");
       }
     };


     poll();
   } catch (error) {
     console.error(error);
     setLoading(false);
   }
 };

```

## Deploying to Koyeb

To deploy the application to Koyeb, there are two main components: the Python application to be deployed on GPU, and the web application to be deployed on CPU. If you opted to download the code, you can deploy these by  selecting your repos from GitHub. Alternatively, you can use the "Deploy to Koyeb" button as previously shown.

## Setting up the frontend

To run the demo, you can deploy the frontend or run it locally on your machine. You can find both options in the following steps.

### Running the frontend locally

1. Create a .env file at the root of the project.

2. Add your Koyeb web URL to the .env file:

```
VITE_API_BASE_URL=https://YOUR-PROJECT-ID.koyeb.app
```

3. Install required dependencies:

```
npm install
```

4. Start the app:

```
npm run dev
```

The app is now running locally at `http://localhost:5173/`

### Deploying the frontend

Use the *Deploy to Koyeb* button in the README to deploy the app. Under *Settings > Environment variables and files*, add the following:
Name:

```
VITE_API_BASE_URL

```
Value (Replace with the unique ID of your Image to Video Generator):

```
https://YOUR-PROJECT-ID.koyeb.app
```

### Running the demo

Provide a URL, a prompt, and optionally, a negative prompt. You can also click **Use Default Values** to populate the fields with an example.
Video generation can take up to ten minutes. You can check the logs in the Koyeb console to verify that the frontend is periodically polling the status of video generation.

## And that's it!

In just a few steps, we were able to deploy an image-to-video generator and a client application in which to view the video. For more image and video generation options with Wan2.1, view the Hugging Face Diffusers documentation. For inspiration and guidance, check out more solutions in Koyeb's tutorial documentation.