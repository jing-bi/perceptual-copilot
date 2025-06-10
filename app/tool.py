import datetime
import json
import cv2
import httpx
from app.config import env
from app.utils import image_w_box, encode_image
from agents import RunContextWrapper, function_tool
from app.memory import Memory,Snapshot




def task(name, image):
    resp = httpx.post(f"{env.end_task}",
        data={"name": name},
        files={"file": ("frame.jpg", image.tobytes(), "image/jpeg")},
        timeout=10,
        headers={"Authorization": env.api_key},
    )
    resp.raise_for_status()
    return resp.json()['result']

def completion(messages, model):
    response = env.client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content


def completion_image(images, prompt, model):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }
        for b64, mime in map(encode_image, images)
    ]
    return completion(messages, model=model)

# ------------------------ Function Tools ------------------------
@function_tool
def caption(wrapper: RunContextWrapper[Memory]) -> str:  
    """
    Generate a descriptive caption for the most recent frame, record it as a snapshot, and return it.
    Returns:
        str:
            The generated caption for the current view (i.e., the latest frame).
    """
    mem = wrapper.context
    prompt = "Describe the image with rich details but in a concise manner."
    result = completion_image([mem.frames[-1]], prompt, env.model_mllm)
    mem.snapshots.append(Snapshot(sender='caption', data=result))
    return result

@function_tool
def ocr(wrapper: RunContextWrapper[Memory]) -> str:  
    """
    Perform OCR on the most recent frame, record it as a snapshot, and return the extracted text.
    Returns:
        str:
            The extracted text from the current view (i.e., the latest frame).
    """
    mem = wrapper.context
    prompt = "Extract all text from image/payslip without miss anything."
    result = completion_image([mem.frames[-1]], prompt, env.model_mllm)
    mem.snapshots.append(Snapshot(sender='ocr', data=result))
    return result

@function_tool
def qa(wrapper: RunContextWrapper[Memory], question: str) -> str:  
    """
    Answer a question based on the most recent frame, record it as a snapshot, and return the answer.

    Args:
        question (str): The question to be answered.
    Returns:
        str:
            The answer to the question based on the current view (i.e., the latest frame).
    """
    mem = wrapper.context
    prompt = f"Answer the question based on the image. Question: {question}"
    result = completion_image([mem.frames[-1]], prompt, env.model_mllm)
    mem.snapshots.append(Snapshot(sender='qa', data=result))
    return result


@function_tool
def localize(wrapper: RunContextWrapper[Memory]) -> str:
    """
    Localize all objects in the most recent frame
    Returns:
        str:
            The localization result for the current view (i.e., the latest frame).
            the format is {name:list of bboxes}
    """
    mem = wrapper.context
    frame = mem.frames[-1]
    _, img = cv2.imencode('.jpg', frame)
    objxbox = task(env.model_loc, img)
    mem.snapshots.append(Snapshot(sender='localize', data=image_w_box(frame, objxbox)))
    return json.dumps(objxbox, indent=2)


@function_tool
def time(wrapper: RunContextWrapper[Memory]) -> str:  
    """
    Get the current time, record it as a snapshot, and return the time.
    Returns:
        str:
            The current time.
    """
    mem = wrapper.context
    result = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mem.snapshots.append(Snapshot(sender='time', data=result))
    return result

def sample_frames(mem: Memory, n: int) -> list:
    """
    Sample frames from the past n seconds of video.
    
    Args:
        mem (Memory): The memory context containing frames.
        n (int): Number of seconds to look back for video frames.
    Returns:
        list: Sampled frames from the video sequence.
    """
    if len(mem.frames) == 0:
        return []
    
    available_frames = min(n * env.fps, len(mem.frames))
    recent_frames = mem.frames[-available_frames:]
    sampled_frames = recent_frames[::env.fps // 2]
    
    return sampled_frames

@function_tool
def video_caption(wrapper: RunContextWrapper[Memory], n=2) -> str:
    """
    Generate a descriptive caption for a video sequence from the past n seconds of frames.
    The n is a required parameter that specifies how many seconds of video frames to consider.
    
    Args:
        n (int): Number of seconds to look back for video frames.
    Returns:
        str:
            The generated caption for the video sequence from the past n seconds.
    """
    mem = wrapper.context
    sampled_frames = sample_frames(mem, n)
    
    if len(sampled_frames) == 0:
        return "No frames available for video caption."
    
    prompt = "Describe this video sequence focusing on any changes or actions that occur over time."
    result = completion_image(sampled_frames, prompt, env.model_mllm)
    mem.snapshots.append(Snapshot(sender='video caption', data=result))
    return result

@function_tool
def video_qa(wrapper: RunContextWrapper[Memory], question: str, n=2) -> str:
    """
    Answer a question based on a video sequence from the past n seconds of frames.
    
    Args:
        question (str): The question to be answered.
        n (int): Number of seconds to look back for video frames.
    Returns:
        str:
            The answer to the question based on the video sequence from the past n seconds.
    """
    mem = wrapper.context
    sampled_frames = sample_frames(mem, n)
    
    if len(sampled_frames) == 0:
        return "No frames available for video Q&A."
    
    prompt = f"Answer the question based on this video sequence. Question: {question}"
    result = completion_image(sampled_frames, prompt, env.model_mllm)
    mem.snapshots.append(Snapshot(sender='video qa', data=result))
    return result
