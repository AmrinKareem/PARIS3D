import argparse
import os
import sys
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import random
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from PARIS3D.PARIS3D import LISAForCausalLM
import PARIS3D.llava.conversation as conversation_lib
from PARIS3D.llava.mm_utils import tokenizer_image_token
from PARIS3D.segment_anything.utils.transforms import ResizeLongestSide
from PARIS3D.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
#set random seed
random.seed(0)
def create_model(version, model_max_length, precision, load_in_4bit, load_in_8bit, local_rank, image_size, legacy):
 # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        version,
        cache_dir=None,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0] #used to obtain the token ID of a special token "[SEG]" in the tokenizer's vocabulary
    #obtained token ID (input_ids[0]) is assigned to args.seg_token_idx

    torch_dtype = torch.float32
    if precision == "bf16":
        torch_dtype = torch.bfloat16
    elif precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "device_map": "auto",
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "device_map": "auto",
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM.from_pretrained(
        version, low_cpu_mem_usage=True, seg_token_idx=seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        precision == "fp16" and (not load_in_4bit) and (not load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        model_engine = deepspeed.init_inference(
            model=model,
            mp_size=world_size,
            dtype=torch.half,
            replace_with_kernel_inject=True,
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower) # initializing the clip_image_processor with the image processing configuration that corresponds to the vision tower of your CLIP model. 
    transform = ResizeLongestSide(image_size)

    return model, clip_image_processor, transform, tokenizer

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def paris3d_inference(model, clip_image_processor, transform, tokenizer, save_dir, part_names, category, sp_dir, precision="fp16", num_views=10, use_mm_start_end=True,
                    save_pred_img=True, save_individual_img=False, save_pred_json=True):
    # model, clip_image_processor, transform, tokenizer=create_model(version="xinlai/LISA-13B-llama2-v1", model_max_length=512, precision="fp16", load_in_4bit=False, load_in_8bit=True, local_rank=0, image_size=1024, use_mm_start_end=True)
    model.eval()
    pred_dir = os.path.join(save_dir, f"paris3d_pred")
    os.makedirs(pred_dir, exist_ok = True)
    shape = {
    "Bottle": ["cylindrical"],
    "Box": ["rectangular"],
    "Bucket": ["curved semi-circular steel"],
    "Camera": ["round or square", "round"],
    "Cart": ["circular"],
    "Chair": ["long rectangular extension", "vertically oriented", "bar shaped", "horizontally oriented" , "round"],
    "Clock": ["thin needle-like"],
    "CoffeeMachine": ["small round or square", "cylindrical", "round", "square or round"],
    "Dishwasher": ["rectangular or square", "thin horizontal bar shaped"],
    "Dispenser": ["cylindrical neck like", "cylindrical"],
    "Display": ["round or square", "rectangular", "vertical bar shaped"],
    "Door": ["rectangular", "rectangular", "oval"],
    "Eyeglasses": ["a pair of circular or rectangular", "thin, long wire-like"],
    "Faucet": ["cylindrical", "cylindrical"],
    "FoldingChair": ["horizontally oriented, rectangular"],
    "Globe": ["spherical"],
    "Kettle": ["circular", "curved", "long cylindrical pipe-like"],
    "Keyboard": [" long wire-like", "rectangular"],
    "KitchenPot": ["circular", "curved loop shaped"],
    "Knife": ["steel"],
    "Lamp": ["circular", "vertical cylindrical", "circular", "cylindrical"],
    "Laptop": ["rectangular", "rectangular", "thin rectangular", "rectangular", "circular"],
    "Lighter": ["cuboidal", "disc-shaped", "rectangular"],
    "Microwave": ["rectangular", "rectangular", " small rectangular", "small round"],
    "Mouse": ["rectangular", "thin wire-like", "thin, disc-like, scrollable"],
    "Oven": ["rectangular", "round"],
    "Pen": ["cylindrical", "cylindrical"],
    "Phone": ["rectangular", "small square"],
    "Pliers": ["long, bar-like and curved teeth-shaped"],
    "Printer": ["round"],
    "Refrigerator": ["rectangular", "bar shaped"],
    "Remote": ["small square or round"],
    "Safe": ["rectangular", "circular rotating", "small square or round"],
    "Scissors": ["long steel", "plastic, loop shaped", "small steel, threaded and cylindrical"],
    "Stapler": ["rectangular", "rectangular"],
    "StorageFurniture": ["rectangular", "rectangular", "loop shaped"],
    "Suitcase": ["rectangular or circular, loop shaped", "circular"],
    "Switch": ["rectangular projection on the wall"],
    "Table": ["rectangular", "rectangular", "vertically oriented, bar shaped", "horizontally oriented, rectangle shaped", "round", "circular"],
    "Toaster": ["rectangular", "rectangular"],
    "Toilet": ["rectangular", "rectangular", "rectangular"],
    "TrashCan": ["rectangular", "rectangular", "small rectangular"],
    "USB": ["cuboidal", "cuboidal, flap-shaped, and movable"],
    "WashingMachine": ["circular or rectangular", "small circular"],
    "Window": ["rectangular panel-shaped"]
    }
    part2cat1 = {
    "Bottle": ["that can be twisted to open and close"],


    "Box": ["that can be opened and shut"],


    "Bucket": ["that you can hold"],


    "Camera": ["clicked on to take a picture", 
    "which captures the image"],

    "Cart": ["that rolls on the ground"],

    "Chair": ["where you rest your arms",
        "on which you lean back",
        "that has the wheels attached to it",
        "on which you sit",
        "that helps the chair move"],

    "Clock": ["that moves to measure the time"],

    "CoffeeMachine": [
        "that is pressed to turn on",
        "that contains the coffee or water",
        "that can be turned to adjust settings",
        "can be opened and closed"
    ],

    "Dishwasher": [
        "that can be opened and closed",
        "that you can hold to open and close"
    ],
    "Dispenser": [
        "that supports the lid of the dispenser",
        "that dispenses the liquid"
    ],
    "Display": [
        "that is flat and it stands on",
        "that displays visuals",
        "between the screen and base used to hold it up"
    ],
    "Door": [
        "that surrounds and frames the door",
        "that is moved for opening and closing the entranceway",
        "that you can hold to open and close it"
    ],
    "Eyeglasses": [
        "that is used to see",
        "that goes behind your ears"
    ],
    "Faucet": [
        "that water comes out of",
        "that is pressed to turn the water on and off"
    ],
    "FoldingChair": [
        "that you sit on"
    ],
    "Globe": [
        "that is round"
    ],
    "Kettle": [
        "that can be opened and closed",
        "that you hold",
        "that liquid comes out of"
    ],
    "Keyboard": [
        "that attaches it to other devices",
        "that you can press on and type"
    ],
    "KitchenPot": [
        "that can be opened and closed",
        "that you can hold"
    ],
    "Knife": [
        "that is sharp and is used to cut"
    ],
    "Lamp": [
        "that it stands on",
        "that forms its body",
        "that emits light",
        "that is cylindrical and goes around the bulb"
    ],
    "Laptop": [
        "that has keys and is pressed to type",
        "that is used to display visuals",
        "that supports the display and the keyboard",
        "that is a flat touch-sensitive surface and can track your finger movements",
        "used to take a picture"
    ],
    "Lighter": [
        "that can be opened and closed",
        "that can be rolled",
        "that can be pressed on"
    ],
    "Microwave": [
        "that displays time",
        "that can be opened and closed",
        "that you can hold",
        "that can be pressed to turn on and off"
    ],
    "Mouse": [
        "that can be clicked",
        "that is used to connect it to other devices",
        "that rolls"
    ],
    "Oven": [
        "that can be opened and closed",
        "that can be turned to set the time"
    ],
    "Pen": [
        "that can be placed on top and removed",
        "that can be clicked"
    ],
    "Phone": [
        "that can be opened and closed",
        "that can be pressed"
    ],
    "Pliers": [
        "that are squeezed"
    ],
    "Printer": [
        "that is pressed to turn on and off"
    ],
    "Refrigerator": [
        "that can be opened and closed",
        "that you can hold or grasp to open and close it"
    ],
    "Remote": [
        "that can be pressed to change options"
    ],
    "Safe": [
        "that can be opened and closed",
        "that can be turned and rotated",
        "that can be pressed"
    ],
    "Scissors": [
        "that can cut objects",
        "that you hold on",
        "that is used to tighten the blades and the handle together"
    ],
    "Stapler": [
        "that forms the base",
        "that can be lifted and pressed on"
    ],
    "StorageFurniture": [
        "that can be opened and closed",
        "that can be slid in and out",
        "like a knob that you can hold to open and close"
    ],
    "Suitcase": [
        "that you can hold",
        "that rolls on the floor"
    ],
    "Switch": [
        "that you can flick on and off"
    ],
    "Table": [
        "that can be opened and closed",
        "that can be slid in and out",
        "that supports it to stand",
        "that is flat on the top",
        "that rolls",
        "that you can hold to open and close"
    ],
    "Toaster": [
        "that you can press to turn on",
        "that you can slide up and down to set the level of toasting"
    ],
    "Toilet": [
        "that opens and closes",
        "that you can sit on",
        "that you can press to flush"
    ],
    "TrashCan": [
        "that you step on to open and close",
        "that opens and closes",
        "that is a door"
    ],
    "USB": [
        "used to secure and close it",
        "that can rotate"
    ],
    "WashingMachine": [
        "that can be opened and closed",
        "that can be pressed to turn it on and off"
    ],
    "Window": [
        "that you look out of"
    ]
    }
    part2cat = {
    "Bottle": [
        "can be twisted to open and close"
    ],
    "Box": [
        "can be opened and shut"
    ],
    "Bucket": [
        "provides a grip for carrying the bucket"
    ],
    "Camera": [
        "can be pressed to capture images",
        "focuses light onto the sensor"
    ],
    "Cart": [
        "rolls and aids in moving objects"
    ],
    "Chair": [
        "provides arm support",
        "forms the backrest",
        "supports body weight when someone sits on the chair",
        "can be used to sit on",
        "rolls and moves for mobility"
    ],
    "Clock": [
        "moves on its face and displays time"
    ],
    "CoffeeMachine": [
        "can be pressed to brew coffee",
        "holds coffee or water",
        "can be turned to adjust settings",
        "can be used to open and close the container"
    ],
    "Dishwasher": [
        "can be opened and closed",
        "serves as a grip for opening"
    ],
    "Dispenser": [
        "is pressed on to dispense liquid",
        "can be twisted to open and close, and typically contains the head"
    ],
    "Display": [
        "serves as a foundation for the screen",
        "shows visual information",
        "supports or holds the display"
    ],
    "Door": [
        "surrounds and provides access to an entrance",
        "can be opened and closed",
        "can be used as a knob or lever to open and close the door"
    ],
    "Eyeglasses": [
        "holds and supports the lenses",
        "extends behind the ears"
    ],
    "Faucet": [
        "where the water comes out of",
        "can be turned on or off and control the flow of water"
    ],
    "FoldingChair": [
        "serves as the seating surface"
    ],
    "Globe": [
        "forms a spherical representation of the Earth"
    ],
    "Kettle": [
        "can be opened and closed and located on top of the kettle",
        "provides a grip for lifting and pouring",
        "is where the liquid flows out of"
    ],
    "Keyboard": [
        "connects like a wire to the computer for typing",
        "consists of individual keys for input"
    ],
    "KitchenPot": [
        "can be opened and closed, usually located on top of the pot",
        "provides a comfortable way to hold for lifting"
    ],
    "Knife": [
        "features a sharp cutting edge"
    ],
    "Lamp": [
        "forms the foundation of the entire structure",
        "encloses all parts of the lamp and forms its body",
        "emits light and functions as the source of light",
        "covers the light source or bulb"
    ],
    "Laptop": [
        "has square keys and can be pressed or typed",
        "displays visual content on a rectangular surface",
        "is a rectangular part connecting the screen to the keyboard",
        "serves as a touch-sensitive surface in the center below the keyboard",
        "houses the built-in lens at the top center of the screen"
    ],
    "Lighter": [
        "can be lifted and shut",
        "can be rolled to produce a spark",
        "can be ignited by pressing on it"
    ],
    "Microwave": [
        "displays cooking information",
        "can be opened and closed",
        "provides a knob or lever for opening",
        "can be pressed to operate"
    ],
    "Mouse": [
        "serves as a pointing and clicking device",
        "is a long wire that connects to the computer",
        "includes a scrolling function and can be rolled"
    ],
    "Oven": [
        "can be opened and closed",
        "allows temperature adjustment"
    ],
    "Pen": [
        "covers and protects the writing tip",
        "can be clicked to extend or retract"
    ],
    "Phone": [
        "can be opened and closed",
        "has small round or square elements that can be pressed on for operation"
    ],
    "Pliers": [
        "provides a grip for grasping objects"
    ],
    "Printer": [
        "can be pressed to start printing"
    ],
    "Refrigerator": [
        "can be opened and closed",
        "provides a grip to hold for opening and closing"
    ],
    "Remote": [
        "features small round or square elements to press for remote control"
    ],
    "Safe": [
        "can be opened and closed securely",
        "can be turned to lock or unlock",
        "includes a round element to press for control"
    ],
    "Scissors": [
        "features a sharp cutting edge",
        "is oval or round for comfortable holding and gripping",
        "may include an adjustment screw located in between the blades"
    ],
    "Stapler": [
        "holds and dispenses staples, usually located at the bottom of the stapler",
        "can be lifted and closed, located at the top of the stapler",
    ],
    "StorageFurniture": [
        "can be opened and closed, located in front of the furniture",
        "is usually rectangular and used for storage, usually located in front of the furniture",
        "serves as a knob or lever like grip for opening"
    ],
    "Suitcase": [
        "is usually held for a grip, and can be elongated, usually located on top of the suitcase",
        "is round in shape and can roll on the floor, located at the bottom of the suitcase"
    ],
    "Switch": [
        "can be turned on or off"
    ],
    "Table": [
        "can be opened and closed",
        "can be drawn out and used for storage",
        "serves as a support on the floor",
        "provides a flat surface",
        "can roll on the floor for mobility",
        "can be held for gripping"
    ],
    "Toaster": [
        "can be pressed to start toasting",
        "can be slid up and down for heat adjustment"
    ],
    "Toilet": [
        "can be lifted and closed",
        "provides a comfortable sitting place",
        "can be pressed or held down for flushing, located on the top of the toilet",
    ],
    "TrashCan": [
        "can be stepped or stamped on to open",
        "can be lifted and shut, usually located on top of the can",
        "can be opened and closed, located in front of the can"
    ],
    "USB": [
        "covers its tip and protects the connection",
        "consists of a rotating part"
    ],
    "WashingMachine": [
        "can be opened and closed and located in front of the machine",
        "is usually round and can be pressed for operation, located on the top or front of the machine"
    ],
    "Window": [
        "allows light and view through",
    ]
    } 
    
    predictions = []
    for i in range(num_views):
            torch.cuda.empty_cache()
            image_np = cv2.imread("%s/rendered_img/%d.png" % (sp_dir, i))
            for j, part in enumerate(part_names):
                prompt = [f"Which {shape[category][j]} part of this {category} {part2cat[category][j]}? Please output the segmentation mask.", f"Can you identify the {shape[category][j]} part of this {category} {part2cat1[category][j]}? Output the segmentation mask.", f"Can you segment the {shape[category][j]} part {part2cat1[category][j]} in this {category}? Please output segmentation mask.", f"What is the {shape[category][j]} part of a {category} {part2cat1[category][j]}?", f"On this image of a {category}, can you identify the {shape[category][j]} part that {part2cat[category][j]}?, Please output segmentation mask.", f"Which {shape[category][j]} part of this {category} {part2cat[category][j]}? Output the segmentation mask.", f"Which {shape[category][j]} part {part2cat[category][j]} in a {category}? Output the segmentation mask.", f"What is commonly found in a {category} and {part2cat[category][j]}?", f"Where is the {shape[category][j]} part that {part2cat[category][j]} in this {category}?, Please output segmentation mask.", f"What is the {shape[category][j]} part of a {category} {part2cat1[category][j]}?"]
                k = random.randint(0,9)
                # prompt = f"Can you segment the part {part2cat[category][j]} in this {category}?, Please output segmentation mask."
                prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt[k] #pass prompt in function instead of accept here 
                if use_mm_start_end:
                    replace_token = (
                        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                    )
                    prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                original_size_list = [image_np.shape[:2]] #original_size_list is a list containing the dimensions (height and width) of the original input image.

                image_clip = (
                    clip_image_processor.preprocess(image_np, return_tensors="pt")[
                        "pixel_values"
                    ][0]
                    .unsqueeze(0)
                    .cuda()
                )
                if precision == "bf16":
                    image_clip = image_clip.bfloat16()
                elif precision == "fp16":
                    image_clip = image_clip.half()
                else:
                    image_clip = image_clip.float()

                image = transform.apply_image(image_np)
                resize_list = [image.shape[:2]] #resize_list is a list that contains the dimensions (height and width) of the image after it has undergone resizing transformations.

                image = (
                    preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
                    .unsqueeze(0)
                    .cuda()
                ) #changed to PyTorch tensor, rearrange dimensions as channels x height x width, moves to contiguous memory, adds a 4th dimension (batch), and moves to CUDA
                if precision == "bf16":
                    image = image.bfloat16()
                elif precision == "fp16":
                    image = image.half()
                else:
                    image = image.float()

                input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt") #The tokenizer converts the text into numerical tokens and wraps them in a PyTorch tensor.
                input_ids = input_ids.unsqueeze(0).cuda() #adds batch dimension
                output_ids = []
                
                output_ids, pred_masks = model.evaluate(
                    image_clip,
                    image,
                    input_ids,
                    resize_list,
                    original_size_list,
                    max_new_tokens=512,
                    tokenizer=tokenizer,
                )
                

                output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
                
                text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
                text_output = text_output.replace("\n", "").replace("  ", " ")

                for p, pred_mask in enumerate(pred_masks):
                    if pred_mask.shape[0] == 0:
                        continue
                    
                    pred_mask = pred_mask.detach().cpu().numpy()[0]
                    pred_mask = pred_mask > 0
                    
                    if save_pred_img:   
                        save_path = f"{pred_dir}/{i}/{j}.png"
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        cv2.imwrite(save_path, pred_mask * 100)
                        # os.system(f"aws s3 cp {save_path} s3://mbz-hpc-aws-master/AROARU6TOWKRU3FNVE2PB:Amrin.Kareem@mbzuai.ac.ae/PartLISA/{save_path} --region me-central-1")
                    
                    predictions.append({"image_id" : i,
                                "category_id" : j,
                                "mask" : pred_mask})
            
                   
    if save_pred_json:
        with open("%s/pred.json" % pred_dir, "w") as outfile:
            json.dump(predictions, outfile)
    
    
    
    return predictions
