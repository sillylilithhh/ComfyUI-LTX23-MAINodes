# MAINodes for LTX2.3/2.5

Experimental custom nodes for LTX 2.3/2.5. Dramatically reduce smearing in LTX 2.3/2.5 outputs. All credits goes to matlowai [for the initial implementation](https://github.com/matlowai/ComfyUI-MAINodes), I just converted it to work with LTX 2.3/2.5.


# Installation


```
cd custom_nodes/

git clone https://github.com/sillylilithhh/ComfyUI-LTX23-MAINodes
```

# Comfy UI Workflow

My workflow that I use, which is an Image to Video workflow for LTX 2.5, uses the MSR lora and its accompanying custom nodes. It is located in the `workflows` directory. Beware, there is quite a bit of spaghetti, and a lot of sampling passes. This was only really meant for myself and it shows. Tried to neat it up a bit. Adjust the workflow as you see fit.
