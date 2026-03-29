import os, re
import time
import json, argparse
from load_longvideobench import LongVideoBenchDataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from call_gpt4o import request
from utils import dump_jsonl


# Global variable for video_data
video_data = LongVideoBenchDataset(os.getenv('LVB_PATH'), "lvb_test_wo_gt.json", max_num_frames=128)


PROMPTS = {
    "role": """**Remember: You are watching a Video.**

A user, characterized by a specific persona, is interacting with two AI assistant models (A and B) to better understand video content using the same question. Here is the user's persona:
```persona
{persona}
```

The user's question is:
```question
{question}
```

The response from Model A is:
```model_a
{answer_a}
```

The response from Model B is:
```model_b
{answer_b}
```

Please act as an impartial judge and carefully evaluate the responses of Model A and Model B to determine which one is better. Use the following standards:

1. [Instruction Following]: The response should closely adhere to the user's instructions, ensuring it directly addresses the specified task.
2. [Accuracy]: The response must accurately utilize information from the video, avoiding fabrication or misquotation. It should maintain factual correctness, avoid hallucinations, and demonstrate contextual coherence with precise terminology and knowledge.
3. [Relevance]: The response should consider the user's background information and needs, providing a comprehensive, detailed answer that addresses the question directly without straying off-topic. Responses should be thorough, offering multiple perspectives where relevant.
4. [Helpfulness]: The response should provide valuable information to aid the user in understanding or solving their issue, avoiding irrelevant or vague content.

If the responses from Model A and Model B are of similar quality (whether both are good or both are bad), you may declare a tie.

**Please follow these steps for your judgment:**

- Step 1: Analyze which model provides a better response for the [Instruction Following] standard.
- Step 2: Analyze which model provides a better response for the [Accuracy] standard.
- Step 3: Analyze which model provides a better response for the [Relevance] standard.
- Step 4: Analyze which model provides a better response for the [Helpfulness] standard.
- St