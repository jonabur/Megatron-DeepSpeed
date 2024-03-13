# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import sys
import json
import requests
import os

if __name__ == "__main__":
    url = sys.argv[1]
    url = 'http://' + url + '/api'
    headers = {'Content-Type': 'application/json'}
    samples=[
        """\"\"\"\nWrite a function to convert the given binary tuple to integer.\nassert binary_to_integer((1, 1, 0, 1, 0, 0, 1)) == '105'\n\"\"\"\n""",
        """"\"\"\"\nWrite a function to convert the given binary tuple to integer.\nassert binary_to_integer((1, 1, 0, 1, 0, 0, 1)) == '105'\n\"\"\"\n""",
        """\"\"\"\nWrite a python function to find the cube sum of first n natural numbers.\nassert sum_Of_Series(5) == 225\n\"\"\"\n""",
        """"\"\"\"\nWrite a python function to find the cube sum of first n natural numbers.\nassert sum_Of_Series(5) == 225\n\"\"\"\n""",
        ]
    repeats = 3
    for sentence in samples:
        for i in range(repeats):
            #sentence = input("Enter prompt: ")
            
            #tokens_to_generate = int(eval(input("Enter number of tokens to generate: ")))
            tokens_to_generate = 200

            data = {"prompts": [sentence], "tokens_to_generate": tokens_to_generate}
            response = requests.put(url, data=json.dumps(data), headers=headers)

            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.json()['message']}")
            else:
                print("Megatron Response: ")
                print(response.json()['text'][0], flush=True)
