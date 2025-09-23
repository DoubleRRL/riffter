---
base_model: microsoft/DialoGPT-small
library_name: peft
pipeline_tag: text-generation
tags:
- comedy
- humor
- cum-town
- nick-mullen
- dialogue
- lora
- peft
- transformers
---

# Riffter: Cum Town Comedy Bot

A LoRA fine-tuned DialoGPT-small model trained on Cum Town podcast transcripts to generate Nick Mullen-style comedy riffs and jokes.

## Model Details

### Model Description

This model was trained from Microsoft's DialoGPT-small using LoRA (Low-Rank Adaptation) on 26,264 chunks of Cum Town podcast dialogue. The result is a comedy bot that captures the show's signature style: deep cuts, absurd connections, raw humor, and that "wrong but feels right" logic that Nick Mullen is known for.

- **Developed by:** DoubleRRL (trained on M2 Air)
- **Model type:** LoRA fine-tuned DialoGPT-small
- **Language(s):** English
- **License:** MIT
- **Base model:** microsoft/DialoGPT-small

### Model Sources

- **Repository:** https://github.com/DoubleRRL/riffter
- **Demo:** Run locally with the Riffter web app

## Uses

### Direct Use

This model generates edgy comedy riffs and jokes in the style of Cum Town. Best used for:
- Comedy writing and ideation
- Entertainment and humor generation
- Creative writing prompts
- Comedy analysis and study

### Example Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "DoubleRRL/riffter-cumtown-lora")

# Generate comedy
prompt = "What's the deal with people who wear socks with sandals?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, temperature=0.8, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Out-of-Scope Use

- Generic conversational AI
- Factual information or advice
- Safe-for-work content (this model generates adult comedy)
- Any use requiring factual accuracy

## Bias, Risks, and Limitations

This model was trained on adult comedy content that includes profanity, controversial topics, and edgy humor. It may generate inappropriate or offensive content.

### Recommendations

- Use for entertainment purposes only
- Be aware of the source material's adult nature
- Don't rely on this model for factual information
- Consider content warnings when sharing generated material

## How to Get Started with the Model

1. Install dependencies: `pip install transformers peft torch`
2. Load the model as shown in the usage example above
3. Start generating comedy with Cum Town-style prompts

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

[More Information Needed]

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

[More Information Needed]


#### Training Hyperparameters

- **Training regime:** [More Information Needed] <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

[More Information Needed]

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

[More Information Needed]

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

[More Information Needed]

### Results

[More Information Needed]

#### Summary



## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** [More Information Needed]
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications [optional]

### Model Architecture and Objective

[More Information Needed]

### Compute Infrastructure

[More Information Needed]

#### Hardware

[More Information Needed]

#### Software

[More Information Needed]

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

[More Information Needed]

## Model Card Contact

[More Information Needed]
### Framework versions

- PEFT 0.17.1