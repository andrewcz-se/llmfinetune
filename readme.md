## Fine Tuning a Local LLM Model

In some of my previous projects I have experimented with **FHIR** (Sunrise FHIR Viewer that provides basic management and viewing of FHIR data, along with Clinical AI Summaries) and **RAG Retrieval** (creating a custom PDF parser, ingesting it into a local LLM instance and then letting a user search the documents) so next i wanted to try fine tuning a local LLM for a related high precision extraction task.

The goal was to transform unstructured clinical narrative text into strictly formatted, FHIR like JSON. For this initial project I chose a custom schema the LLM had never seen before, a fictitious schema for my “Sunrise EMR”. Given the hardware available this project used **QLoRA (Quantised Low-Rank Adaptation)** to put complex proprietary schemas and logic rules directly into the model weights, achieving higher reliability and lower latency than Zero Shot approaches. I had tried using Zero Shot and prompt engineering to get the LLM in my RAG Retrieval project to output specific schemas from information in my PDF files that had been ingested, and whilst it did work quite successfully had several issues:

* **Hallucination:** The base models often invent fields (e.g., extracting symptoms when the schema requires reasonCode).  
* **Verbosity:** The base models tend to be 'chatty' wrapping JSON in conversational text, breaking downstream parsers. The smaller Qwen models I use are often affected by this. 

Lastly, I wanted to test the capabilities of Groq's Speech to Text models, specifically Whisper. Whilst not true Realtime Speech to Text, the standard modes are extremely quick. This project offered a good use case for dictation as in real world scenarios clinicians and healthcare workers would expect dictation to record clinical notes, and would not expect to have to type in their text. So, in the manual testing web page application I have included a dictate button. This records the users input and sends it to Whisper via Groq's AI SDK and returns the text. This is suprisingly fast, and results are returned almost instantly once the recording is stopped.

### My Approach

For this project I chose to use Qwen again (Qwen/Qwen2.5-3B-Instruct). Realistically it is the smallest model I can run and train locally whilst still retaining significant reasoning capabilities. It also has some good pre-trained coding/JSON capabilities, making it ideal for transfer learning.

Still, to make the training possible on my hardware, the base model is loaded in **4-bit (NF4)** precision, freezing the weights and reducing VRAM usage from ~6GB to ~2.5GB.

I used **LoRA (Low-Rank Adaptation)** to attach small, trainable rank decomposition matrices to the attention layers. Only these adapters (approx. 1-2% of total parameters) are updated during training.

As mentioned above, to prove the model wasn't just leveraging pre-trained knowledge, I engineered a synthetic dataset with specific traps and rules that a base model would never guess:

* **The JSON Schema:** I defined a custom resourceType: 'SunriseEMRClinicalSummary'. If the model outputs standard FHIR Resource or JSON schema, I know the fine-tuning failed.  
* **Strict Logic Rules:**  
  * **Unit Separation:** Meds doses 500mg must be split into {'value': 500, 'unit': 'mg'}.  
  * **Null Handling:** If a patient's age is not explicitly mentioned, the field ageInt must be null.   
* **Data Leakage Prevention:**  
  * *Training Set:* Diabetes, Hypertension, Metformin.  
  * *Validation Set:* COVID-19, Fractures, Paxlovid.  
  * *Result:* This proves the model learned the **structure extraction task**, not just the specific medical terms.

### Training Data (generate_dataset.py)

I built a custom Python script (**generate_dataset.py**) to create synthetic data. It created unstructured clinical notes with varying levels of noise (e.g., 'Pt John presents with...' vs. 'History: 55yo male...'). As mentioned above these jsonl files contained strict logic rules and data to prevent data leakage. The training data consisted of 500 records, 450 training items and 50 records used in the validation steps.

* **train.jsonl:**  
  * **Operation:** Used during the backward_pass.  
  * **Mechanism:** The model predicts the next token, calculates the error (Loss), and updates the **Adapter Weights** via backpropagation.  
  * **Content:** Restricted to specific clinical concepts (e.g., Diabetes, Hypertension) to establish baseline pattern recognition.  
* **val.jsonl:**  
  * **Operation:** Used during the evaluation_loop (triggered every 50 steps).  
  * **Mechanism:** The model predicts tokens, but **no weight updates occur**. It is purely a read-only check to measure generalisation error.  
  * **Content:** Contains **Hold-Out Concepts** (e.g., COVID-19, Fractures) that the model *never* sees in the training set.  
  * **Success Metric:** If the model successfully extracts a 'Fracture' (which it never studied) using the JSON structure it learned from 'Diabetes', it should have achieved **Transfer Learning**.

### Training (train_lora.py)

To fit the **Qwen 2.5 3B** model into my **GTX 1070 Ti (8GB VRAM)**, I utilised 4-bit quantisation (NF4) and trained **LoRA Adapters:**

| Parameter | Value | Technical Justification |
| :---- | :---- | :---- |
| **LoRA Rank (r)** | 16 | A balance point. r=8 is often too low for complex schema learning; r=64 increases VRAM usage with diminishing returns. 16 provides sufficient capacity for syntax adaptation. |
| **LoRA Alpha** | 32 | Set to 2x Rank. To ensure the updates from the adapter are strong enough to influence the frozen base model weights. |
| **Target Modules** | q_proj, v_proj... | I targeted all linear layers in the attention blocks in an attempt for better adherence to strict syntax rules. |
| **Dropout** | 0.05 | A 5% dropout rate prevents overfitting on the small (500 sample) dataset. |
| **Batch Size** | 2 | Constrained by my 8GB VRAM. |
| **Gradient Accumulation** | 4 | Since Batch Size was 2, I used accumulation to simulate a Batch Size of 8. This stabilises the gradient updates, preventing the training from being too 'noisy' |

Training on my **GTX 1070 Ti** presented some unique challenges due to its lack of **Gradient Scaling** support. During training I was always stopped by errors from AMP: *RuntimeError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'*. To get around this lack of support for BF16 and FP16 on my old GPU, there were two adjustments I had to make:

* **Turn off Gradient Scaling:** I explicitly disabled AMP (Automatic Mixed Precision), fp16 and bf16 modes, forcing the training to run in FP32 (Full Precision / 32-bit Float) mode. Since the adapters are tiny (Rank 16 is only ~50MB of parameters) although storing them in 32-bit instead of 16-bit takes longer, it only adds a few Mb of VRAM, and given the model was in 4-bit quantisation the main issue of VRAM overhead was still fine.  
* **Eager Attention:** I disabled Flash Attention 2 (which uses FP16/BF16 and requires newer Ampere and above chipsets) and forced attn_implementation='eager' to ensure stability.

### Evaluation Results (evaluate_and_plot.py)

I developed a Python script (**evaluate_and_plot.py**) to run tests between the base model and my tuned adapter, and to plot a Loss Curve to chart the training outcome. The script loads the Base Model, then dynamically merges the LoRA weights into memory. It feeds unstructured text prompts without the JSON schema definition and measures outcomes, for example:

* Did it generate valid JSON?  
* Did it output the SunriseEMRClinicalExtract schema?  
* Did it output null for missing ages?  
* Did it correctly add repetitions for multiple medications?

For the Loss Curve the model reads the val.jsonl conversation, at every token, it assigns a probability to what comes next. The loss is calculated, if the model predicts a token but the file contains a different value the loss spikes.

### Results & Conclusion

**Loss Curve:**

![Loss Curve](/results/loss_curve.png "Loss Curve")

* **Loss Curve:** The model showed a sharp descent from **Loss 1.0 to 0.1** within the first epoch. This should indicate the task (my JSON formatting) is highly learnable.  
* **Validation:** The validation loss tracked closely with training loss, confirming that the model generalises to unseen diseases (e.g., Pneumonia) without overfitting to the training diseases (e.g., Diabetes). If Validation Loss had spiked (for example to 0.5), it would indicate **Overfitting** (e.g. memorising the specific patients or conditions in train.jsonl).

**JSON Output:**

The trained model was able to successfully create a valid JSON schema from all the evaluation prompts and their scenarios. I am still running further tests but from the training so far it is able to parse the unstructured prompt into a successful schema output, with a wide variety of differing text input, for example: 

* *Pt John Doe, 55y, history of T2DM. Meds: Metformin 500mg.*  
* *Patient Sarah Connor presents with severe anxiety. Rx: Diazepam 5mg.*  
* *Pt Paul Bunyon, 48 yrs, reports fever. Prescribe: Ibuprofen 400mg and Paracetamol 500mg.*  
* *Dave Smith, 28y, reports headache. Rx: Codeine 400mg and Paracetamol 500mg.*  
* *Pete Jones (68), presents diarrhea. Dx gastroenteritis. Prescribe loperamide 20mg, Gaviscon 250ml, omprazol 250mg.*

### Conclusion

Although this is a basic first attempt at fine tuning, it has been a valuable learning experience with the process on my local device. It demonstrates that a 3B parameter model, when fine tuned with constraint based data, can create usable output on specific enterprise tasks with a very small hardware platform. By using a quantised model with QLoRA, although slowish (45-60 mins to train each time) this was achieved on basically obsolete hardware.
