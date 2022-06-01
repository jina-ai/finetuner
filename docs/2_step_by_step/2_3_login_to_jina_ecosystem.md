# Login to Jina ecosystem

Since Finetuner leverages cloud resources for Fine-tuning.
Before starting a fine-tuning job,
it is required to login and obtain a token from Jina.
It is as simple as:

```python
import finetuner

finetuner.login()
```

A browser window should pop-up with different login options,
After login you will see this in your terminal:

```bash
🔐 Successfully login to Jina Ecosystem!
```

```{admonition} Why I need to login?
:class: hint
Login is required since Finetuner needs to push your `DocumentArray` into the cloud as training data.
Once you succesfuuly logged in, your training data will be linked to your personal user profile, and only visible for you. 

Once fine-tuning is ready, the fine-tuned model will be only visiable from you.
```