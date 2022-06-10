(login-jina)=
# Login to Jina ecosystem

Since Finetuner leverages cloud resources for fine-tuning,
you are required to login and obtain a token from Jina before starting a fine-tuning job.
It is as simple as:

```python
import finetuner

finetuner.login()
```

A browser window should pop-up with different login options.
After login you will see this in your terminal:

```bash
🔐 Successfully login to Jina Ecosystem!
```

```{admonition} Why do I need to login?
:class: hint
Login is required since Finetuner needs to push your `DocumentArray` into the cloud as training data.
Once you have successfully logged in, your training data will be linked to your personal user profile and will only be visible to you.

Once fine-tuning is completed, the fine-tuned model will be visible only to you in the cloud.
```